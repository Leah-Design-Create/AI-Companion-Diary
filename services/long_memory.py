# -*- coding: utf-8 -*-
"""长期记忆：使用 ChromaDB 存储并检索历史对话片段。"""
from __future__ import annotations

import threading
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    LONG_MEMORY_MAX_CHARS,
    LONG_MEMORY_TOP_K,
)
from services.embedding import get_embedding

try:
    import chromadb
except Exception:
    chromadb = None


_LOCK = threading.Lock()
_CLIENT = None
_COLLECTION = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    return s[:limit]


def _normalize_for_compare(text: str) -> str:
    s = (text or "").strip().lower()
    # 去掉常见标点/空白，避免"同一句不同标点"重复命中。
    s = re.sub(r"[\s\.,!?;:，。！？；：、\"'""‘’（）()\[\]【】<>《》]+", "", s)
    return s


def _extract_user_name_profiles(text: str) -> list[tuple[str, str]]:
    """从用户文本中抽取可用于"姓名/称呼"回忆的结构化片段。

    返回: [(doc_text, kind)]
    """
    s = (text or "").strip()
    if not s:
        return []

    # 常见自述：我叫张三 / 名字是张三 / 我姓李
    m1 = re.search(r"(我叫|我名叫|我叫做|名字是)\s*([^\s,，。！？!?:：]{1,12})", s)
    if m1 and m1.group(2):
        return [(f"用户姓名：{m1.group(2)}", "profile_name")]

    m2 = re.search(r"(我姓)\s*([^\s,，。！？!?:：]{1,6})", s)
    if m2 and m2.group(2):
        return [(f"用户姓氏：{m2.group(2)}", "profile_surname")]

    return []


def _is_name_query(q: str) -> bool:
    s = (q or "").strip()
    if not s:
        return False
    return any(k in s for k in ("名字", "称呼", "怎么叫", "你叫", "我叫", "我叫什么", "你还记得我"))


def _get_collection():
    global _CLIENT, _COLLECTION
    if chromadb is None:
        return None
    if _COLLECTION is not None:
        return _COLLECTION
    with _LOCK:
        if _COLLECTION is not None:
            return _COLLECTION
        path = str(Path(CHROMA_DIR))
        _CLIENT = chromadb.PersistentClient(path=path)
        _COLLECTION = _CLIENT.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"description": "长期对话记忆（用户+助手历史片段）"},
        )
        return _COLLECTION


async def add_message_to_memory(
    *,
    user_id: int,
    session_id: int,
    role: str,
    content: str,
    created_at: str | None = None,
    item_id: str | None = None,
    _profile_source: str | None = None,
) -> bool:
    """将一条消息写入 Chroma。若 embedding 不可用或未安装 chromadb，则静默跳过。
    _profile_source: 若指定，则用此文本做姓名 profile 提取；传空字符串则跳过提取。
    """
    c = _get_collection()
    text = _truncate(content, LONG_MEMORY_MAX_CHARS)
    if c is None or not text:
        return False
    emb = await get_embedding(text)
    if not emb:
        return False
    doc_base = item_id or f"{user_id}:{session_id}:{role}:{uuid.uuid4().hex}"
    try:
        c.add(
            ids=[doc_base],
            documents=[text],
            embeddings=[emb],
            metadatas=[{
                "user_id": int(user_id),
                "session_id": int(session_id),
                "role": role,
                "created_at": created_at or _now_iso(),
            }],
        )
        # 结构化写入姓名/称呼片段：只从纯用户原文提取，避免误匹配助手回复里的词。
        profile_text = _profile_source if _profile_source is not None else (text if role == "user" else "")
        if profile_text:
            profiles = _extract_user_name_profiles(profile_text)
            for i, (pdoc, kind) in enumerate(profiles):
                try:
                    pemb = await get_embedding(pdoc)
                    if not pemb:
                        continue
                    c.add(
                        ids=[f"{doc_base}:profile:{kind}:{i}"],
                        documents=[pdoc],
                        embeddings=[pemb],
                        metadatas=[{
                            "user_id": int(user_id),
                            "session_id": int(session_id),
                            "role": role,
                            "kind": kind,
                            "created_at": created_at or _now_iso(),
                        }],
                    )
                except Exception as e:
                    print(f"[LongMemory] add profile 失败: {e}")

        return True
    except Exception as e:
        print(f"[LongMemory] add 失败: {e}")
        return False


async def add_message_with_msg_id(
    *,
    msg_id: int,
    user_id: int,
    session_id: int,
    role: str,
    content: str,
    created_at: str | None = None,
) -> bool:
    """用稳定的消息 ID 写入，适合历史回填避免重复。"""
    return await add_message_to_memory(
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content,
        created_at=created_at,
        item_id=f"msg:{int(msg_id)}",
    )


async def add_user_name_profiles_with_msg_id(
    *,
    msg_id: int,
    user_id: int,
    session_id: int,
    role: str,
    content: str,
    created_at: str | None = None,
) -> int:
    """只补写姓名/称呼类 profile 片段（不写整段原文），用于对已存在的旧消息做增量修复。"""
    if role != "user":
        return 0
    c = _get_collection()
    if c is None:
        return 0
    text = _truncate(content, LONG_MEMORY_MAX_CHARS)
    profiles = _extract_user_name_profiles(text)
    if not profiles:
        return 0

    doc_base = f"msg:{int(msg_id)}"
    added = 0
    for i, (pdoc, kind) in enumerate(profiles):
        try:
            emb = await get_embedding(pdoc)
            if not emb:
                continue
            pid = f"{doc_base}:profile:{kind}:{i}"
            meta = {
                "user_id": int(user_id),
                "session_id": int(session_id),
                "role": role,
                "kind": kind,
                "created_at": created_at or _now_iso(),
            }
            if hasattr(c, "upsert"):
                c.upsert(ids=[pid], documents=[pdoc], embeddings=[emb], metadatas=[meta])
            else:
                c.add(ids=[pid], documents=[pdoc], embeddings=[emb], metadatas=[meta])
            added += 1
        except Exception as e:
            print(f"[LongMemory] add name profile 失败: {e}")
    return added


async def add_turn_to_memory(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_reply: str,
    created_at: str | None = None,
) -> bool:
    """将一轮完整对话（用户消息 + 助手回复）作为一个单元写入长期记忆。
    相比分条存储，保留了上下文，召回时能看到完整的一问一答。
    """
    max_each = max(50, (LONG_MEMORY_MAX_CHARS - 20) // 2)
    user_part = _truncate(user_message, max_each)
    assistant_part = _truncate(assistant_reply, max_each)
    if not user_part:
        return False
    combined = f"用户：{user_part}\n小伴：{assistant_part}" if assistant_part else f"用户：{user_part}"
    item_id = f"turn:{user_id}:{session_id}:{uuid.uuid4().hex}"
    return await add_message_to_memory(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=combined,
        created_at=created_at,
        item_id=item_id,
        _profile_source=user_part,  # 只从用户原文提取姓名，不扫描助手回复
    )


async def retrieve_relevant_memories(
    *,
    user_id: int,
    query: str,
    limit: int | None = None,
    exclude_session_id: int | None = None,
) -> list[str]:
    """按用户检索相关历史对话片段。"""
    c = _get_collection()
    if c is None:
        return []
    q = (query or "").strip()
    if not q:
        return []
    q_norm = _normalize_for_compare(q)
    emb = await get_embedding(q)
    if not emb:
        return []
    n = max(1, int(limit or LONG_MEMORY_TOP_K))
    is_name_q = _is_name_query(q)
    try:
        candidates: list[tuple[float, str]] = []
        seen: set[str] = set()

        # 名字查询：先直接取出所有 profile 条目，强制加入候选（不依赖向量排名）
        if is_name_q:
            try:
                profile_result = c.get(
                    where={"$and": [{"user_id": int(user_id)}, {"kind": "profile_name"}]},
                    include=["documents", "metadatas"],
                )
                for pdoc, pmeta in zip(profile_result.get("documents") or [], profile_result.get("metadatas") or []):
                    if not pdoc:
                        continue
                    doc_text = str(pdoc).strip()
                    text = f"用户：{doc_text}"
                    if text not in seen:
                        seen.add(text)
                        candidates.append((-1.0, text))  # 最优先
            except Exception as e:
                print(f"[LongMemory] profile 直接查询失败: {e}")

        result = c.query(
            query_embeddings=[emb],
            n_results=n * 6 if exclude_session_id else n * 4,
            where={"user_id": int(user_id)},
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        dists = result.get("distances") or []
        if not docs:
            return [t for _, t in sorted(candidates)[:n]]
        for idx, doc in enumerate(docs[0]):
            if not doc:
                continue
            doc_text = str(doc).strip()
            if not doc_text:
                continue
            # 避免把"当前提问本身"当作记忆召回，浪费召回槽位。
            if _normalize_for_compare(doc_text) == q_norm:
                continue
            meta = metas[0][idx] if metas and metas[0] and idx < len(metas[0]) else {}
            if exclude_session_id is not None and int(meta.get("session_id", -1)) == int(exclude_session_id):
                continue
            role = str(meta.get("role", "user"))
            prefix = "用户" if role == "user" else "小伴"
            text = f"{prefix}：{doc_text}"
            if text in seen:
                continue
            seen.add(text)
            dist = 9e9
            if dists and dists[0] and idx < len(dists[0]):
                try:
                    dist = float(dists[0][idx])
                except Exception:
                    pass
            # 偏向保留用户自己说过的话，跨会话"记住我"场景更稳。
            if role == "user":
                dist -= 0.08
            candidates.append((dist, text))
        candidates.sort(key=lambda x: x[0])
        out = [t for _, t in candidates[:n]]
        return out
    except Exception as e:
        print(f"[LongMemory] query 失败: {e}")
        return []


# ---------- 用户画像提取 + 情绪检测（合并为一次后台 LLM 调用）----------

_ANALYZE_SYSTEM = """分析用户这条消息，完成以下两件事：
1. 调用 save_profiles：提取用户明确透露的个人信息（姓名、偏好、经历、情绪状态），没有则传空数组
2. 调用 report_emotion：判断用户当前情绪（2-4个字）和是否存在焦虑/压力

规则：
- 只提取用户明确说出的内容，不要推断
- name：用户说了自己的名字（"我叫X""我是X"）
- preference：用户提到的喜好（食物、饮料、爱好等）
- experience：用户提到的近期经历或重要事件
- feeling：用户明确表达的持续性情绪
- 情绪词示例：平静、开心、焦虑、难过、疲惫、烦躁、委屈、期待、放松"""

_SAVE_PROFILES_TOOL = {
    "type": "function",
    "function": {
        "name": "save_profiles",
        "description": "保存从用户消息中提取的个人信息，没有信息时传空数组",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["name", "preference", "experience", "feeling"]},
                            "content": {"type": "string"},
                        },
                        "required": ["kind", "content"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}

_REPORT_EMOTION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_emotion",
        "description": "报告用户当前情绪状态",
        "parameters": {
            "type": "object",
            "properties": {
                "mood": {"type": "string", "description": "2-4个字的情绪词，如：平静、焦虑、开心、难过"},
                "anxiety": {"type": "boolean", "description": "是否存在明显焦虑或压力"},
            },
            "required": ["mood", "anxiety"],
        },
    },
}

_KIND_LABEL = {
    "name": "姓名",
    "preference": "偏好",
    "experience": "经历",
    "feeling": "情绪状态",
}


async def extract_and_save_profiles(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    db_path: str | None = None,
) -> tuple[int, str, bool]:
    """后台合并调用：提取用户画像 + 检测情绪，一次 LLM 完成。
    返回 (保存的 profile 条数, mood 词, anxiety 布尔值)。
    """
    s = (user_message or "").strip()
    if not s or len(s) < 2:
        return 0, "平静", False

    # 跳过明确无意义的应答词，节省 LLM 调用
    _TRIVIAL = {
        "嗯", "嗯嗯", "嗯嗯嗯", "嗯？", "啊", "哦", "哦哦", "哦？",
        "好", "好的", "好吧", "好啊", "好好", "好了", "行", "行吧", "行的",
        "知道", "知道了", "知道了哦", "明白", "明白了", "懂了", "懂",
        "谢谢", "谢", "谢了", "感谢", "多谢",
        "ok", "okay",
        "哈哈", "哈哈哈", "哈哈哈哈", "呵呵", "嘻嘻", "哈",
        "是", "是的", "是啊", "是哦", "对", "对的", "对啊", "对哦",
        "然后", "然后呢", "继续", "说下去",
        "没事", "没关系", "不用了", "算了",
        "?", "？", "...", "……",
    }
    if s.lower() in _TRIVIAL:
        return 0, "平静", False

    try:
        from openai import AsyncOpenAI
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        import os, json as _json
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL,
        )
        r = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": s},
            ],
            tools=[_SAVE_PROFILES_TOOL, _REPORT_EMOTION_TOOL],
            tool_choice="required",
            temperature=0,
            max_tokens=512,
            parallel_tool_calls=True,
        )
    except Exception as e:
        print(f"[AnalyzeTurn] LLM 调用失败: {e}")
        return 0, "平静", False

    msg = r.choices[0].message
    tool_calls = msg.tool_calls or []

    profile_items: list[dict] = []
    mood = "平静"
    anxiety = False

    for tc in tool_calls:
        try:
            args = _json.loads(tc.function.arguments)
        except Exception:
            continue
        if tc.function.name == "save_profiles":
            profile_items = args.get("items") or []
        elif tc.function.name == "report_emotion":
            mood = (args.get("mood") or "平静").strip()[:8]
            anxiety = bool(args.get("anxiety", False))

    # 保存 profile 到 ChromaDB
    c = _get_collection()
    saved = 0
    if c is not None:
        for item in profile_items:
            kind = str(item.get("kind", "")).strip()
            content = str(item.get("content", "")).strip()
            if not kind or not content:
                continue
            doc = f"用户{_KIND_LABEL.get(kind, kind)}：{content}"
            emb = await get_embedding(doc)
            if not emb:
                continue
            pid = f"profile:{user_id}:{session_id}:{kind}:{uuid.uuid4().hex}"
            try:
                c.add(
                    ids=[pid],
                    documents=[doc],
                    embeddings=[emb],
                    metadatas=[{
                        "user_id": int(user_id),
                        "session_id": int(session_id),
                        "role": "user",
                        "kind": kind,
                        "created_at": _now_iso(),
                    }],
                )
                saved += 1
                print(f"[Profile] 保存 {kind}: {content}")
            except Exception as e:
                print(f"[Profile] 存储失败: {e}")

    print(f"[AnalyzeTurn] mood={mood} anxiety={anxiety} profiles={saved}")

    # 把情绪写回 sessions 表
    if db_path:
        try:
            import aiosqlite
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "UPDATE sessions SET mood = ?, anxiety_detected = ? WHERE id = ?",
                    (mood, 1 if anxiety else 0, int(session_id)),
                )
                await conn.commit()
        except Exception as e:
            print(f"[AnalyzeTurn] 写回 session 失败: {e}")

    return saved, mood, anxiety
