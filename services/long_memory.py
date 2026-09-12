# -*- coding: utf-8 -*-
"""长期记忆：使用 ChromaDB 存储并检索历史对话片段。"""
from __future__ import annotations

import hashlib
import threading
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DB_PATH,
    LONG_MEMORY_MAX_CHARS,
    LONG_MEMORY_MAX_DOCS,
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


def _should_analyze_for_memory(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 4:
        return False
    if _extract_user_name_profiles(s):
        return True
    signals = (
        "我叫", "我名叫", "名字是", "我姓", "我是",
        "生日", "家乡", "老家", "住在",
        "我喜欢", "我爱", "我讨厌", "我不喜欢", "我偏好",
        "我最近", "最近在", "这段时间", "一直在", "正在",
        "我爸", "我妈", "我朋友", "我同事", "我对象", "我男朋友", "我女朋友",
        "下周", "明天", "后天", "月底", "下个月", "以后提醒我", "下次问我",
        "准备", "面试", "考试", "考研", "工作", "项目", "搬家",
    )
    return any(k in s for k in signals)


def _normalize_memory_type(kind: str) -> str:
    mapping = {
        "name": "identity",
        "identity": "identity",
        "preference": "preference",
        "relationship": "relationship",
        "experience": "event",
        "event": "event",
        "feeling": "emotion_pattern",
        "emotion_pattern": "emotion_pattern",
        "followup": "followup",
    }
    return mapping.get((kind or "").strip(), "")


def _normalize_memory_content(text: str) -> str:
    s = re.sub(r"^用户(身份|姓名|姓氏|偏好|关系|经历|事件|情绪状态|情绪模式|待跟进)[:：]", "", text or "")
    return _normalize_for_compare(s)


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

        _trim_user_memory(user_id)
        return True
    except Exception as e:
        print(f"[LongMemory] add 失败: {e}")
        return False


def _trim_user_memory(user_id: int) -> None:
    """若该用户文档数超过上限，删除最旧的一批，保持库不无限膨胀。"""
    c = _get_collection()
    if c is None or LONG_MEMORY_MAX_DOCS <= 0:
        return
    try:
        result = c.get(
            where={"user_id": int(user_id)},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        if len(ids) <= LONG_MEMORY_MAX_DOCS:
            return
        metas = result.get("metadatas") or []
        pairs = sorted(zip(ids, metas), key=lambda x: x[1].get("created_at", ""))
        n_delete = len(ids) - LONG_MEMORY_MAX_DOCS
        ids_to_delete = [p[0] for p in pairs[:n_delete]]
        c.delete(ids=ids_to_delete)
        print(f"[LongMemory] trimmed {n_delete} old docs for user {user_id}")
    except Exception as e:
        print(f"[LongMemory] trim 失败: {e}")


async def _ensure_memories_schema(conn) -> None:
    await conn.executescript("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        source_message_id INTEGER,
        type TEXT NOT NULL,
        content TEXT NOT NULL,
        source_quote TEXT,
        scope TEXT DEFAULT 'stable',
        sensitivity TEXT DEFAULT 'low',
        confidence REAL DEFAULT 1.0,
        status TEXT DEFAULT 'active',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        expires_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_memories_user_status
        ON memories(user_id, status);
    CREATE INDEX IF NOT EXISTS idx_memories_user_type
        ON memories(user_id, type);
    """)


async def _index_memory_doc(
    *,
    memory_id: int,
    user_id: int,
    session_id: int | None,
    memory_type: str,
    content: str,
    created_at: str | None = None,
) -> bool:
    c = _get_collection()
    if c is None:
        return False
    doc = _truncate(content, LONG_MEMORY_MAX_CHARS)
    if not doc:
        return False
    emb = await get_embedding(doc)
    if not emb:
        return False
    meta = {
        "user_id": int(user_id),
        "memory_id": int(memory_id),
        "session_id": int(session_id or 0),
        "role": "user",
        "kind": memory_type,
        "status": "active",
        "created_at": created_at or _now_iso(),
    }
    try:
        cid = f"memory:{int(memory_id)}"
        if hasattr(c, "upsert"):
            c.upsert(ids=[cid], documents=[doc], embeddings=[emb], metadatas=[meta])
        else:
            try:
                c.delete(ids=[cid])
            except Exception:
                pass
            c.add(ids=[cid], documents=[doc], embeddings=[emb], metadatas=[meta])
        _trim_user_memory(user_id)
        return True
    except Exception as e:
        print(f"[LongMemory] index memory 失败: {e}")
        return False


async def _save_structured_memory(
    *,
    user_id: int,
    session_id: int,
    memory_type: str,
    content: str,
    source_quote: str = "",
    scope: str = "stable",
    sensitivity: str = "low",
    confidence: float = 1.0,
    expires_at: str | None = None,
    db_path: str | None = None,
) -> bool:
    import aiosqlite

    mtype = _normalize_memory_type(memory_type)
    text = _truncate(content, LONG_MEMORY_MAX_CHARS)
    if not mtype or not text:
        return False
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    if conf < 0.7:
        return False
    if sensitivity not in {"low", "medium", "high"}:
        sensitivity = "low"
    if sensitivity == "high":
        return False
    if scope not in {"stable", "recent", "temporary"}:
        scope = "stable"

    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        await _ensure_memories_schema(conn)

        norm = _normalize_memory_content(text)
        cursor = await conn.execute(
            """SELECT id, content FROM memories
               WHERE user_id = ? AND type = ? AND status = 'active'
               ORDER BY updated_at DESC LIMIT 50""",
            (int(user_id), mtype),
        )
        for row in await cursor.fetchall():
            if _normalize_memory_content(row["content"]) == norm:
                await conn.execute(
                    "UPDATE memories SET updated_at = datetime('now') WHERE id = ?",
                    (int(row["id"]),),
                )
                await conn.commit()
                await _index_memory_doc(
                    memory_id=int(row["id"]),
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=mtype,
                    content=text,
                )
                return False

        if mtype == "identity":
            await conn.execute(
                """UPDATE memories
                   SET status = 'superseded', updated_at = datetime('now')
                   WHERE user_id = ? AND type = 'identity' AND status = 'active'""",
                (int(user_id),),
            )

        cursor = await conn.execute(
            """INSERT INTO memories
               (user_id, session_id, type, content, source_quote, scope,
                sensitivity, confidence, status, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (
                int(user_id),
                int(session_id),
                mtype,
                text,
                _truncate(source_quote, LONG_MEMORY_MAX_CHARS),
                scope,
                sensitivity,
                conf,
                expires_at,
            ),
        )
        memory_id = int(cursor.lastrowid)
        await conn.commit()

    await _index_memory_doc(
        memory_id=memory_id,
        user_id=user_id,
        session_id=session_id,
        memory_type=mtype,
        content=text,
    )
    return True


async def add_message_with_msg_id(
    *,
    msg_id: int,
    user_id: int,
    session_id: int,
    role: str,
    content: str,
    created_at: str | None = None,
) -> bool:
    """Compatibility backfill hook.

    Full historical messages are no longer written to Chroma as long-term
    memory. For old backfills, only deterministic user identity snippets are
    promoted into structured memories.
    """
    added = await add_user_name_profiles_with_msg_id(
        msg_id=msg_id,
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content,
        created_at=created_at,
    )
    return added > 0


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
    text = _truncate(content, LONG_MEMORY_MAX_CHARS)
    profiles = _extract_user_name_profiles(text)
    if not profiles:
        return 0

    added = 0
    for pdoc, kind in profiles:
        try:
            ok = await _save_structured_memory(
                user_id=user_id,
                session_id=session_id,
                memory_type="identity",
                content=pdoc,
                source_quote=text,
                scope="stable",
                sensitivity="low",
                confidence=1.0,
            )
            if ok:
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
    """Compatibility hook: do not store full turns as long-term memory.

    Long-term memory is now written by extract_and_save_profiles() as
    structured memory rows in SQLite, with ChromaDB used only as an index.
    This hook keeps a cheap deterministic path for explicit name statements.
    """
    user_part = _truncate(user_message, LONG_MEMORY_MAX_CHARS)
    saved = False
    for doc, kind in _extract_user_name_profiles(user_part):
        saved = await _save_structured_memory(
            user_id=user_id,
            session_id=session_id,
            memory_type="identity" if kind == "profile_name" else "identity",
            content=doc,
            source_quote=user_part,
            scope="stable",
            sensitivity="low",
            confidence=1.0,
        ) or saved
    return saved


async def retrieve_relevant_memories(
    *,
    user_id: int,
    query: str,
    limit: int | None = None,
    exclude_session_id: int | None = None,
) -> list[str]:
    """按用户检索相关结构化长期记忆。"""
    q = (query or "").strip()
    if not q:
        return []
    n = max(1, int(limit or LONG_MEMORY_TOP_K))
    is_name_q = _is_name_query(q)
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    if is_name_q:
        try:
            import aiosqlite

            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                await _ensure_memories_schema(conn)
                cursor = await conn.execute(
                    """SELECT content FROM memories
                       WHERE user_id = ? AND type = 'identity'
                         AND status = 'active'
                         AND (expires_at IS NULL OR expires_at > datetime('now'))
                       ORDER BY updated_at DESC, id DESC
                       LIMIT 3""",
                    (int(user_id),),
                )
                rows = await cursor.fetchall()
            for row in rows:
                doc_text = str(row["content"] or "").strip()
                if not doc_text:
                    continue
                text = f"用户记忆：{doc_text}"
                if text not in seen:
                    seen.add(text)
                    candidates.append((-1.0, text))
        except Exception as e:
            print(f"[LongMemory] identity 直接查询失败: {e}")

    c = _get_collection()
    if c is None:
        return [t for _, t in sorted(candidates)[:n]]
    q_norm = _normalize_for_compare(q)
    emb = await get_embedding(q)
    if not emb:
        return [t for _, t in sorted(candidates)[:n]]
    try:
        import aiosqlite

        result = c.query(
            query_embeddings=[emb],
            n_results=n * 6 if exclude_session_id else n * 4,
            where={"$and": [{"user_id": int(user_id)}, {"status": "active"}]},
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        dists = result.get("distances") or []
        if not docs:
            return [t for _, t in sorted(candidates)[:n]]
        memory_hits: list[tuple[float, int]] = []
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
            memory_id = meta.get("memory_id")
            if memory_id is None:
                continue
            dist = 9e9
            if dists and dists[0] and idx < len(dists[0]):
                try:
                    dist = float(dists[0][idx])
                except Exception:
                    pass
            memory_hits.append((dist, int(memory_id)))
        if memory_hits:
            ids = []
            for _, mid in sorted(memory_hits, key=lambda x: x[0]):
                if mid not in ids:
                    ids.append(mid)
            placeholders = ",".join("?" for _ in ids)
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                await _ensure_memories_schema(conn)
                cursor = await conn.execute(
                    f"""SELECT id, type, content FROM memories
                        WHERE user_id = ? AND status = 'active'
                          AND (expires_at IS NULL OR expires_at > datetime('now'))
                          AND id IN ({placeholders})""",
                    (int(user_id), *ids),
                )
                rows = await cursor.fetchall()
            by_id = {int(r["id"]): r for r in rows}
            for dist, mid in sorted(memory_hits, key=lambda x: x[0]):
                row = by_id.get(mid)
                if not row:
                    continue
                text = f"用户记忆（{row['type']}）：{row['content']}"
                if text in seen:
                    continue
                seen.add(text)
                candidates.append((dist, text))
        candidates.sort(key=lambda x: x[0])
        out = [t for _, t in candidates[:n]]
        return out
    except Exception as e:
        print(f"[LongMemory] query 失败: {e}")
        return []


# ---------- 用户画像提取 + 情绪检测（合并为一次后台 LLM 调用）----------

_ANALYZE_SYSTEM = """分析用户这条消息，完成以下两件事：
1. 调用 save_profiles：只提取值得长期记住的结构化信息，没有则传空数组
2. 调用 report_emotion：判断用户当前情绪（2-4个字）和是否存在焦虑/压力

规则：
- 只提取用户明确说出的内容，不要推断
- 不保存寒暄、一次性闲聊、无具体对象的普通情绪词
- identity：姓名、称呼等稳定身份信息
- preference：稳定偏好（食物、饮料、爱好、习惯等）
- relationship：用户反复提及或重要的人际关系
- event：近期重要事件或阶段状态，如面试、考试、搬家、项目
- emotion_pattern：持续性情绪模式，不保存短暂情绪
- followup：适合后续主动追问的未完成事项
- should_store=false 表示不应写入长期记忆
- sensitivity=high 的内容默认不要保存，除非用户明确要求记住
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
                            "kind": {"type": "string", "enum": ["identity", "preference", "relationship", "event", "emotion_pattern", "followup", "name", "experience", "feeling"]},
                            "content": {"type": "string"},
                            "source_quote": {"type": "string"},
                            "scope": {"type": "string", "enum": ["stable", "recent", "temporary"]},
                            "sensitivity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "confidence": {"type": "number"},
                            "should_store": {"type": "boolean"},
                            "expires_at": {"type": "string", "description": "可选，YYYY-MM-DD；短期事项可设置过期时间"},
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
    "identity": "身份",
    "name": "姓名",
    "preference": "偏好",
    "relationship": "关系",
    "experience": "经历",
    "event": "事件",
    "feeling": "情绪状态",
    "emotion_pattern": "情绪模式",
    "followup": "待跟进",
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
    memory_signal = _should_analyze_for_memory(s)

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

    saved = 0
    for item in profile_items:
        if item.get("should_store") is False:
            continue
        kind = str(item.get("kind", "")).strip()
        mtype = _normalize_memory_type(kind)
        content = str(item.get("content", "")).strip()
        if not mtype or not content:
            continue
        if not memory_signal and mtype not in {"identity", "followup"}:
            continue
        doc = f"用户{_KIND_LABEL.get(mtype, mtype)}：{content}"
        try:
            ok = await _save_structured_memory(
                user_id=user_id,
                session_id=session_id,
                memory_type=mtype,
                content=doc,
                source_quote=str(item.get("source_quote") or user_message).strip(),
                scope=str(item.get("scope") or "stable").strip(),
                sensitivity=str(item.get("sensitivity") or "low").strip(),
                confidence=float(item.get("confidence", 1.0)),
                expires_at=(str(item.get("expires_at")).strip() or None) if item.get("expires_at") else None,
                db_path=db_path,
            )
            if ok:
                saved += 1
                print(f"[Memory] 保存 {mtype}: {content}")
        except Exception as e:
            print(f"[Memory] 存储失败: {e}")

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
