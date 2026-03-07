# -*- coding: utf-8 -*-
"""RAG：语义检索（向量相似度）优先，无向量时回退到关键词检索"""
import json
import math
import aiosqlite
from config import DB_PATH

from services.embedding import get_embedding

# 单条片段最大字符数（「有哪些」「几种」等问列表时需足够长才能包含完整列举）
SNIPPET_MAX_LEN = 4500


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度，范围约 [-1, 1]。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _extract_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """从用户输入中提取检索用关键词，兼容中文（无空格）：用 2～3 字滑动窗口切分。"""
    s = (text or "").replace("，", " ").replace("。", " ").replace("！", " ").replace("？", " ").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split() if len(p.strip()) >= 2]
    seen = set()
    out = []
    for p in parts:
        if p not in seen and len(p) <= 20:
            seen.add(p)
            out.append(p)
        if len(p) >= 3:
            for n in (3, 2):
                for i in range(0, len(p) - n + 1):
                    w = p[i : i + n]
                    if w not in seen:
                        seen.add(w)
                        out.append(w)
                        if len(out) >= max_keywords:
                            return out[:max_keywords]
    return out[:max_keywords]


def _score_snippet(snippet: str, keywords: list[str]) -> int:
    """片段与关键词的匹配次数，用于按相关度排序。"""
    if not keywords:
        return 0
    lower = snippet.lower()
    return sum(1 for kw in keywords if kw in snippet or kw.lower() in lower)


def _best_snippet(content: str, keywords: list[str], max_len: int) -> str:
    """从 content 中截取最相关的一段：优先包含关键词出现位置，否则取开头。避免答案在文档后面被截掉。"""
    if not (content or "").strip():
        return ""
    if not keywords or max_len <= 0:
        return (content or "")[:max_len]
    text = content.strip()
    if len(text) <= max_len:
        return text
    # 优先用最长关键词定位（如「广泛性焦虑」比「焦虑」更准），再以其为中心取一段
    sorted_kw = sorted([k for k in keywords if k and k.strip()], key=len, reverse=True)
    first_pos = -1
    for kw in sorted_kw:
        p = text.find(kw)
        if p >= 0:
            first_pos = p
            break
    if first_pos < 0:
        return text[:max_len]
    # 关键词前留约 300 字，后面多留（列表、定义多在关键词后面）
    before = 300
    start = max(0, first_pos - before)
    end = min(len(text), first_pos + (max_len - before))
    if end - start < max_len and start > 0:
        start = max(0, end - max_len)
    return text[start:end]


async def _knowledge_by_semantic(conn, user_input: str, limit: int, keywords: list[str]) -> list[str]:
    """用查询向量与知识库 embedding 做余弦相似度，返回最相关的 limit 条。失败或无向量时返回 []。"""
    query_emb = await get_embedding(user_input)
    if not query_emb:
        return []
    try:
        cursor = await conn.execute(
            """SELECT title, content, embedding FROM knowledge WHERE embedding IS NOT NULL AND embedding != ''"""
        )
        rows = await cursor.fetchall()
    except Exception as e:
        print(f"[RAG] 语义检索跳过（表无 embedding 或查询失败）: {e}")
        return []
    if not rows:
        print("[RAG] 语义检索未命中：知识库暂无向量，请到 /upload 重新上传文档以生成 embedding")
        return []
    scored: list[tuple[float, str]] = []
    for r in rows:
        try:
            raw = r["embedding"]
            emb = json.loads(raw) if raw else None
        except (KeyError, TypeError, ValueError):
            continue
        if not emb:
            continue
        sim = _cosine(query_emb, emb)
        title = r["title"] or ""
        raw_content = r["content"] or ""
        part = _best_snippet(raw_content, keywords, SNIPPET_MAX_LEN) or raw_content[:SNIPPET_MAX_LEN]
        snippet = (title + "\n" + part).strip()
        if snippet:
            scored.append((sim, snippet))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:limit]]


async def _knowledge_by_keywords(conn, keywords: list[str], limit: int) -> list[tuple[int, str]]:
    """按关键词 LIKE 检索知识库，返回 (相关度分, 片段) 列表。片段围绕关键词位置截取，而非只取文档开头。"""
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for kw in keywords:
        cursor = await conn.execute(
            """SELECT title, content FROM knowledge
               WHERE title LIKE ? OR content LIKE ?
               ORDER BY id DESC LIMIT 3""",
            (f"%{kw}%", f"%{kw}%"),
        )
        for r in await cursor.fetchall():
            title = r["title"] or ""
            raw_content = r["content"] or ""
            part = _best_snippet(raw_content, keywords, SNIPPET_MAX_LEN) or raw_content[:SNIPPET_MAX_LEN]
            snippet = (title + "\n" + part).strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            scored.append((_score_snippet(snippet, keywords), snippet))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:limit]


async def get_relevant_context(user_input: str, user_id: int = 1, limit: int = 8) -> tuple[list[str], str]:
    """优先语义检索（需配置 OPENAI_EMBEDDING_MODEL 且知识库有 embedding），否则用关键词检索。返回 (片段列表, "semantic"|"keyword")。"""
    if not user_input.strip():
        return [], "keyword"
    keywords = _extract_keywords(user_input)
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    mode = "keyword"
    try:
        # 1) 知识库：先尝试语义检索，没有再关键词（都按「含关键词的片段」截取，避免只看到文档开头）
        out: list[str] = await _knowledge_by_semantic(conn, user_input, limit, keywords)
        if out:
            mode = "semantic"
        else:
            kw_results = await _knowledge_by_keywords(conn, keywords, limit)
            out = [s for _, s in kw_results]
        if not out and keywords:
            cursor = await conn.execute(
                """SELECT title, content FROM knowledge ORDER BY id DESC LIMIT 2"""
            )
            for r in await cursor.fetchall():
                title = r["title"] or ""
                raw = r["content"] or ""
                part = _best_snippet(raw, keywords, SNIPPET_MAX_LEN) or raw[:SNIPPET_MAX_LEN]
                snippet = (title + "\n" + part).strip()
                if snippet and snippet not in out:
                    out.append(snippet)

        # 只返回知识库内容，不混入「近期对话总结」——总结由模型之前生成，可能含错误（如 5-4-3-2-1 法），不能当事实依据
        return out[:limit], mode
    finally:
        await conn.close()
