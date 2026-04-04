# -*- coding: utf-8 -*-
"""RAG：语义检索（向量相似度）优先，无向量时回退到关键词检索"""
import json
import math
import aiosqlite
from config import DB_PATH

from services.embedding import get_embedding

try:
    import jieba
    _JIEBA_OK = True
    jieba.setLogLevel(60)  # 关掉 jieba 初始化日志
except ImportError:
    _JIEBA_OK = False

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


_KW_STOPWORDS = frozenset({
    "想", "知道", "是", "的", "你", "我", "有", "吗", "什么", "怎么", "如何", "哪些",
    "告诉", "认识", "谁", "了", "在", "也", "都", "就", "但", "和", "或", "与",
    "一", "一个", "一些", "这", "那", "这个", "那个", "他", "她", "它", "们",
    "为", "为什么", "因为", "所以", "可以", "能", "会", "要", "不", "没有",
})


def _extract_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """从用户输入中提取检索用关键词。优先使用 jieba 分词，回退到滑动窗口。"""
    s = (text or "").strip()
    if not s:
        return []
    if _JIEBA_OK:
        words = jieba.cut(s)
        seen: set[str] = set()
        out: list[str] = []
        for w in words:
            w = w.strip()
            if len(w) >= 2 and w not in _KW_STOPWORDS and w not in seen:
                seen.add(w)
                out.append(w)
                if len(out) >= max_keywords:
                    break
        return out
    # 回退：空格分词 + 2-3 字滑动窗口（原实现）
    s = s.replace("，", " ").replace("。", " ").replace("！", " ").replace("？", " ")
    parts = [p.strip() for p in s.split() if len(p.strip()) >= 2]
    seen2: set[str] = set()
    out2: list[str] = []
    for p in parts:
        if p not in seen2 and len(p) <= 20:
            seen2.add(p)
            out2.append(p)
        if len(p) >= 3:
            for n in (3, 2):
                for i in range(0, len(p) - n + 1):
                    w = p[i: i + n]
                    if w not in seen2:
                        seen2.add(w)
                        out2.append(w)
                        if len(out2) >= max_keywords:
                            return out2[:max_keywords]
    return out2[:max_keywords]


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


async def _knowledge_by_semantic(conn, user_input: str, limit: int, keywords: list[str]) -> tuple[list[str], float]:
    """用查询向量与知识库 embedding 做余弦相似度，返回 (最相关的 limit 条, 最高相似度)。失败或无向量时返回 ([], 0.0)。"""
    query_emb = await get_embedding(user_input)
    if not query_emb:
        return [], 0.0
    try:
        cursor = await conn.execute(
            """SELECT title, content, embedding FROM knowledge WHERE embedding IS NOT NULL AND embedding != ''"""
        )
        rows = await cursor.fetchall()
    except Exception as e:
        print(f"[RAG] 语义检索跳过（表无 embedding 或查询失败）: {e}")
        return [], 0.0
    if not rows:
        print("[RAG] 语义检索未命中：知识库暂无向量，请到 /upload 重新上传文档以生成 embedding")
        return [], 0.0
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
    best_score = scored[0][0] if scored else 0.0
    return [s for _, s in scored[:limit]], best_score


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


async def get_relevant_context(user_input: str, user_id: int = 1, limit: int = 8) -> tuple[list[str], str, float]:
    """优先语义检索（需配置 OPENAI_EMBEDDING_MODEL 且知识库有 embedding），否则用关键词检索。
    返回 (片段列表, "semantic"|"keyword", 最佳相关度分数)。
    语义模式：分数为余弦相似度 [0,1]；关键词模式：分数为最佳匹配关键词数。
    """
    if not user_input.strip():
        return [], "keyword", 0.0
    keywords = _extract_keywords(user_input)
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    mode = "keyword"
    best_score = 0.0
    try:
        # 1) 先尝试语义检索
        out, best_score = await _knowledge_by_semantic(conn, user_input, limit, keywords)
        if out:
            mode = "semantic"
        else:
            # 2) 回退到关键词检索
            kw_results = await _knowledge_by_keywords(conn, keywords, limit)
            out = [s for _, s in kw_results]
            best_score = float(kw_results[0][0]) if kw_results else 0.0

        return out[:limit], mode, best_score
    finally:
        await conn.close()
