# -*- coding: utf-8 -*-
"""调用 OpenAI 兼容的 Embedding API，用于 RAG 语义检索"""
import os
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL

# 单次请求最大 token 约 8k，中文按约 1.5 字/token，截断到约 6000 字
MAX_TEXT_LEN = 6000


def _client():
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
        base_url=OPENAI_BASE_URL,
    )


async def get_embedding(text: str) -> list[float] | None:
    """把一段文本转成向量。未配置 OPENAI_EMBEDDING_MODEL 或调用失败时返回 None。"""
    if not OPENAI_EMBEDDING_MODEL:
        return None
    if not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        return None
    s = (text or "").strip()[:MAX_TEXT_LEN]
    if not s:
        return None
    try:
        r = await _client().embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=s)
        if r.data and len(r.data) > 0:
            return r.data[0].embedding
    except Exception as e:
        print(f"[Embedding] API 调用失败: {e}")
    return None
