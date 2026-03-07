# -*- coding: utf-8 -*-
"""意图识别：判断用户是想闲聊还是问知识库，再决定是否做 RAG"""
import os
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from prompts import INTENT_CLASSIFY


def _client():
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
        base_url=OPENAI_BASE_URL,
    )


async def detect_intent(user_message: str) -> str:
    """返回 'ask_kb'（问知识库）或 'chat'（闲聊）。失败或未配置 API 时默认返回 'chat'。"""
    msg = (user_message or "").strip()
    if not msg or not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        return "chat"
    prompt = INTENT_CLASSIFY + msg[:500]
    try:
        r = await _client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        out = (r.choices[0].message.content or "").strip()
        if "问知识库" in out or "知识库" in out:
            return "ask_kb"
        return "chat"
    except Exception:
        return "chat"
