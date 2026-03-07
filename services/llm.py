# -*- coding: utf-8 -*-
"""调用 OpenAI 兼容 API"""
import os
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


def get_client():
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
        base_url=OPENAI_BASE_URL,
    )


async def chat(messages: list[dict], extra_system: str = "") -> str:
    client = get_client()
    system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    if extra_system:
        # 有 RAG 时把参考内容放最前面，让模型优先遵守「只依据参考回答」
        system = extra_system.rstrip() + "\n\n" + (system or "")
    elif system:
        system = system or ""
    if system and messages and messages[0].get("role") == "system":
        msgs = [{"role": "system", "content": system}] + messages[1:]
    elif system:
        msgs = [{"role": "system", "content": system}] + list(messages)
    else:
        msgs = list(messages)
    temp = 0.15 if extra_system else 0.8  # 有 RAG 时低温度，强制贴参考
    r = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=temp,
        max_tokens=1024,
    )
    return (r.choices[0].message.content or "").strip()


async def chat_stream(messages: list[dict], extra_system: str = ""):
    client = get_client()
    system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    if extra_system:
        system = extra_system.rstrip() + "\n\n" + (system or "")
    elif system:
        system = system or ""
    if system and messages and messages[0].get("role") == "system":
        msgs = [{"role": "system", "content": system}] + messages[1:]
    elif system:
        msgs = [{"role": "system", "content": system}] + list(messages)
    else:
        msgs = list(messages)
    temp = 0.15 if extra_system else 0.8
    stream = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=temp,
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
