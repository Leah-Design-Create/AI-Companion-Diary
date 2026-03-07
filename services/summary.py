# -*- coding: utf-8 -*-
"""会话结束后生成生活记录总结"""
from services.llm import chat
from prompts import SUMMARY


async def generate_summary(messages: list[dict]) -> str:
    """根据本轮对话生成简短总结。"""
    if not messages:
        return ""
    text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in messages
    )
    prompt = SUMMARY.strip() + "\n\n对话：\n" + text[:3000]
    return (await chat([{"role": "user", "content": prompt}])).strip()
