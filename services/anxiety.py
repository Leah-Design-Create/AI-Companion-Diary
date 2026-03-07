# -*- coding: utf-8 -*-
"""焦虑/压力检测：基于当前会话对话内容调用 LLM 判断"""
from services.llm import chat
from prompts import ANXIETY_ANALYSIS


async def analyze_anxiety(messages_for_analysis: list[dict]) -> bool:
    """根据对话内容判断是否存在明显焦虑。messages 为 user/assistant 轮次。"""
    if not messages_for_analysis:
        return False
    text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in messages_for_analysis
    )
    prompt = ANXIETY_ANALYSIS.strip() + "\n\n对话：\n" + text[:2000]
    out = await chat([{"role": "user", "content": prompt}])
    return "是" in (out or "")
