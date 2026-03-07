# -*- coding: utf-8 -*-
"""当前心情识别：基于最近几条对话调用 LLM 判断用户情绪"""
from services.llm import chat
from prompts import MOOD_ANALYSIS


async def analyze_mood(messages_for_analysis: list[dict]) -> str:
    """根据最近对话内容判断用户当前心情，返回 2～4 字词语。"""
    if not messages_for_analysis:
        return "平静"
    text = "\n".join(
        f"{m.get('role', '')}: {(m.get('content') or '')[:500]}"
        for m in messages_for_analysis
    )
    prompt = MOOD_ANALYSIS.strip() + "\n\n对话：\n" + text[:2500]
    try:
        out = await chat([{"role": "user", "content": prompt}])
        out = (out or "").strip()
        for ch in ("。", "，", "\n", " ", "　"):
            out = out.replace(ch, "")
        if not out:
            return "平静"
        return out[:8]
    except Exception:
        return "平静"
