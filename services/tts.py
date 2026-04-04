# -*- coding: utf-8 -*-
"""通义千问 TTS（DashScope qwen3-tts-flash）封装"""
import os

import dashscope
from dashscope.audio.qwen_tts import SpeechSynthesizer
from fastapi import HTTPException

from config import DASHSCOPE_API_KEY, DASHSCOPE_TTS_MODEL, DASHSCOPE_TTS_VOICE

_VOICE_ALIAS = {"芊悦": "Cherry"}


def _get_api_key() -> str:
    key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="未配置 DASHSCOPE_API_KEY")
    return key


async def synthesize_to_mp3(text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="TTS 文本不能为空")

    api_key = _get_api_key()
    dashscope.api_key = api_key

    model = DASHSCOPE_TTS_MODEL or "qwen3-tts-flash"
    voice_raw = (DASHSCOPE_TTS_VOICE or "Cherry").strip()
    voice = _VOICE_ALIAS.get(voice_raw, voice_raw)

    try:
        resp = SpeechSynthesizer.call(
            model=model,
            api_key=api_key,
            text=text,
            voice=voice,
            format="mp3",
        )
    except Exception as e:
        print(f"[TTS] 调用异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"TTS 调用失败: {e}")

    try:
        audio = resp.output.audio
        data_b64 = audio.get("data", "") if isinstance(audio, dict) else ""
        url = audio.get("url", "") if isinstance(audio, dict) else ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS 解析响应失败: {e}")

    if data_b64:
        import base64
        return base64.b64decode(data_b64)

    if url:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    raise HTTPException(status_code=502, detail="TTS 响应中无音频数据")
