# -*- coding: utf-8 -*-
"""通义千问 TTS（DashScope qwen3-tts-flash）封装"""
import asyncio
import base64
import os
import re

import dashscope
import httpx
from dashscope.audio.qwen_tts import SpeechSynthesizer
from fastapi import HTTPException

from config import DASHSCOPE_API_KEY, DASHSCOPE_TTS_MODEL, DASHSCOPE_TTS_VOICE

_VOICE_ALIAS = {"芊悦": "Cherry"}

# 同时只允许 1 个 TTS 请求，避免 DashScope 因并发关闭连接（10054）
_tts_sem = asyncio.Semaphore(1)


def _get_api_key() -> str:
    key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="未配置 DASHSCOPE_API_KEY")
    return key


async def synthesize_to_mp3(text: str) -> bytes:
    text = (text or "").strip()
    text = re.sub(r'\[NEXT\]', '', text).strip()
    text = re.sub(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', '', text)  # 纯标点清空
    if not text:
        raise HTTPException(status_code=400, detail="TTS 文本不能为空")

    api_key = _get_api_key()
    model = DASHSCOPE_TTS_MODEL or "qwen3-tts-flash"
    voice = _VOICE_ALIAS.get((DASHSCOPE_TTS_VOICE or "Cherry").strip(), DASHSCOPE_TTS_VOICE or "Cherry")

    loop = asyncio.get_event_loop()

    async with _tts_sem:  # 串行化，避免并发触发连接重置
        last_err = None
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(0.6 * attempt)  # 重试前等待
            try:
                dashscope.api_key = api_key
                resp = await loop.run_in_executor(None, lambda: SpeechSynthesizer.call(
                    model=model,
                    api_key=api_key,
                    text=text,
                    voice=voice,
                    format="mp3",
                ))
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[TTS] 调用异常(第{attempt+1}次): {type(e).__name__}: {e}")

        if last_err:
            raise HTTPException(status_code=502, detail=f"TTS 调用失败: {last_err}")

    status_code = getattr(resp, "status_code", None)
    if status_code and status_code != 200:
        msg = getattr(resp, "message", "") or getattr(resp, "code", "") or str(resp)
        print(f"[TTS] API 错误: status={status_code} msg={msg}")
        raise HTTPException(status_code=502, detail=f"TTS API 错误 {status_code}: {msg}")

    try:
        output = getattr(resp, "output", None)
        audio = output.audio if output else None
        data_b64 = audio.get("data", "") if isinstance(audio, dict) else ""
        url = audio.get("url", "") if isinstance(audio, dict) else (audio if isinstance(audio, str) else "")
    except Exception as e:
        print(f"[TTS] 解析响应失败: {e} | resp={resp}")
        raise HTTPException(status_code=502, detail=f"TTS 解析响应失败: {e}")

    if data_b64:
        return base64.b64decode(data_b64)

    if url:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    raise HTTPException(status_code=502, detail="TTS 响应中无音频数据")
