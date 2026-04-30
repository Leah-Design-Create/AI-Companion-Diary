# -*- coding: utf-8 -*-
"""DashScope Paraformer 语音识别封装"""
import asyncio
import io

from fastapi import HTTPException
from openai import AsyncOpenAI

from config import DASHSCOPE_API_KEY, OPENAI_API_KEY

_stt_sem = asyncio.Semaphore(2)


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    api_key = DASHSCOPE_API_KEY or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=503, detail="未配置 DASHSCOPE_API_KEY")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    async with _stt_sem:
        result = await client.audio.transcriptions.create(
            model="paraformer-realtime-v2",
            file=(filename, io.BytesIO(audio_bytes)),
        )

    return result.text or ""
