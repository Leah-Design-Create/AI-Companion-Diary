# -*- coding: utf-8 -*-
"""DashScope Paraformer 语音识别封装"""
import asyncio
import os
import tempfile

import dashscope
from fastapi import HTTPException

from config import DASHSCOPE_API_KEY, OPENAI_API_KEY

_stt_sem = asyncio.Semaphore(2)
_SUPPORTED_FMTS = {'pcm', 'wav', 'mp3', 'mp4', 'm4a', 'ogg', 'webm', 'opus', 'amr'}


def _recognize_sync(audio_path: str, fmt: str) -> str:
    from dashscope.audio.asr import Recognition
    resp = Recognition(
        model='paraformer-realtime-v2',
        format=fmt,
        sample_rate=16000,
    ).call(audio_path)
    if getattr(resp, 'status_code', None) != 200:
        code = getattr(resp, 'status_code', '?')
        msg = getattr(resp, 'message', '') or getattr(resp, 'code', '') or str(resp)
        raise RuntimeError(f"DashScope STT {code}: {msg}")
    output = getattr(resp, 'output', {}) or {}
    sentences = output.get('sentence', []) if isinstance(output, dict) else (getattr(output, 'sentence', []) or [])
    return ''.join(
        (s.get('text', '') if isinstance(s, dict) else getattr(s, 'text', ''))
        for s in sentences
    )


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    api_key = DASHSCOPE_API_KEY or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=503, detail="未配置 DASHSCOPE_API_KEY")

    dashscope.api_key = api_key
    ext = (filename.rsplit('.', 1)[-1] if '.' in filename else 'webm').lower()
    fmt = ext if ext in _SUPPORTED_FMTS else 'webm'

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        loop = asyncio.get_event_loop()
        async with _stt_sem:
            text = await loop.run_in_executor(None, _recognize_sync, tmp_path, fmt)
        return text
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STT] 识别失败: {e}", flush=True)
        raise HTTPException(status_code=502, detail=f"STT 识别失败: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
