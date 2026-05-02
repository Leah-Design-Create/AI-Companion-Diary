# -*- coding: utf-8 -*-
"""DashScope Paraformer 语音识别封装（ffmpeg 转码 → WAV → Paraformer）"""
import asyncio
import os
import subprocess
import tempfile

import dashscope
from fastapi import HTTPException

from config import DASHSCOPE_API_KEY, OPENAI_API_KEY

_stt_sem = asyncio.Semaphore(2)


def _to_wav(audio_bytes: bytes) -> bytes:
    """用 ffmpeg 把任意格式音频转为 16kHz mono WAV"""
    proc = subprocess.run(
        ['ffmpeg', '-y', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'wav', 'pipe:1'],
        input=audio_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors='ignore')[:300])
    return proc.stdout


def _recognize_sync(wav_path: str) -> str:
    from dashscope.audio.asr import Recognition
    resp = Recognition(
        model='paraformer-realtime-v2',
        format='wav',
        sample_rate=16000,
        callback=None,
    ).call(wav_path)
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
    loop = asyncio.get_event_loop()

    try:
        wav_bytes = await loop.run_in_executor(None, _to_wav, audio_bytes)
    except Exception as e:
        print(f"[STT] 音频转换失败: {e}", flush=True)
        raise HTTPException(status_code=400, detail=f"音频格式转换失败: {e}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        async with _stt_sem:
            text = await loop.run_in_executor(None, _recognize_sync, tmp_path)
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
