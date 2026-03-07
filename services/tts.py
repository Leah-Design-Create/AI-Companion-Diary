# -*- coding: utf-8 -*-
"""通义千问 TTS（DashScope qwen3-tts-flash）封装

注意：DashScope 的 voice 参数通常要求英文 voice_id（如 Cherry）。
若传中文（如“芊悦”），部分 SDK/requests 版本会在构造 HTTP Header 时触发编码错误。
"""
import base64
import os

import dashscope
from dashscope import MultiModalConversation
from fastapi import HTTPException
import httpx

from config import DASHSCOPE_API_KEY, DASHSCOPE_TTS_MODEL, DASHSCOPE_TTS_VOICE


def _ensure_api_key() -> str:
    """确保 DashScope API Key 已配置，并设置到 SDK。"""
    api_key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 TTS 密钥。请在 .env 中设置 DASHSCOPE_API_KEY。",
        )
    dashscope.api_key = api_key
    return api_key


def _find_audio_obj(obj):
    """在 DashScope 返回体中尽量找到 audio 对象(dict)。"""
    # 对象（SDK 返回值）转 dict 递归查找
    if not isinstance(obj, (dict, list)) and hasattr(obj, "__dict__"):
        try:
            return _find_audio_obj(vars(obj))
        except Exception:  # noqa: BLE001
            pass

    if isinstance(obj, dict):
        a = obj.get("audio")
        if isinstance(a, dict):
            return a
        # 有些返回把 audio 放在 content 列表里：{"content":[{"audio":{...}}]}
        for v in obj.values():
            found = _find_audio_obj(v)
            if found:
                return found

    if isinstance(obj, list):
        for it in obj:
            found = _find_audio_obj(it)
            if found:
                return found
    return None


async def synthesize_to_mp3(text: str) -> bytes:
    """调用 DashScope 的 qwen3-tts-flash，将文本转为 mp3 二进制数据。"""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="TTS 文本不能为空")

    _ensure_api_key()
    model = DASHSCOPE_TTS_MODEL or "qwen3-tts-flash"
    voice_raw = (DASHSCOPE_TTS_VOICE or "Cherry").strip()
    voice_alias = {
        "芊悦": "Cherry",  # 中文别名 → 英文 voice_id
    }
    voice = voice_alias.get(voice_raw, voice_raw)
    if any(ord(ch) > 127 for ch in voice):
        raise HTTPException(
            status_code=400,
            detail=f"voice 参数需使用英文 voice_id（例如 Cherry）。当前: {voice_raw}",
        )

    try:
        # 使用 MultiModalConversation 调用 Qwen-TTS
        resp = MultiModalConversation.call(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": text},
                    ],
                }
            ],
            audio={"voice": voice, "format": "mp3"},
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TTS 调用失败: {e}")

    # 文档：音频数据位于 output.audio.data（base64）。部分场景只返回 output.audio.url
    try:
        if isinstance(resp, dict):
            output = resp.get("output", resp)
        else:
            output = getattr(resp, "output", None) or resp
            # 某些 SDK 返回对象，需要转 dict 才能递归查找
            if not isinstance(output, (dict, list)) and hasattr(output, "__dict__"):
                try:
                    output = vars(output)
                except Exception:  # noqa: BLE001
                    pass

        audio_obj = _find_audio_obj(output)
        if not isinstance(audio_obj, dict):
            raise ValueError("未在响应中找到音频对象 audio（可能是接口未返回音频字段）")

        data_b64 = audio_obj.get("data")
        if data_b64:
            return base64.b64decode(data_b64)

        url = audio_obj.get("url")
        if url:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.content

        raise ValueError("未在响应中找到音频数据 output.audio.data 或 output.audio.url")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TTS 解析失败: {e}")


