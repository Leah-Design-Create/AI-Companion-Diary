# -*- coding: utf-8 -*-
"""配置：从环境变量读取 API 等"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# LLM：OpenAI 或兼容接口（如 DeepSeek、通义等）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 兼容第三方
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 语义检索（可选）：填了则 RAG 用向量相似度检索，否则仅用关键词
# 示例：OpenAI text-embedding-3-small；通义 text-embedding-v3
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "")

# 通义 TTS（DashScope）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_TTS_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "")
DASHSCOPE_TTS_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "")

# 应用
DB_PATH = os.getenv("DB_PATH", "companion.db")
INACTIVE_DAYS_FOR_REMINDER = int(os.getenv("INACTIVE_DAYS_FOR_REMINDER", "2"))
