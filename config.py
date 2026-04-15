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

# RAG 知识库上传上限（整文件读入内存；过大易 MemoryError，可改大或拆文件）
MAX_KNOWLEDGE_UPLOAD_BYTES = int(os.getenv("MAX_KNOWLEDGE_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_KNOWLEDGE_TEXT_CHARS = int(os.getenv("MAX_KNOWLEDGE_TEXT_CHARS", str(2_000_000)))

# 长期记忆（ChromaDB）
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "chat_long_memory")
LONG_MEMORY_TOP_K = int(os.getenv("LONG_MEMORY_TOP_K", "4"))
LONG_MEMORY_MAX_CHARS = int(os.getenv("LONG_MEMORY_MAX_CHARS", "700"))

# 通义 TTS（DashScope）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_TTS_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "")
DASHSCOPE_TTS_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "")

# 心情画文生图（DashScope 或其他兼容接口）
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "")
IMAGE_GEN_SIZE = os.getenv("IMAGE_GEN_SIZE", "512*512")

# 评估模型（LLM-as-Judge，默认 DeepSeek，也兼容其他 OpenAI 兼容接口）
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")
EVAL_BASE_URL = os.getenv("EVAL_BASE_URL", "https://api.deepseek.com")
EVAL_MODEL = os.getenv("EVAL_MODEL", "deepseek-chat")

# 兼容旧版 Gemini 变量名（若有旧 .env 则自动回退）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")

# 应用
DB_PATH = os.getenv("DB_PATH", "companion.db")
INACTIVE_DAYS_FOR_REMINDER = int(os.getenv("INACTIVE_DAYS_FOR_REMINDER", "2"))
