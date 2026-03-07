# -*- coding: utf-8 -*-
"""SQLite 数据库：用户、会话、消息、总结、知识库"""
import aiosqlite
from pathlib import Path

from config import DB_PATH


async def get_db():
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row
    return conn


async def init_db():
    conn = await get_db()
    try:
        await conn.executescript("""
        -- 用户（简化：单用户也可用，多用户用 user_id 区分）
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            last_login_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- 会话（每次打开对话可视为一个会话，用于生成总结）
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            started_at TEXT DEFAULT (datetime('now')),
            ended_at TEXT,
            summary TEXT,
            anxiety_detected INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- 消息
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        -- 知识库（RAG：外接文章、笔记等）
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT NOT NULL,
            source_url TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            embedding TEXT
        );

        -- 久未登录时 agent 的「分享」记录（可选，用于去重）
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        await conn.commit()
        # 为旧库补充 embedding 列（语义检索用）
        try:
            await conn.execute("ALTER TABLE knowledge ADD COLUMN embedding TEXT")
            await conn.commit()
        except Exception:
            pass
        # 为旧库补充 sessions.title（会话重命名用）
        try:
            await conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
            await conn.commit()
        except Exception:
            pass
        # 为旧库补充 messages.image_path（用户分享图片）
        try:
            await conn.execute("ALTER TABLE messages ADD COLUMN image_path TEXT")
            await conn.commit()
        except Exception:
            pass
        # 为旧库补充 sessions.mood（日记卡片显示当时心情）
        try:
            await conn.execute("ALTER TABLE sessions ADD COLUMN mood TEXT")
            await conn.commit()
        except Exception:
            pass
    finally:
        await conn.close()


async def ensure_user(user_id: int = 1):
    conn = await get_db()
    try:
        await conn.execute(
            "INSERT OR IGNORE INTO users (id, name, last_login_at) VALUES (?, ?, datetime('now'))",
            (user_id, "用户"),
        )
        await conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user_id,),
        )
        await conn.commit()
    finally:
        await conn.close()
