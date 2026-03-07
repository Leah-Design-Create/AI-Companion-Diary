# -*- coding: utf-8 -*-
"""久未登录时生成 agent 的关心/分享话术"""
from datetime import datetime, timedelta
from services.llm import chat
from prompts import REMINDER_GREETING
from config import INACTIVE_DAYS_FOR_REMINDER


def parse_sqlite_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


async def get_reminder_if_inactive(last_login_at: str | None) -> str | None:
    """若距离上次登录已超过 INACTIVE_DAYS_FOR_REMINDER 天，返回一句小伴的问候；否则返回 None。"""
    if not last_login_at:
        return None
    last = parse_sqlite_datetime(last_login_at)
    if not last:
        return None
    if datetime.now() - last < timedelta(days=INACTIVE_DAYS_FOR_REMINDER):
        return None
    text = await chat([{"role": "user", "content": REMINDER_GREETING}])
    return (text or "").strip() or None
