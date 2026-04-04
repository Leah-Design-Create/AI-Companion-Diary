# -*- coding: utf-8 -*-
"""把 SQLite 里的旧聊天记录回填到 Chroma 长期记忆。"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiosqlite

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_PATH
from services.long_memory import add_message_with_msg_id


async def run_backfill(limit: int, since_id: int, user_id: int | None):
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        where = ["m.id > ?"]
        params: list = [since_id]
        if user_id is not None:
            where.append("s.user_id = ?")
            params.append(user_id)
        where_sql = " AND ".join(where)
        sql = f"""
            SELECT m.id AS msg_id, s.user_id AS user_id, m.session_id AS session_id,
                   m.role AS role, m.content AS content, m.created_at AS created_at
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            ORDER BY m.id ASC
            LIMIT ?
        """
        params.append(limit)
        cursor = await conn.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        if not rows:
            print("没有可回填的历史消息。")
            return
        ok = 0
        skip = 0
        fail = 0
        for idx, r in enumerate(rows, start=1):
            content = (r["content"] or "").strip()
            if not content:
                skip += 1
                continue
            done = await add_message_with_msg_id(
                msg_id=int(r["msg_id"]),
                user_id=int(r["user_id"]),
                session_id=int(r["session_id"]),
                role=(r["role"] or "user"),
                content=content,
                created_at=r["created_at"],
            )
            if done:
                ok += 1
            else:
                fail += 1
            if idx % 20 == 0:
                print(f"进度: {idx}/{len(rows)} | 成功 {ok} | 失败 {fail} | 跳过 {skip}")
        last_id = int(rows[-1]["msg_id"])
        print(f"回填完成: 总计 {len(rows)} | 成功 {ok} | 失败 {fail} | 跳过 {skip} | 最后消息ID {last_id}")
        print("提示: 再次回填可加参数 --since-id 上次最后消息ID")
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="回填旧聊天记录到 Chroma 长期记忆")
    parser.add_argument("--limit", type=int, default=500, help="本次最多处理多少条消息（默认 500）")
    parser.add_argument("--since-id", type=int, default=0, help="仅处理消息ID大于该值（默认 0）")
    parser.add_argument("--user-id", type=int, default=None, help="仅回填指定用户ID（可选）")
    args = parser.parse_args()
    asyncio.run(run_backfill(limit=max(1, args.limit), since_id=max(0, args.since_id), user_id=args.user_id))


if __name__ == "__main__":
    main()
