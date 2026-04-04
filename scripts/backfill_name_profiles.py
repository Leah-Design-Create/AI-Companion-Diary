# -*- coding: utf-8 -*-
"""对已存在的旧聊天记录，增量补写“姓名/称呼 profile 片段”到 Chroma。

用于修复：历史里用户说过名字，但在接入长期记忆后模型跨会话不一定能直接召回姓名。
"""

from __future__ import annotations

import argparse
import asyncio

import aiosqlite

from config import DB_PATH
from services.long_memory import add_user_name_profiles_with_msg_id


async def run_backfill(
    *,
    limit: int,
    since_id: int,
    user_id: int | None,
) -> None:
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        where = ["m.id > ?", "m.role = 'user'"]
        params: list = [since_id]
        if user_id is not None:
            where.append("s.user_id = ?")
            params.append(user_id)

        sql = f"""
            SELECT m.id AS msg_id, s.user_id AS user_id, m.session_id AS session_id,
                   m.role AS role, m.content AS content, m.created_at AS created_at
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.id ASC
            LIMIT ?
        """
        params.append(limit)
        cursor = await conn.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        if not rows:
            print("没有可补写的姓名资料。")
            return

        ok = 0
        total_profiles = 0
        skipped = 0
        for idx, r in enumerate(rows, start=1):
            added = await add_user_name_profiles_with_msg_id(
                msg_id=int(r["msg_id"]),
                user_id=int(r["user_id"]),
                session_id=int(r["session_id"]),
                role=str(r["role"] or "user"),
                content=r["content"] or "",
                created_at=r["created_at"],
            )
            if added > 0:
                ok += 1
                total_profiles += added
            else:
                skipped += 1
            if idx % 50 == 0:
                print(f"进度: {idx}/{len(rows)} | 有效msg {ok} | profiles {total_profiles} | 跳过 {skipped}")

        print(f"完成: msgs={len(rows)} | 有效msgs={ok} | profiles_added={total_profiles} | 跳过={skipped}")
        print("提示: 下次可用 --since-id 继续增量。")
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="补写姓名 profile 到 Chroma 长期记忆")
    parser.add_argument("--limit", type=int, default=2000, help="本次最多处理多少条消息（默认 2000）")
    parser.add_argument("--since-id", type=int, default=0, help="仅处理消息ID大于该值（默认 0）")
    parser.add_argument("--user-id", type=int, default=None, help="仅处理指定 user_id（可选）")
    args = parser.parse_args()

    asyncio.run(run_backfill(limit=max(1, args.limit), since_id=max(0, args.since_id), user_id=args.user_id))


if __name__ == "__main__":
    main()

