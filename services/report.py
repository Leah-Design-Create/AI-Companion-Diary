# -*- coding: utf-8 -*-
"""情绪周报：聚合一周情绪数据，生成高光时刻与成长建议。"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

import aiosqlite

from config import DB_PATH
from services.llm import chat

# ---------- 情绪词 → 数值（1-10）----------
_MOOD_SCORE: dict[str, int] = {
    "兴奋": 9, "开心": 8, "愉快": 8, "高兴": 8, "喜悦": 8,
    "放松": 7, "安心": 7, "满足": 7, "轻松": 7,
    "平和": 6, "平静": 5,
    "无聊": 4, "困惑": 4,
    "疲惫": 3, "烦躁": 3, "委屈": 3,
    "低落": 2, "焦虑": 2, "难过": 2, "压力": 2, "紧张": 2,
    "恐惧": 1, "绝望": 1, "崩溃": 1,
}
_DEFAULT_SCORE = 5

_MOOD_CATEGORY: dict[str, str] = {
    "兴奋": "positive", "开心": "positive", "愉快": "positive",
    "高兴": "positive", "喜悦": "positive", "放松": "positive",
    "安心": "positive", "满足": "positive", "轻松": "positive",
    "平和": "neutral", "平静": "neutral",
    "无聊": "neutral", "困惑": "neutral",
    "疲惫": "negative", "烦躁": "negative", "委屈": "negative",
    "低落": "negative", "焦虑": "negative", "难过": "negative",
    "压力": "negative", "紧张": "negative", "恐惧": "negative",
    "绝望": "negative", "崩溃": "negative",
}


def _mood_score(mood: str) -> int:
    return _MOOD_SCORE.get((mood or "").strip(), _DEFAULT_SCORE)


def _mood_category(mood: str) -> str:
    return _MOOD_CATEGORY.get((mood or "").strip(), "neutral")


# ---------- 关键词提取（无需 jieba）----------
_STOPWORDS = frozenset({
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "和", "也", "都", "就", "但", "而", "或", "与", "一", "个",
    "说", "到", "会", "能", "要", "不", "没", "很", "都", "还", "然后",
    "如果", "因为", "所以", "虽然", "但是", "今天", "昨天", "明天",
    "时候", "什么", "怎么", "为什么", "可以", "一些", "这样", "那么",
    "感觉", "觉得", "知道", "一直", "已经", "其实", "可能", "应该",
})


def _extract_keywords(texts: list[str], top_n: int = 30) -> list[dict]:
    """从文本列表中提取高频双/三字词作为关键词云数据。"""
    freq: dict[str, int] = {}
    for text in texts:
        text = re.sub(r"[^\u4e00-\u9fff]", " ", text or "")
        words = text.split()
        for w in words:
            for n in (2, 3):
                for i in range(len(w) - n + 1):
                    chunk = w[i:i + n]
                    if chunk not in _STOPWORDS and len(chunk) >= 2:
                        freq[chunk] = freq.get(chunk, 0) + 1
    sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
    # 过滤频次过低的
    return [{"word": w, "weight": c} for w, c in sorted_kw if c >= 2][:top_n]


# ---------- 周报数据聚合 ----------
async def get_weekly_report(user_id: int, week_offset: int = 0) -> dict:
    """
    聚合最近一周（或 week_offset 周前）的情绪数据，调用 LLM 生成高光时刻与成长建议。
    week_offset=0 为本周，-1 为上周，以此类推。
    """
    today = date.today()
    # 以周一为起点
    week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
    week_end = week_start + timedelta(days=6)

    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row

        # 取本周所有已结束的 session（有 summary 或 mood）
        cursor = await conn.execute(
            """SELECT id, date(started_at) as day, mood, anxiety_detected, summary
               FROM sessions
               WHERE user_id = ?
                 AND date(started_at) BETWEEN ? AND ?
               ORDER BY started_at ASC""",
            (user_id, week_start.isoformat(), week_end.isoformat()),
        )
        sessions = await cursor.fetchall()

        # 取本周用户消息（用于关键词提取）
        cursor = await conn.execute(
            """SELECT m.content FROM messages m
               JOIN sessions s ON s.id = m.session_id
               WHERE s.user_id = ?
                 AND m.role = 'user'
                 AND date(m.created_at) BETWEEN ? AND ?""",
            (user_id, week_start.isoformat(), week_end.isoformat()),
        )
        msg_rows = await cursor.fetchall()

    # ---------- 情绪曲线：每天取最后一次 session 的 mood ----------
    daily: dict[str, dict] = {}
    for s in sessions:
        day = s["day"]
        mood = (s["mood"] or "平静").strip()
        daily[day] = {
            "date": day,
            "mood": mood,
            "score": _mood_score(mood),
            "category": _mood_category(mood),
            "anxiety": bool(s["anxiety_detected"]),
        }

    # 补全本周每一天（无数据的天默认平静）
    mood_curve = []
    for i in range(7):
        d = (week_start + timedelta(days=i)).isoformat()
        if d in daily:
            mood_curve.append(daily[d])
        else:
            mood_curve.append({
                "date": d, "mood": None, "score": None,
                "category": "neutral", "anxiety": False,
            })

    # ---------- 统计 ----------
    active_days = [d for d in mood_curve if d["mood"]]
    moods_with_data = [d["mood"] for d in active_days]
    anxiety_days = sum(1 for d in active_days if d["anxiety"])
    dominant_mood = max(set(moods_with_data), key=moods_with_data.count) if moods_with_data else "平静"
    avg_score = round(sum(d["score"] for d in active_days) / len(active_days), 1) if active_days else 5.0

    stats = {
        "total_sessions": len(sessions),
        "active_days": len(active_days),
        "anxiety_days": anxiety_days,
        "dominant_mood": dominant_mood,
        "avg_score": avg_score,
    }

    # ---------- LLM 生成高光时刻 + 成长建议 + 关键词 ----------
    highlights = ""
    suggestions = ""
    keywords = []
    summaries = [s["summary"] for s in sessions if s["summary"]]
    mood_list = [(s["day"], s["mood"] or "平静", "有焦虑" if s["anxiety_detected"] else "无焦虑") for s in sessions]

    if summaries or mood_list:
        context_lines = []
        for day, mood, anx in mood_list:
            context_lines.append(f"{day} | 心情：{mood} | {anx}")
        context = "\n".join(context_lines)
        if summaries:
            context += "\n\n本周对话摘要：\n" + "\n".join(f"- {s}" for s in summaries[:7])

        prompt = f"""以下是用户过去一周的情绪与对话数据：

{context}

请生成情绪周报的三个部分，用温暖、第二人称「你」的语气，像好朋友在写给你的信：

1. 高光时刻（80字以内）：提炼本周最值得珍藏的积极瞬间或进步，若整体情绪低落则以温柔的方式肯定用户坚持下来
2. 成长建议（120字以内）：基于本周的情绪走势和话题，给出1-2条温暖且具体可执行的建议
3. 关键词（5-8个）：从本周对话话题中提取有实际意义的名词或短语（如"工作压力"、"睡眠"、"家人"），不要提取语气词、代词或无意义的字符组合

只输出 JSON，格式：{{"highlights": "...", "suggestions": "...", "keywords": ["词1", "词2", ...]}}"""

        try:
            raw = await chat([{"role": "user", "content": prompt}], extra_system="只输出 JSON，不要有任何其他文字。")
            raw = raw.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            highlights = parsed.get("highlights", "")
            suggestions = parsed.get("suggestions", "")
            llm_keywords = parsed.get("keywords", [])
            if isinstance(llm_keywords, list) and llm_keywords:
                keywords = [{"word": w, "weight": len(llm_keywords) - i}
                            for i, w in enumerate(llm_keywords) if isinstance(w, str) and w.strip()]
        except Exception as e:
            print(f"[WeeklyReport] LLM 生成失败: {e}")
            highlights = "本周数据已收集，继续记录会生成更丰富的高光时刻。"
            suggestions = "坚持每天和小伴聊聊，情绪的积累会让你更了解自己。"
            keywords = []

    return {
        "period": {
            "start": week_start.isoformat(),
            "end": week_end.isoformat(),
        },
        "mood_curve": mood_curve,
        "keywords": keywords,
        "stats": stats,
        "highlights": highlights,
        "suggestions": suggestions,
    }
