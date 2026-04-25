# -*- coding: utf-8 -*-
"""情感陪伴 AI - 主入口：FastAPI + 聊天 / 总结 / RAG / 久未登录提醒"""
import asyncio
import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import secrets
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Cookie, Depends, Header, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import bcrypt as _bcrypt
from pydantic import BaseModel

from config import OPENAI_API_KEY, MAX_KNOWLEDGE_UPLOAD_BYTES, MAX_KNOWLEDGE_TEXT_CHARS, DB_PATH
from db import init_db, get_db, ensure_user
from prompts import COMPANION_SYSTEM, build_chat_context
from services.llm import chat, chat_stream, chat_with_knowledge, chat_stream_with_knowledge
from services.anxiety import analyze_anxiety
from services.mood import analyze_mood
from services.embedding import get_embedding
from services.rag import get_relevant_context, _extract_keywords
from services.summary import generate_summary
from services.image_gen import generate_mood_image
from services.reminder import get_reminder_if_inactive
from services.tts import synthesize_to_mp3
from services.long_memory import add_turn_to_memory, retrieve_relevant_memories, extract_and_save_profiles
from services.report import get_weekly_report
from services.weather import get_weather
from services.evaluate import evaluate_response

# RAG 调用统计（总次数、命中次数），用于看命中率
RAG_STATS = {"total": 0, "hits": 0}

# 用户上传的图片存放目录（对话中分享的图片）
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _image_path_to_content_parts(text: str, image_path: Optional[str], upload_dir: Path):
    """若 image_path 存在且文件可读，返回多模态 content（OpenAI 格式）；否则返回纯文本。"""
    if not image_path or not (upload_dir / image_path).exists():
        return text
    path = upload_dir / image_path
    try:
        raw = path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        ext = path.suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/gif" if ext == ".gif" else "image/webp"
        url = f"data:{mime};base64,{b64}"
    except Exception:
        return text
    if not text.strip():
        text = "[用户分享了一张图片]"
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]

# 明显是「接着上一句说」的短回复，不触发 RAG，避免误注入无关参考、打断对话
_FOLLOW_UP_PHRASES = ("详细说说", "然后呢", "继续", "还有呢", "怎么说", "再讲讲", "具体点", "还有吗", "嗯嗯", "好的", "哦")

# 含这些词时强制调用知识库（模型可能自认为知道答案而跳过搜索）
_KNOWLEDGE_KEYWORDS = frozenset({
    "焦虑", "情绪", "心理", "压力", "恐惧", "抑郁", "紧张", "症状", "治疗",
    "缓解", "性格", "恐慌", "焦虑症", "心理健康", "认知", "行为", "神经",
    "杏仁核", "皮质醇", "应激", "创伤", "惊恐", "强迫", "社交恐惧",
})


def _is_rag_skip_follow_up(msg: str) -> bool:
    """True 表示本条不查 RAG，只按对话历史继续聊（避免「详细说说」误命中知识库而答成别的）。"""
    s = (msg or "").strip()
    if len(s) <= 2:
        return True
    if len(s) <= 6 and any(s == p or s.startswith(p) or p in s for p in _FOLLOW_UP_PHRASES):
        return True
    return False


def _should_inject_rag_by_score(mode: str, score: float) -> bool:
    """根据检索相关度分数决定是否注入 RAG 内容。
    语义模式：余弦相似度 >= 0.45 视为相关；关键词模式：命中关键词数 >= 1 即注入。
    """
    if mode == "semantic":
        return score >= 0.45
    else:  # keyword
        return score >= 1


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """将长文本切成若干 chunk，优先在段落/句子边界分割，相邻 chunk 有 overlap 字重叠。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        is_last = end >= len(text)
        # 优先在段落或句尾分割，避免截断句子中间
        if not is_last:
            for sep in ("\n\n", "\n", "。", "！", "？", ".", "!", "?"):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if is_last:
            break
        start = end - overlap
    return chunks


def _read_upload_bytes_limited(file: UploadFile) -> bytes:
    """按字节上限分块读取上传体，避免一次性 read() 撑爆内存。"""
    max_b = MAX_KNOWLEDGE_UPLOAD_BYTES
    buf = bytearray()
    step = 1024 * 1024
    while True:
        piece = file.file.read(step)
        if not piece:
            break
        if len(buf) + len(piece) > max_b:
            mb = max_b // (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"文件过大（超过 {mb} MB）。请拆成多个较小的文件上传，或在 .env 中增大 MAX_KNOWLEDGE_UPLOAD_BYTES。",
            )
        buf.extend(piece)
    return bytes(buf)


def _enforce_knowledge_text_limit(text: str) -> None:
    cap = MAX_KNOWLEDGE_TEXT_CHARS
    n = len(text or "")
    if n > cap:
        raise HTTPException(
            status_code=400,
            detail=f"正文过长（{n} 字，上限 {cap}）。请拆成多个文件，或在 .env 中增大 MAX_KNOWLEDGE_TEXT_CHARS。",
        )


_MEMORY_RECALL_HINTS = ("记得", "还记得", "上次", "之前", "我说过", "我提过", "你知道我", "你还记得")


def _is_memory_recall_query(msg: str) -> bool:
    s = (msg or "").strip()
    return bool(s) and any(h in s for h in _MEMORY_RECALL_HINTS)


# ---------- Auth 依赖 ----------
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


async def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),   # sendBeacon 走 query string
) -> int:
    """FastAPI 依赖：从 Authorization: Bearer <token> 或 ?token= 读取 token，返回 user_id；未登录则 401。"""
    raw = token  # 优先 query string（sendBeacon 场景）
    if not raw and authorization and authorization.startswith("Bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    if not raw:
        raise HTTPException(status_code=401, detail="未登录，请先注册或登录")
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT user_id, expires_at FROM user_tokens WHERE token = ?", (raw,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        if row["expires_at"] < datetime.now().isoformat():
            raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
        return row["user_id"]
    finally:
        await conn.close()


async def _create_token(user_id: int) -> str:
    """生成 64 位随机 token，写入 user_tokens 表，30 天有效。"""
    token = secrets.token_hex(32)
    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
    conn = await get_db()
    try:
        await conn.execute(
            "INSERT INTO user_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at),
        )
        await conn.commit()
    finally:
        await conn.close()
    return token


# ---------- 请求体 ----------
class ChatRequest(BaseModel):
    user_id: int = 1
    session_id: int | None = None  # 不传则新建会话
    message: str
    weather: str | None = None  # 天气摘要，如 "☀️ 北京，晴，28°C"
    voice_hint: str | None = None  # 语音情绪信号，如 "声音较轻，语速偏慢"


class EndSessionRequest(BaseModel):
    user_id: int = 1
    session_id: int


class AddKnowledgeRequest(BaseModel):
    title: str = ""
    content: str
    source_url: str = ""


class TTSRequest(BaseModel):
    text: str


class RenameSessionRequest(BaseModel):
    title: str = ""


# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    key_status = "OK" if (OPENAI_API_KEY or "").strip() else "MISSING"
    visible_keys = [k for k in os.environ if "OPENAI" in k or "DASHSCOPE" in k or "RAILWAY" in k]
    print(f"[startup] OPENAI_API_KEY={key_status}, visible env keys: {visible_keys}", flush=True)
    yield
    # 关闭时如需清理可在此处理


app = FastAPI(title="情感陪伴 AI - 小伴", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态目录：以 main.py 所在目录为基准，避免工作目录影响
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
async def index():
    """前端页面：直接返回 HTML 内容，强制内联显示，避免浏览器另存为"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        html = index_file.read_text(encoding="utf-8")
        return HTMLResponse(
            content=html,
            headers={
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )
    return HTMLResponse(
        "<h1>情感陪伴 API</h1><p>请确保 static/index.html 存在，或访问 <a href='/docs'>/docs</a></p>"
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/config.js")
async def serve_config_js():
    """提供前端 API 地址配置（同源部署时为空；Vercel 部署时可指向 Render 后端）。"""
    config_file = STATIC_DIR / "config.js"
    if config_file.exists():
        return Response(
            content=config_file.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )
    return Response(
        content="window.__API_BASE__ = '';",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/uploads/{filename}")
async def serve_upload(filename: str):
    """提供用户上传的图片（对话中分享的图片），仅允许单层文件名。"""
    if ".." in filename or "/" in filename or "\\" in filename or not filename.strip():
        raise HTTPException(status_code=400, detail="invalid path")
    path = UPLOAD_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type=None)


UPLOAD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>知识库上传（仅管理员）</title>
  <style>
    body { font-family: sans-serif; max-width: 520px; margin: 40px auto; padding: 20px; }
    h1 { font-size: 1.2rem; color: #333; }
    label { display: block; margin-top: 12px; color: #666; font-size: 0.9rem; }
    input[type="text"], input[type="file"] { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
    button { margin-top: 16px; padding: 10px 20px; background: #c97b63; color: #fff; border: none; border-radius: 8px; cursor: pointer; }
    button:hover { opacity: 0.9; }
    .hint { font-size: 0.8rem; color: #888; margin-top: 8px; }
    .ok { color: green; margin-top: 12px; }
    .err { color: #c00; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>知识库上传（仅管理员）</h1>
  <p class="hint">上传的书/文章会进入 RAG 知识库，聊天时小伴会在后端自动引用，用户端不显示此功能。</p>
  <form id="f">
    <label>标题（可选，不填则用文件名）</label>
    <input type="text" name="title" placeholder="例如：某本书名或文章标题" />
    <label>来源链接（可选）</label>
    <input type="text" name="source_url" placeholder="https://..." />
    <label>选择文件（.txt / .pdf）</label>
    <input type="file" name="file" accept=".txt,.pdf" required />
    <button type="submit">上传到知识库</button>
  </form>
  <p id="msg"></p>
  <script>
    document.getElementById('f').onsubmit = async function(e) {
      e.preventDefault();
      var fd = new FormData(this);
      var msg = document.getElementById('msg');
      msg.className = ''; msg.textContent = '上传中…';
      try {
        var r = await fetch('/api/knowledge/upload', { method: 'POST', body: fd });
        var d = await r.json().catch(function(){ return {}; });
        if (r.ok) { msg.className = 'ok'; msg.textContent = '已添加到知识库：' + (d.title || d.id); this.reset(); }
        else { msg.className = 'err'; msg.textContent = d.detail || '上传失败'; }
      } catch (err) { msg.className = 'err'; msg.textContent = err.message || '请求失败'; }
    };
  </script>
</body>
</html>
"""


@app.get("/upload")
async def upload_page():
    """知识库上传页（仅管理员使用，不对外展示）。上传的书/文章进入 RAG，用户无感知。"""
    return HTMLResponse(UPLOAD_HTML)


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    """文本转语音：调用 DashScope qwen3-tts-flash，返回 mp3 音频。"""
    audio_bytes = await synthesize_to_mp3(req.text)
    return Response(content=audio_bytes, media_type="audio/wav")


# ---------- 对话 ----------
async def _get_current_mood(conn, session_id: int) -> str:
    """读取当前 session 上一轮写入的情绪（后台任务写入），无则返回平静。"""
    try:
        cursor = await conn.execute("SELECT mood FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return (row["mood"] or "平静") if row else "平静"
    except Exception:
        return "平静"


def _clean_reply(text: str) -> str:
    """去掉 [NEXT] 标记，返回纯文本，用于存库/长期记忆/评分。"""
    return re.sub(r'\[NEXT\]', ' ', text).strip()


def _split_reply(text: str) -> list[str]:
    """将 AI 回复拆分成 2-3 条短消息，模拟朋友连发的对话感。
    优先使用模型输出的 [NEXT] 分隔符，否则按句末标点自动分句合并。
    """
    # 模型主动分条
    if "[NEXT]" in text:
        parts = [p.strip() for p in text.split("[NEXT]") if p.strip()]
        if len(parts) > 1:
            return parts[:3]

    # 自动按句末标点分句
    raw = re.split(r'(?<=[。！？…~～\n])', text)
    sentences = [s.strip() for s in raw if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()]

    # 合并成 2-3 块，每块 ≤55 字
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) > 55:
            chunks.append(current)
            current = s
        else:
            current += s
    if current:
        chunks.append(current)

    # 超过 3 条时把末尾合并，少于 2 条不拆
    if len(chunks) > 3:
        chunks = chunks[:2] + ["".join(chunks[2:])]
    if len(chunks) < 2:
        return [text.strip()]
    return chunks


async def _run_evaluation(session_id: int, user_message: str, assistant_reply: str,
                          latency_ms: int | None = None, token_count: int | None = None):
    """后台异步评分，结果写入 evaluations 表，不阻塞主流程。"""
    scores = await evaluate_response(user_message, assistant_reply)
    if not scores:
        return
    conn = await get_db()
    try:
        await conn.execute(
            """INSERT INTO evaluations
               (session_id, user_message, assistant_reply, empathy, naturalness, helpfulness, safety, overall, comment, latency_ms, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_message[:500], assistant_reply[:1000],
             scores["empathy"], scores["naturalness"], scores["helpfulness"],
             scores["safety"], scores["overall"], scores["comment"], latency_ms, token_count),
        )
        await conn.commit()
        print(f"[Evaluate] session={session_id} overall={scores['overall']} | {scores['comment']}")
    except Exception as e:
        print(f"[Evaluate] 写库失败: {e}")
    finally:
        await conn.close()


_FOLLOWUP_KEYWORDS = ["准备", "打算", "计划", "在找", "在等", "想试", "想去", "想学", "要去", "正在", "面试", "申请", "等结果", "等消息"]


async def _get_followup_hint(user_id: int, conn) -> str:
    """查询 2-14 天前未跟进的话题，返回主动关怀提示（无则返回空字符串）。"""
    from datetime import date, timedelta
    today = date.today()
    date_from = (today - timedelta(days=14)).isoformat()
    date_to = (today - timedelta(days=2)).isoformat()
    try:
        cursor = await conn.execute(
            """SELECT summary FROM sessions
               WHERE user_id = ?
                 AND summary IS NOT NULL AND summary != ''
                 AND date(started_at) BETWEEN ? AND ?
               ORDER BY started_at DESC LIMIT 5""",
            (user_id, date_from, date_to),
        )
        rows = await cursor.fetchall()
        for row in rows:
            summary = row["summary"] or ""
            if any(kw in summary for kw in _FOLLOWUP_KEYWORDS):
                # 取第一个匹配的摘要
                return (
                    f"用户上次（2-14天前）曾提到以下事项，你可以在对话中找自然时机询问进展："
                    f"「{summary[:80]}」。不必强行提起，若用户主动聊其他事则顺着走。"
                )
    except Exception:
        pass
    return ""


async def _run_chat(user_id: int, session_id: Optional[int], message: str, image_path: Optional[str] = None, weather: Optional[str] = None):
    """内部：执行一轮对话（可带图片），返回 (session_id, reply, reminder)。"""
    await ensure_user(user_id)
    conn = await get_db()
    try:
        cursor = await conn.execute("SELECT last_login_at FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        last_login = row["last_login_at"] if row else None
        reminder = await get_reminder_if_inactive(last_login)
        followup_hint = ""
        if session_id is None:
            followup_hint = await _get_followup_hint(user_id, conn)
            await conn.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
            await conn.commit()
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            session_id = (await cursor.fetchone())["id"]
        try:
            cursor = await conn.execute(
                "SELECT role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
        except Exception:
            cursor = await conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
        rows = await cursor.fetchall()
        # 上下文窗口优化：只保留最近 12 条，若有更早内容则注入 session 摘要作补充
        CONTEXT_LIMIT = 12
        context_prefix = []
        if len(rows) > CONTEXT_LIMIT:
            older_rows = rows[:-CONTEXT_LIMIT]
            rows = rows[-CONTEXT_LIMIT:]
            # 尝试从 session 摘要获取早期内容的精华
            cursor2 = await conn.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
            srow2 = await cursor2.fetchone()
            session_summary = srow2["summary"] if srow2 and srow2["summary"] else None
            if session_summary:
                context_prefix = [{"role": "system", "content": f"[本次对话前期摘要] {session_summary}"}]
            else:
                # 没有摘要时用截断提示，避免模型产生"我们之前没聊过"的误解
                context_prefix = [{"role": "system", "content": f"[本次对话已有 {len(older_rows) + len(rows)} 条消息，以下仅展示最近 {CONTEXT_LIMIT} 条]"}]
        history = []
        for r in rows:
            role, content = r["role"], r["content"]
            img_path = r["image_path"] if "image_path" in r.keys() and r["image_path"] else None
            if role == "user" and img_path:
                content = _image_path_to_content_parts(content or "", img_path, UPLOAD_DIR)
            history.append({"role": role, "content": content})
        cursor = await conn.execute("SELECT anxiety_detected FROM sessions WHERE id = ?", (session_id,))
        srow = await cursor.fetchone()
        has_anxiety = bool(srow and srow["anxiety_detected"])
        memories = await retrieve_relevant_memories(
            user_id=user_id,
            query=message,
            exclude_session_id=session_id,
        )
        extra_system_parts = []
        if followup_hint:
            extra_system_parts.append(followup_hint)
        if weather:
            extra_system_parts.append(f"当前天气：{weather}。可在自然的情况下将天气融入对话，不要强行提及。")
        if has_anxiety:
            extra_system_parts.append("注意：用户近期对话中曾表现出焦虑或压力，回复时请更温柔、多些共情与支持。")
        if memories:
            memory_block = (
                "以下是该用户的历史对话片段（长期记忆）：\n"
                + "\n".join(f"- {m}" for m in memories[:6])
                + "\n\n请优先依据这些片段回答用户的偏好/经历/之前提到的事情。"
                + " 若片段包含用户的姓名/称呼/自我介绍，当用户询问你还记得我名字吗等身份信息时，直接给出姓名或称呼。"
                + " 若用户在追问是否记得/之前聊过什么，请先给出结论再自然展开。"
            )
            if _is_memory_recall_query(message):
                memory_block += "（追问场景：优先从长期记忆给出直接结论）"
            extra_system_parts.append(memory_block)
        extra_system = "\n\n".join(extra_system_parts)
        print(f"[Memory] 命中 {len(memories)} 条 | session={session_id} | q={message[:30]}")
        current_user_content = message
        if image_path:
            current_user_content = _image_path_to_content_parts(message, image_path, UPLOAD_DIR)
        messages = [{"role": "system", "content": COMPANION_SYSTEM}]
        messages.extend(context_prefix)
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_content})
        # function call RAG：追问类跳过工具，否则让模型自己决定是否查知识库
        use_tool = not _is_rag_skip_follow_up(message)
        force_tool = use_tool and any(kw in message for kw in _KNOWLEDGE_KEYWORDS)
        if use_tool:
            async def _rag_fn(query: str) -> list[str]:
                texts, _, _ = await get_relevant_context(query, user_id)
                RAG_STATS["total"] += 1
                if texts:
                    RAG_STATS["hits"] += 1
                print(f"[FunctionCall] RAG 返回 {len(texts)} 条")
                return texts
        else:
            _rag_fn = None
        try:
            _t0 = time.monotonic()
            reply, token_count = await chat_with_knowledge(messages, extra_system=extra_system, rag_fn=_rag_fn, force_tool=force_tool)
            latency_ms = int((time.monotonic() - _t0) * 1000)
        except Exception as e:
            err = str(e).strip() or "API 调用失败"
            if "api_key" in err.lower() or "auth" in err.lower() or "401" in err:
                err = "API 密钥无效或已过期，请检查 .env 中的 OPENAI_API_KEY"
            elif "429" in err or "402" in err or "quota" in err.lower() or "insufficient_quota" in err.lower() or "insufficient balance" in err.lower() or "insufficient_balance" in err.lower():
                err = "账户余额不足或额度已用完。请到对应平台充值，或改用 DeepSeek 等 API。"
            elif "connection" in err.lower() or "network" in err.lower():
                err = "无法连接至 AI 服务，请检查网络或 OPENAI_BASE_URL"
            elif "vision" in err.lower() or "image" in err.lower() or "multimodal" in err.lower() or "content" in err.lower():
                err = "当前模型可能不支持看图。发图片时请使用支持多模态的模型（如 gpt-4o、qwen-vl、glm-4v 等），并在 .env 中设置对应的 OPENAI_MODEL。"
            raise HTTPException(status_code=502, detail=err)
        clean_reply = _clean_reply(reply)
        try:
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, image_path) VALUES (?, ?, ?, ?)",
                (session_id, "user", message, image_path),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, image_path) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", clean_reply, None),
            )
        except Exception:
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", message),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", clean_reply),
            )
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() AS id")
        msg_row = await cursor.fetchone()
        message_id = msg_row["id"] if msg_row else None
        await add_turn_to_memory(
            user_id=user_id,
            session_id=session_id,
            user_message=message,
            assistant_reply=clean_reply,
        )
        import asyncio
        asyncio.ensure_future(extract_and_save_profiles(
            user_id=user_id,
            session_id=session_id,
            user_message=message,
            db_path=DB_PATH,
        ))
        asyncio.ensure_future(_run_evaluation(session_id, message, clean_reply, latency_ms, token_count))
        mood = await _get_current_mood(conn, session_id)
        messages_list = _split_reply(reply)
        return {
            "session_id": session_id,
            "reply": reply,
            "messages": messages_list,
            "reminder": reminder,
            "mood": mood or "平静",
            "anxiety_detected": has_anxiety,
            "message_id": message_id,
        }
    finally:
        await conn.close()


# ---------- Auth 端点 ----------
@app.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    """注册新账号，成功后返回 token（前端存 localStorage）。"""
    email = (req.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="请输入有效的邮箱地址")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    conn = await get_db()
    try:
        # 邮箱已注册则拒绝
        cursor = await conn.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            raise HTTPException(status_code=409, detail="该邮箱已注册，请直接登录")
        hashed = _bcrypt.hashpw(req.password.encode(), _bcrypt.gensalt()).decode()
        name = (req.name or "").strip() or email.split("@")[0]
        # 若 user_id=1 是无邮箱的旧匿名用户（加多用户前的遗留数据），直接接管以保留历史记录
        cursor = await conn.execute("SELECT id, email FROM users WHERE id = 1")
        legacy = await cursor.fetchone()
        if legacy and not legacy["email"]:
            user_id = 1
            await conn.execute(
                "UPDATE users SET name=?, email=?, password_hash=?, last_login_at=datetime('now') WHERE id=1",
                (name, email, hashed),
            )
        else:
            await conn.execute(
                "INSERT INTO users (name, email, password_hash, last_login_at) VALUES (?, ?, ?, datetime('now'))",
                (name, email, hashed),
            )
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            user_id = (await cursor.fetchone())["id"]
        await conn.commit()
    finally:
        await conn.close()
    token = await _create_token(user_id)
    return {"user_id": user_id, "name": name, "email": email, "token": token}


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    """用邮箱+密码登录，成功后返回 token。"""
    email = (req.email or "").strip().lower()
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT id, name, password_hash FROM users WHERE email = ?", (email,)
        )
        row = await cursor.fetchone()
        if not row or not row["password_hash"]:
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        if not _bcrypt.checkpw(req.password.encode(), row["password_hash"].encode()):
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        user_id = row["id"]
        name = row["name"] or email.split("@")[0]
        await conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?", (user_id,)
        )
        await conn.commit()
    finally:
        await conn.close()
    token = await _create_token(user_id)
    return {"user_id": user_id, "name": name, "email": email, "token": token}


@app.post("/api/auth/logout")
async def api_logout(authorization: Optional[str] = Header(None)):
    """登出：删除服务端 token。"""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        conn = await get_db()
        try:
            await conn.execute("DELETE FROM user_tokens WHERE token = ?", (token,))
            await conn.commit()
        finally:
            await conn.close()
    return {"ok": True}


@app.post("/api/auth/change-password")
async def api_change_password(
    current_password: str = Body(...),
    new_password: str = Body(...),
    user_id: int = Depends(get_current_user),
):
    """修改密码：需要提供当前密码验证身份。"""
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 位")
    conn = await get_db()
    try:
        cursor = await conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        if not row or not row["password_hash"]:
            raise HTTPException(status_code=400, detail="账号异常，请重新登录")
        if not _bcrypt.checkpw(current_password.encode(), row["password_hash"].encode()):
            raise HTTPException(status_code=401, detail="当前密码错误")
        new_hash = _bcrypt.hashpw(new_password.encode(), _bcrypt.gensalt()).decode()
        await conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        # 让其他设备上的 token 全部失效（安全起见），保留当前 token
        await conn.commit()
    finally:
        await conn.close()
    return {"ok": True}


@app.delete("/api/auth/account")
async def api_delete_account(
    password: str = Body(...),
    user_id: int = Depends(get_current_user),
):
    """注销账号：删除该用户的全部数据（消息、会话、token、向量记忆、上传图片）。"""
    conn = await get_db()
    try:
        # 验证密码
        cursor = await conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        if not row or not row["password_hash"]:
            raise HTTPException(status_code=400, detail="账号异常")
        if not _bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            raise HTTPException(status_code=401, detail="密码错误，注销已取消")
        # 删除消息（通过 session）
        cursor = await conn.execute("SELECT id FROM sessions WHERE user_id = ?", (user_id,))
        session_ids = [r["id"] for r in await cursor.fetchall()]
        if session_ids:
            ph = ",".join("?" * len(session_ids))
            # 收集图片路径
            cursor = await conn.execute(
                f"SELECT image_path FROM messages WHERE session_id IN ({ph}) AND image_path IS NOT NULL",
                session_ids,
            )
            img_paths = [r["image_path"] for r in await cursor.fetchall()]
            await conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", session_ids)
        else:
            img_paths = []
        await conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        await conn.execute("DELETE FROM user_tokens WHERE user_id = ?", (user_id,))
        await conn.execute("DELETE FROM reminders WHERE user_id = ?", (user_id,))
        await conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await conn.commit()
    finally:
        await conn.close()
    # 删除上传图片
    for img in img_paths:
        try:
            (UPLOAD_DIR / img).unlink(missing_ok=True)
        except Exception:
            pass
    # 删除向量记忆
    try:
        from services.long_memory import _get_collection
        import asyncio
        def _del_chroma():
            c = _get_collection()
            if c:
                results = c.get(where={"user_id": int(user_id)}, include=[])
                if results and results.get("ids"):
                    c.delete(ids=results["ids"])
        await asyncio.get_event_loop().run_in_executor(None, _del_chroma)
    except Exception:
        pass
    return {"ok": True}


@app.get("/api/auth/me")
async def api_me(user_id: int = Depends(get_current_user)):
    """返回当前登录用户信息（前端用于判断是否已登录）。"""
    conn = await get_db()
    try:
        cursor = await conn.execute("SELECT name, email FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="用户不存在")
        return {"user_id": user_id, "name": row["name"], "email": row["email"]}
    finally:
        await conn.close()


@app.get("/api/mood")
async def api_mood(user_id: int = Depends(get_current_user)):
    """获取当天综合心情（以当天截至当前的所有对话计算），用于页面展示。"""
    try:
        conn = await get_db()
        try:
            mood = await _get_today_mood(conn, user_id)
            return {"mood": mood or "平静"}
        finally:
            await conn.close()
    except Exception:
        return {"mood": "平静"}


@app.post("/api/chat")
async def api_chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """发送一条消息，返回小伴回复。若久未登录会先返回提醒再正常对话。"""
    if not (OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=503, detail="未配置 API 密钥。请在项目文件夹中创建 .env 文件，填写：OPENAI_API_KEY=你的密钥")
    return await _run_chat(user_id, req.session_id, req.message, None, req.weather)


@app.post("/api/chat/send")
async def api_chat_send(
    message: str = Form(""),
    session_id: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    user_id: int = Depends(get_current_user),
):
    """发送消息并可附带一张图片（multipart/form-data）。"""
    if not (OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=503, detail="未配置 API 密钥")
    image_path = None
    if image and image.filename:
        ext = Path(image.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT:
            raise HTTPException(status_code=400, detail="仅支持图片格式：jpg、png、gif、webp")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{ext}"
        path = UPLOAD_DIR / name
        try:
            path.write_bytes(await image.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"图片保存失败: {e}")
        image_path = name
    try:
        return await _run_chat(user_id, session_id, message or "", image_path)
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).strip() or "请求处理失败"
        if "api_key" in err.lower() or "auth" in err.lower():
            err = "API 密钥未配置或无效，请检查 .env 中的 OPENAI_API_KEY"
        raise HTTPException(status_code=502, detail=err)


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """流式回复（仅返回 assistant 内容，不写库；如需写库可先非流式写再这里只做展示）。"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
    await ensure_user(user_id)
    conn = await get_db()
    session_id = req.session_id
    if session_id is None:
        await conn.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() AS id")
        session_id = (await cursor.fetchone())["id"]
    cursor = await conn.execute(
        """SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 30""",
        (session_id,),
    )
    history = [{"role": r["role"], "content": r["content"]} for r in await cursor.fetchall()]
    cursor = await conn.execute("SELECT anxiety_detected FROM sessions WHERE id = ?", (session_id,))
    srow = await cursor.fetchone()
    has_anxiety = bool(srow and srow["anxiety_detected"])
    memories = await retrieve_relevant_memories(
        user_id=user_id,
        query=req.message,
        exclude_session_id=session_id,
    )
    extra_system_parts = []
    if req.weather:
        extra_system_parts.append(f"当前天气：{req.weather}。可在自然的情况下将天气融入对话，不要强行提及。")
    if req.voice_hint:
        print(f"[VoiceHint] {req.voice_hint!r}")
        extra_system_parts.append(f"【语音状态参考】用户本次说话时：{req.voice_hint}。请据此调整回应的语气和关注点，但不要直接提及这条信息。")
    if has_anxiety:
        extra_system_parts.append("注意：用户近期对话中曾表现出焦虑或压力，回复时请更温柔、多些共情与支持。")
    if memories:
        memory_block = (
            "以下是该用户的历史对话片段（长期记忆）：\n"
            + "\n".join(f"- {m}" for m in memories[:6])
            + "\n\n请优先依据这些片段回答用户的偏好/经历/之前提到的事情。"
            + " 若片段包含用户的姓名/称呼/自我介绍，当用户询问你还记得我名字吗等身份信息时，直接给出姓名或称呼。"
            + " 若用户在追问是否记得/之前聊过什么，请先给出结论再自然展开。"
        )
        if _is_memory_recall_query(req.message):
            memory_block += "（追问场景：优先从长期记忆给出直接结论）"
        extra_system_parts.append(memory_block)
    extra_system = "\n\n".join(extra_system_parts)
    print(f"[Memory] 命中 {len(memories)} 条 | session={session_id} | q={req.message[:30]}")
    messages = [{"role": "system", "content": COMPANION_SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})
    use_tool = not _is_rag_skip_follow_up(req.message)
    force_tool = use_tool and any(kw in req.message for kw in _KNOWLEDGE_KEYWORDS)
    if use_tool:
        async def _stream_rag_fn(query: str) -> list[str]:
            texts, _, _ = await get_relevant_context(query, user_id)
            RAG_STATS["total"] += 1
            if texts:
                RAG_STATS["hits"] += 1
            print(f"[FunctionCall] RAG 返回 {len(texts)} 条")
            return texts
    else:
        _stream_rag_fn = None

    async def gen():
        try:
            # 第一帧返回 session_id，让前端能追踪会话
            yield f"data: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            full = []
            _t0 = time.monotonic()
            async for chunk in chat_stream_with_knowledge(messages, extra_system=extra_system, rag_fn=_stream_rag_fn, force_tool=force_tool):
                full.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            stream_latency_ms = int((time.monotonic() - _t0) * 1000)
            content = "".join(full)
            clean_content = _clean_reply(content)
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", req.message),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", clean_content),
            )
            await conn.commit()
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            msg_row = await cursor.fetchone()
            if msg_row:
                yield f"data: {json.dumps({'message_id': msg_row['id']})}\n\n"
            await add_turn_to_memory(
                user_id=user_id,
                session_id=session_id,
                user_message=req.message,
                assistant_reply=clean_content,
            )
            import asyncio
            asyncio.ensure_future(extract_and_save_profiles(
                user_id=user_id,
                session_id=session_id,
                user_message=req.message,
                db_path=DB_PATH,
            ))
            asyncio.ensure_future(_run_evaluation(session_id, req.message, clean_content, stream_latency_ms))
        finally:
            await conn.close()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 结束会话并生成总结、焦虑分析 ----------
@app.post("/api/session/end")
async def api_end_session(req: EndSessionRequest):
    """结束当前会话：分析焦虑 + 生成总结并落库。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT id, role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC",
            (req.session_id,),
        )
        rows = await cursor.fetchall()
        messages = []
        for r in rows:
            content = r["content"] or ""
            if r["role"] == "user" and ("image_path" in r.keys() and r["image_path"]):
                content = content + " [用户分享了一张图片]"
            messages.append({"role": r["role"], "content": content})
        if not messages:
            return {"session_id": req.session_id, "summary": None, "anxiety_detected": False}
        anxiety = await analyze_anxiety(messages)
        summary = await generate_summary(messages)
        mood = "平静"
        try:
            mood = await analyze_mood(messages) or "平静"
        except Exception:
            pass
        mood_image_url = await generate_mood_image(mood, summary)
        await conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), summary = ?, anxiety_detected = ?, mood = ?, mood_image_url = ? WHERE id = ?",
            (summary, 1 if anxiety else 0, mood, mood_image_url, req.session_id),
        )
        await conn.commit()
        return {"session_id": req.session_id, "summary": summary, "anxiety_detected": anxiety, "mood": mood, "mood_image_url": mood_image_url}
    finally:
        await conn.close()


# ---------- 会话列表（左侧栏：不同话题，可继续上次对话）----------
@app.get("/api/sessions")
async def api_sessions(user_id: int = Depends(get_current_user), limit: int = 50):
    """获取用户的会话列表（含未结束的），按开始时间倒序。用于左侧栏展示、点击后加载该会话消息。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, started_at, ended_at, summary, title, mood, mood_image_url
               FROM sessions WHERE user_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        out = []
        for r in rows:
            title = r["title"] if "title" in r.keys() and r["title"] else None
            out.append({
                "session_id": r["id"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "summary": r["summary"] if r["summary"] else None,
                "title": title,
                "mood": r["mood"] if "mood" in r.keys() else None,
                "mood_image_url": r["mood_image_url"] if "mood_image_url" in r.keys() else None,
            })
        return out
    finally:
        await conn.close()


@app.get("/api/sessions/{session_id}/messages")
async def api_session_messages(session_id: int, user_id: int = Depends(get_current_user)):
    """获取某会话的全部消息，用于切换会话时加载历史并继续对话。校验 session 属于当前用户。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT user_id FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row or row["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        cursor = await conn.execute(
            "SELECT role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        out = []
        for r in rows:
            item = {"role": r["role"], "content": r["content"]}
            if "image_path" in r.keys() and r["image_path"]:
                item["image_path"] = r["image_path"]
            out.append(item)
        return out
    finally:
        await conn.close()


@app.get("/api/search")
async def api_search(user_id: int = Depends(get_current_user), q: str = ""):
    """搜索摘要和消息内容，返回匹配的 session 列表。"""
    q = q.strip()
    if not q:
        return []
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT DISTINCT s.id, s.started_at, s.summary, s.mood, s.anxiety_detected
               FROM sessions s
               LEFT JOIN messages m ON m.session_id = s.id
               WHERE s.user_id = ?
                 AND (s.summary LIKE ? OR m.content LIKE ?)
               ORDER BY s.started_at DESC
               LIMIT 30""",
            (user_id, f"%{q}%", f"%{q}%"),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": r["id"],
                "started_at": r["started_at"],
                "summary": r["summary"] or "",
                "mood": r["mood"] or "",
                "anxiety_detected": bool(r["anxiety_detected"]),
            }
            for r in rows
        ]
    finally:
        await conn.close()


@app.get("/api/export/markdown")
async def api_export_markdown(user_id: int = Depends(get_current_user)):
    """将用户所有会话导出为 Markdown 文件。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, started_at, mood, summary
               FROM sessions WHERE user_id = ?
               ORDER BY started_at ASC""",
            (user_id,),
        )
        sessions = await cursor.fetchall()

        lines = [f"# 树洞日记 · 我的记录\n\n导出时间：{__import__('datetime').date.today()}\n\n---\n"]
        for s in sessions:
            date = (s["started_at"] or "")[:16].replace("T", " ")
            mood = s["mood"] or "—"
            lines.append(f"\n## {date}　心情：{mood}\n")
            if s["summary"]:
                lines.append(f"{s['summary']}\n")
            cursor2 = await conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (s["id"],),
            )
            msgs = await cursor2.fetchall()
            if msgs:
                lines.append("\n<details><summary>查看对话记录</summary>\n")
                for m in msgs:
                    role_label = "我" if m["role"] == "user" else "小伴"
                    lines.append(f"\n**{role_label}**：{m['content'] or ''}")
                lines.append("\n\n</details>\n")
            lines.append("\n---\n")

        content = "\n".join(lines)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename*=UTF-8''treeholediary.md"},
        )
    finally:
        await conn.close()


@app.get("/api/mood/calendar")
async def api_mood_calendar(user_id: int = Depends(get_current_user), year: int = 0, month: int = 0):
    """返回指定月份每天的情绪（用于日历热力图）。"""
    import datetime
    if not year:
        year = datetime.date.today().year
    if not month:
        month = datetime.date.today().month
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-31"
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT date(started_at) as day, mood, anxiety_detected
               FROM sessions WHERE user_id = ? AND date(started_at) BETWEEN ? AND ?
               ORDER BY started_at ASC""",
            (user_id, start, end),
        )
        rows = await cursor.fetchall()
        result: dict[str, dict] = {}
        for r in rows:
            day = r["day"]
            mood = r["mood"] or ""
            neg = bool(re.search(r"焦虑|难过|疲惫|烦躁|委屈|低落|压力|紧张|恐惧|绝望|崩溃", mood))
            pos = bool(re.search(r"开心|兴奋|愉快|高兴|喜悦|放松|安心|满足|轻松|平和|平静", mood))
            result[day] = {
                "mood": mood,
                "category": "negative" if neg else ("positive" if pos else "neutral"),
                "anxiety": bool(r["anxiety_detected"]),
            }
        return result
    finally:
        await conn.close()


@app.patch("/api/sessions/{session_id}")
async def api_rename_session(session_id: int, req: RenameSessionRequest, user_id: int = Depends(get_current_user)):
    """重命名会话（设置 title）。仅允许修改当前用户的会话。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT id FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        await conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND user_id = ?",
            ((req.title or "").strip() or None, session_id, user_id),
        )
        await conn.commit()
        return {"session_id": session_id, "title": (req.title or "").strip() or None}
    finally:
        await conn.close()


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: int, user_id: int = Depends(get_current_user)):
    """删除会话及其全部消息。仅允许删除当前用户的会话。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT id FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await conn.commit()
        return {"session_id": session_id, "deleted": True}
    finally:
        await conn.close()


# ---------- 总结列表 ----------
@app.get("/api/summaries")
async def api_summaries(user_id: int = Depends(get_current_user), limit: int = 20):
    """获取用户的历史会话总结（生活记录），含该会话中用户分享的图片列表。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, started_at, ended_at, summary, anxiety_detected, mood, mood_image_url
               FROM sessions WHERE user_id = ? AND summary IS NOT NULL AND summary != ''
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        session_ids = [r["id"] for r in rows]
        images_by_session = {}
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            try:
                cursor = await conn.execute(
                    f"SELECT session_id, image_path FROM messages WHERE session_id IN ({placeholders}) AND image_path IS NOT NULL AND image_path != '' ORDER BY id ASC",
                    session_ids,
                )
                for row in await cursor.fetchall():
                    sid = row["session_id"]
                    if sid not in images_by_session:
                        images_by_session[sid] = []
                    images_by_session[sid].append(row["image_path"])
            except Exception:
                pass
        def _mood(r):
            if "mood" in r.keys() and r["mood"]:
                return r["mood"]
            return None
        out = [
            {
                "session_id": r["id"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "summary": r["summary"],
                "anxiety_detected": bool(r["anxiety_detected"]),
                "mood": _mood(r),
                "mood_image_url": r["mood_image_url"] if "mood_image_url" in r.keys() else None,
                "images": images_by_session.get(r["id"], []),
            }
            for r in rows
        ]
        backfill_count = 0
        for item in out:
            if item["mood"] is not None or backfill_count >= 3:
                continue
            try:
                cursor = await conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (item["session_id"],),
                )
                msgs = [{"role": r["role"], "content": (r["content"] or "")} for r in await cursor.fetchall()]
                if msgs:
                    mood = await analyze_mood(msgs) or "平静"
                    await conn.execute("UPDATE sessions SET mood = ? WHERE id = ?", (mood, item["session_id"]))
                    await conn.commit()
                    item["mood"] = mood
                    backfill_count += 1
            except Exception:
                pass
        return out
    finally:
        await conn.close()


# ---------- RAG 调试：查看某问题会注入的参考内容 ----------
@app.get("/api/debug/rag")
async def api_debug_rag(q: str = ""):
    """调试用：查看问题 q 会检索到并注入的参考内容。方便核对「答案在知识库但没答对」是检索问题还是模型问题。"""
    if not q.strip():
        return {"query": "", "mode": "", "count": 0, "snippets": [], "hint": "请加参数 ?q=你的问题，例如 ?q=什么是广泛性焦虑"}
    rag_texts, mode, score = await get_relevant_context(q.strip(), user_id=1, limit=8)
    return {
        "query": q.strip(),
        "mode": mode,
        "best_score": round(score, 4),
        "injected": bool(rag_texts),
        "count": len(rag_texts),
        "snippets": [s[:800] + ("…" if len(s) > 800 else "") for s in rag_texts],
    }


@app.get("/api/debug/memory")
async def api_debug_memory(q: str = "", user_id: int = 1, session_id: int | None = None, limit: int = 6):
    """调试长期记忆召回：查看问题 q 会命中的历史对话片段。"""
    if not q.strip():
        return {"query": "", "count": 0, "snippets": [], "hint": "请加参数 ?q=你的问题，例如 ?q=你还记得我喜欢什么吗"}
    items = await retrieve_relevant_memories(
        user_id=user_id,
        query=q.strip(),
        exclude_session_id=session_id,
        limit=max(1, min(limit, 12)),
    )
    return {
        "query": q.strip(),
        "count": len(items),
        "snippets": [s[:500] + ("…" if len(s) > 500 else "") for s in items],
    }


# ---------- RAG 知识库 ----------
@app.get("/api/knowledge")
async def api_list_knowledge(limit: int = 50):
    """获取知识库列表（用于 RAG 检索）。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, title, content, source_url, created_at
               FROM knowledge ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"] or "",
                "content": (r["content"] or "")[:500],
                "source_url": r["source_url"] or "",
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        await conn.close()


@app.post("/api/knowledge")
async def api_add_knowledge(req: AddKnowledgeRequest):
    """添加一条知识/文章（标题、内容、来源链接），聊天时会做语义/关键词检索并引用。"""
    import asyncio
    conn = await get_db()
    chunk_ids: list[int] = []
    try:
        use_title = (req.title or "").strip()
        content = (req.content or "").strip()
        _enforce_knowledge_text_limit(content)
        src = (req.source_url or "").strip()
        chunks = _chunk_text(content) or [content]
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_title = f"{use_title}（{i + 1}/{total}）" if total > 1 and use_title else use_title
            await conn.execute(
                "INSERT INTO knowledge (title, content, source_url) VALUES (?, ?, ?)",
                (chunk_title, chunk, src),
            )
            await conn.commit()
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            chunk_ids.append((await cursor.fetchone())["id"])
    finally:
        await conn.close()
    if chunk_ids:
        asyncio.create_task(_backfill_embeddings(chunk_ids))
    return {"id": chunk_ids[0] if chunk_ids else None, "chunks": total}


@app.delete("/api/knowledge/{kid}")
async def api_delete_knowledge(kid: int):
    """删除一条知识库条目。"""
    conn = await get_db()
    try:
        await conn.execute("DELETE FROM knowledge WHERE id = ?", (kid,))
        await conn.commit()
        return {"ok": True}
    finally:
        await conn.close()


def _extract_text_from_file(file: UploadFile) -> str:
    """根据文件名后缀解析文本：支持 .txt、.pdf。读入与提取均受配置上限保护。"""
    name = (file.filename or "").lower()
    raw = _read_upload_bytes_limited(file)
    if name.endswith(".txt"):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="replace")
        _enforce_knowledge_text_limit(text)
        return text
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 读取失败: {e}")
        cap = MAX_KNOWLEDGE_TEXT_CHARS
        parts: list[str] = []
        total = 0
        try:
            for page in reader.pages:
                t = (page.extract_text() or "").strip()
                if not t:
                    continue
                if total + len(t) > cap:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF 提取正文超过上限（{cap} 字）。请拆分或减少页数，或在 .env 增大 MAX_KNOWLEDGE_TEXT_CHARS。",
                    )
                parts.append(t)
                total += len(t)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")
        return "\n\n".join(parts) if parts else ""
    raise HTTPException(status_code=400, detail="仅支持 .txt 或 .pdf 文件")


async def _backfill_embeddings(chunk_ids: list[int]) -> None:
    """后台任务：为刚入库的 chunk 逐一生成 embedding 并回填。不阻塞上传响应。"""
    import asyncio
    conn = await get_db()
    try:
        for kid in chunk_ids:
            cursor = await conn.execute(
                "SELECT title, content FROM knowledge WHERE id = ?", (kid,)
            )
            row = await cursor.fetchone()
            if not row:
                continue
            text_for_emb = (row["title"] or "") + "\n" + (row["content"] or "")
            try:
                emb = await get_embedding(text_for_emb)
                if emb:
                    await conn.execute(
                        "UPDATE knowledge SET embedding = ? WHERE id = ?",
                        (json.dumps(emb), kid),
                    )
                    await conn.commit()
            except Exception as e:
                print(f"[Upload] embedding 回填失败 id={kid}: {e}")
            await asyncio.sleep(0.05)  # 避免并发太快触发限速
    finally:
        await conn.close()


@app.post("/api/knowledge/upload")
async def api_upload_knowledge(
    file: UploadFile = File(...),
    title: str = Form(""),
    source_url: str = Form(""),
):
    """上传文件到知识库：支持 .txt、.pdf，自动分块入库，embedding 在后台异步生成。"""
    import asyncio
    if not file.filename:
        raise HTTPException(status_code=400, detail="请选择文件")
    text = _extract_text_from_file(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析出文字")
    use_title = (title or "").strip() or (file.filename or "上传文件")
    content = text.strip()
    src = (source_url or "").strip()
    chunks = _chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析")
    conn = await get_db()
    chunk_ids: list[int] = []
    try:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_title = f"{use_title}（{i + 1}/{total}）" if total > 1 else use_title
            try:
                await conn.execute(
                    "INSERT INTO knowledge (title, content, source_url) VALUES (?, ?, ?)",
                    (chunk_title, chunk, src),
                )
                await conn.commit()
                cursor = await conn.execute("SELECT last_insert_rowid() AS id")
                chunk_ids.append((await cursor.fetchone())["id"])
            except Exception as e:
                print(f"[Upload] 写入 chunk {i+1} 失败: {e}")
    finally:
        await conn.close()
    if not chunk_ids:
        raise HTTPException(status_code=500, detail="写入知识库失败")
    # 后台异步生成 embedding，不阻塞本次请求
    asyncio.create_task(_backfill_embeddings(chunk_ids))
    return {"id": chunk_ids[0], "title": use_title, "chunks": total, "note": "文件已入库，向量索引正在后台生成，稍后即可语义检索"}


# ---------- 健康检查与久未登录提醒 ----------
@app.get("/api/check-in")
async def api_check_in(user_id: int = Depends(get_current_user), test_reminder: bool = False):
    """用户打开应用时调用：先根据上次登录判断是否久未登录并返回提醒，再更新 last_login。
    test_reminder=True 时强制返回一条问候（用于本地测试久未登录效果），不修改 last_login。"""
    conn = await get_db()
    try:
        if test_reminder:
            reminder = await get_reminder_if_inactive("2000-01-01 00:00:00")
            return {"reminder": reminder or "好久不见呀～最近怎么样？"}
        cursor = await conn.execute(
            "SELECT last_login_at FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        last_login = row["last_login_at"] if row else None
        reminder = await get_reminder_if_inactive(last_login)
        await ensure_user(user_id)
        return {"reminder": reminder}
    finally:
        await conn.close()


@app.get("/api/weather")
async def api_weather(lat: float, lon: float):
    """根据经纬度获取当前天气（通过 open-meteo.com，无需 API Key）。"""
    try:
        result = await get_weather(lat, lon)
        print(f"[Weather] 成功: {result.get('summary')}")
        return result
    except Exception as e:
        print(f"[Weather] 失败: {type(e).__name__}: {e}")
        return None


@app.get("/api/report/weekly")
async def api_weekly_report(user_id: int = Depends(get_current_user), week_offset: int = 0):
    """获取情绪周报。week_offset=0 为本周，-1 为上周，以此类推。"""
    print(f"[report] start user={user_id} week_offset={week_offset}", flush=True)
    try:
        data = await get_weekly_report(user_id, week_offset)
        print(f"[report] done", flush=True)
        return data
    except Exception as e:
        print(f"[report] error: {e}", flush=True)
        raise


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int  # 1 = 👍, -1 = 👎


@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest, user_id: int = Depends(get_current_user)):
    """保存用户对某条 AI 回复的 👍/👎 反馈。"""
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating 只能为 1 或 -1")
    conn = await get_db()
    try:
        # 拿 session_id（方便后续统计）
        cursor = await conn.execute(
            "SELECT session_id FROM messages WHERE id = ?", (req.message_id,)
        )
        row = await cursor.fetchone()
        session_id = row["session_id"] if row else None
        # 如有旧反馈则更新，否则插入
        await conn.execute(
            """INSERT INTO message_feedback (message_id, session_id, user_id, rating)
               VALUES (?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (req.message_id, session_id, user_id, req.rating),
        )
        # aiosqlite 不支持 UPSERT，改为先删后插
        await conn.execute(
            "DELETE FROM message_feedback WHERE message_id = ? AND user_id = ?",
            (req.message_id, user_id),
        )
        await conn.execute(
            "INSERT INTO message_feedback (message_id, session_id, user_id, rating) VALUES (?, ?, ?, ?)",
            (req.message_id, session_id, user_id, req.rating),
        )
        await conn.commit()
        return {"ok": True}
    finally:
        await conn.close()


@app.get("/api/feedback/export")
async def api_feedback_export(user_id: int = Depends(get_current_user)):
    """导出该用户全部反馈为 CSV。"""
    import csv, io
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT mf.id, mf.message_id, mf.rating, mf.created_at,
                      m.content AS assistant_reply, mf.session_id
               FROM message_feedback mf
               LEFT JOIN messages m ON m.id = mf.message_id
               WHERE mf.user_id = ?
               ORDER BY mf.created_at DESC""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "message_id", "rating", "created_at", "assistant_reply", "session_id"])
        for r in rows:
            writer.writerow([r["id"], r["message_id"], r["rating"], r["created_at"],
                             (r["assistant_reply"] or "")[:200], r["session_id"]])
        return Response(
            content=output.getvalue().encode("utf-8-sig"),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="feedback.csv"'},
        )
    finally:
        await conn.close()


@app.get("/api/dashboard/stats")
async def api_dashboard_stats(user_id: int = Depends(get_current_user)):
    """看板数据：评分趋势 + 用户反馈统计。"""
    conn = await get_db()
    try:
        # 评分趋势（最近 30 天，按天聚合）
        cursor = await conn.execute(
            """SELECT date(e.created_at) AS date,
                      ROUND(AVG(e.overall), 2) AS avg_overall,
                      ROUND(AVG(e.empathy), 2) AS empathy,
                      ROUND(AVG(e.naturalness), 2) AS naturalness,
                      ROUND(AVG(e.helpfulness), 2) AS helpfulness,
                      ROUND(AVG(e.latency_ms), 0) AS avg_latency_ms,
                      ROUND(AVG(e.token_count), 0) AS avg_token_count,
                      COUNT(*) AS count
               FROM evaluations e
               JOIN sessions s ON s.id = e.session_id
               WHERE s.user_id = ? AND e.created_at >= date('now', '-30 days')
               GROUP BY date(e.created_at)
               ORDER BY date ASC""",
            (user_id,),
        )
        eval_trend = [dict(r) for r in await cursor.fetchall()]

        # 反馈总计
        cursor = await conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS up,
                      SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS down
               FROM message_feedback WHERE user_id = ?""",
            (user_id,),
        )
        fb_row = await cursor.fetchone()
        feedback_summary = {
            "total": fb_row["total"] or 0,
            "up": fb_row["up"] or 0,
            "down": fb_row["down"] or 0,
        }

        # 反馈趋势（最近 30 天）
        cursor = await conn.execute(
            """SELECT date(created_at) AS date,
                      SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS up,
                      SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS down
               FROM message_feedback
               WHERE user_id = ? AND created_at >= date('now', '-30 days')
               GROUP BY date(created_at)
               ORDER BY date ASC""",
            (user_id,),
        )
        feedback_trend = [dict(r) for r in await cursor.fetchall()]

        # 最近 10 条差评对话
        cursor = await conn.execute(
            """SELECT mf.created_at, mf.session_id, m.content AS assistant_reply,
                      prev.content AS user_message
               FROM message_feedback mf
               LEFT JOIN messages m ON m.id = mf.message_id
               LEFT JOIN messages prev ON prev.session_id = m.session_id
                   AND prev.role = 'user'
                   AND prev.id = (
                       SELECT MAX(id) FROM messages
                       WHERE session_id = m.session_id AND role = 'user' AND id < m.id
                   )
               WHERE mf.user_id = ? AND mf.rating = -1
               ORDER BY mf.created_at DESC LIMIT 10""",
            (user_id,),
        )
        recent_negative = [dict(r) for r in await cursor.fetchall()]

        return {
            "eval_trend": eval_trend,
            "feedback_summary": feedback_summary,
            "feedback_trend": feedback_trend,
            "recent_negative": recent_negative,
        }
    finally:
        await conn.close()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """数据看板页面。"""
    from pathlib import Path
    p = Path("static/dashboard.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="dashboard.html not found")


@app.get("/api/evaluations/ping")
async def api_evaluation_ping():
    """直接测试 Gemini 评分是否可用，无需登录。"""
    from config import GEMINI_API_KEY, GEMINI_MODEL
    if not GEMINI_API_KEY:
        return {"ok": False, "error": "GEMINI_API_KEY 未配置"}
    try:
        result = await evaluate_response("今天心情很糟糕", "听起来你今天过得不太好，能说说发生什么了吗？")
        if result:
            return {"ok": True, "result": result}
        return {"ok": False, "error": "evaluate_response 返回 None（详见服务端终端日志）"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/evaluations")
async def api_evaluations(user_id: int = Depends(get_current_user), limit: int = 50):
    """查看该用户最近的 AI 回复质量评分（LLM-as-Judge）。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT e.id, e.session_id, e.user_message, e.assistant_reply,
                      e.empathy, e.naturalness, e.helpfulness, e.safety, e.overall, e.comment, e.created_at
               FROM evaluations e
               JOIN sessions s ON s.id = e.session_id
               WHERE s.user_id = ?
               ORDER BY e.id DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "user_message": r["user_message"],
                "assistant_reply": r["assistant_reply"],
                "scores": {
                    "empathy": r["empathy"],
                    "naturalness": r["naturalness"],
                    "helpfulness": r["helpfulness"],
                    "safety": r["safety"],
                    "overall": r["overall"],
                },
                "comment": r["comment"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        await conn.close()


@app.get("/api/evaluations/stats")
async def api_evaluation_stats(user_id: int = Depends(get_current_user)):
    """返回该用户所有评分的汇总统计（各维度均值、总体均值、评分条数）。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT COUNT(*) as cnt,
                      ROUND(AVG(e.empathy), 2) as avg_empathy,
                      ROUND(AVG(e.naturalness), 2) as avg_naturalness,
                      ROUND(AVG(e.helpfulness), 2) as avg_helpfulness,
                      ROUND(AVG(e.safety), 2) as avg_safety,
                      ROUND(AVG(e.overall), 2) as avg_overall,
                      ROUND(AVG(e.latency_ms), 0) as avg_latency_ms,
                      ROUND(AVG(e.token_count), 0) as avg_token_count
               FROM evaluations e
               JOIN sessions s ON s.id = e.session_id
               WHERE s.user_id = ?""",
            (user_id,),
        )
        row = await cursor.fetchone()
        return {
            "count": row["cnt"],
            "avg_empathy": row["avg_empathy"],
            "avg_naturalness": row["avg_naturalness"],
            "avg_helpfulness": row["avg_helpfulness"],
            "avg_safety": row["avg_safety"],
            "avg_overall": row["avg_overall"],
            "avg_latency_ms": row["avg_latency_ms"],
            "avg_token_count": row["avg_token_count"],
        }
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────
# 个人成长报告
# ─────────────────────────────────────────────────────────────

_GROWTH_PROMPT = """\
你是一位温暖、睿智、观察力极强的成长伴侣，正在为"{name}"生成一份个人成长报告。
你拥有这位用户与树洞日记 AI 的全部对话摘要，以及他/她说过的部分原话。

这位用户对自己要求很严格，常常看不见自己的价值，总觉得自己不够好。
你的任务：从旁观者视角，真实、具体地告诉他/她——在这段时间里，你观察到了什么。

重要原则：
- 从数据中找具体证据，绝不空洞地说"你很棒"
- 语气像一位了解你多年的老朋友：温暖、直接、真实，不煽情
- 用"你"称呼用户，不用"用户"
- 聚焦于他/她已经拥有的，不聚焦于缺失的
- 如果数据不足以支持某结论，不要编造

【对话摘要（时间顺序，共 {n_summaries} 条）】
{summaries}

【用户说过的部分原话（最近 50 条）】
{user_messages}

请只输出 JSON，格式如下：
{{
  "period_summary": "一句话描述时间跨度，如「从2025年9月到2026年4月，共XX次对话」",
  "highlights": ["这段时间你经历/做到的具体事（3-5条，要有细节）"],
  "strengths": [
    {{"trait": "特质名称（2-4字）", "evidence": "从对话中找到的具体佐证（1-2句）"}}
  ],
  "values": ["你反复在意的人或事（2-3条）"],
  "growing": "你正在努力面对的一件事，用成长视角描述，不是批评（1-2句）",
  "letter": "亲爱的{name}，\\n\\n（200-300字，温暖且具体，有真实细节，结尾署名：\\n\\n一直在这里的树洞）"
}}"""


async def _generate_growth_report(user_id: int, conn) -> dict:
    """用全量历史数据生成成长报告，调用主 LLM。"""
    # 用户名
    cursor = await conn.execute("SELECT name FROM users WHERE id=?", (user_id,))
    row = await cursor.fetchone()
    name = (row["name"] if row else None) or "你"

    # 全量会话摘要
    cursor = await conn.execute(
        """SELECT summary, started_at FROM sessions
           WHERE user_id=? AND summary IS NOT NULL AND summary != ''
           ORDER BY started_at""",
        (user_id,),
    )
    summaries_rows = await cursor.fetchall()
    summaries_text = "\n".join(
        f"[{r['started_at'][:10]}] {r['summary']}" for r in summaries_rows
    ) or "（暂无摘要）"

    # 最近 50 条用户原话
    cursor = await conn.execute(
        """SELECT m.content FROM messages m
           JOIN sessions s ON s.id = m.session_id
           WHERE s.user_id=? AND m.role='user'
           ORDER BY m.id DESC LIMIT 50""",
        (user_id,),
    )
    msg_rows = await cursor.fetchall()
    user_msgs_text = "\n".join(f"- {r['content'][:200]}" for r in reversed(msg_rows)) or "（暂无）"

    prompt = _GROWTH_PROMPT.format(
        name=name,
        n_summaries=len(summaries_rows),
        summaries=summaries_text[:6000],
        user_messages=user_msgs_text[:2000],
    )

    from services.llm import get_client
    from config import OPENAI_MODEL
    import re as _re
    client = get_client()
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "只输出 JSON，不要有任何其他文字。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2500,
        ),
        timeout=90.0,
    )
    raw = resp.choices[0].message.content or ""
    raw = raw.strip()
    raw = _re.sub(r"^```json\s*", "", raw)
    raw = _re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return data


@app.get("/api/growth-report/list")
async def api_growth_report_list(user_id: int = Depends(get_current_user)):
    """返回该用户所有历史成长报告（按时间倒序）。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            "SELECT id, month_label, generated_at, report_json FROM growth_reports WHERE user_id=? ORDER BY id DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "month_label": r["month_label"],
                "generated_at": r["generated_at"],
                "report": json.loads(r["report_json"]),
            }
            for r in rows
        ]
    finally:
        await conn.close()


@app.post("/api/growth-report/generate")
async def api_growth_report_generate(user_id: int = Depends(get_current_user)):
    """生成本月成长报告。若本月已有则直接返回，不重复生成（除非传 force=true）。"""
    conn = await get_db()
    try:
        current_month = datetime.now().strftime("%Y年%m月")
        cursor = await conn.execute(
            "SELECT id, generated_at, report_json FROM growth_reports WHERE user_id=? AND month_label=? ORDER BY id DESC LIMIT 1",
            (user_id, current_month),
        )
        existing = await cursor.fetchone()
        if existing:
            return {
                "id": existing["id"],
                "month_label": current_month,
                "generated_at": existing["generated_at"],
                "report": json.loads(existing["report_json"]),
                "cached": True,
            }

        report = await _generate_growth_report(user_id, conn)
        cursor = await conn.execute(
            "INSERT INTO growth_reports (user_id, month_label, report_json) VALUES (?, ?, ?)",
            (user_id, current_month, json.dumps(report, ensure_ascii=False)),
        )
        await conn.commit()
        new_id = cursor.lastrowid
        return {
            "id": new_id,
            "month_label": current_month,
            "generated_at": datetime.now().isoformat(),
            "report": report,
            "cached": False,
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"模型返回格式错误：{e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        await conn.close()



if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print()
    print("  小伴 · 情感陪伴 已启动")
    print("  在浏览器中打开:  http://127.0.0.1:%s" % port)
    print("  本机访问也可用:  http://localhost:%s" % port)
    print()
    uvicorn.run(app, host=host, port=port)
