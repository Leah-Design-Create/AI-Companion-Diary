# -*- coding: utf-8 -*-
"""情感陪伴 AI - 主入口：FastAPI + 聊天 / 总结 / RAG / 久未登录提醒"""
import base64
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import OPENAI_API_KEY
from db import init_db, get_db, ensure_user
from prompts import COMPANION_SYSTEM, build_chat_context
from services.llm import chat, chat_stream
from services.anxiety import analyze_anxiety
from services.mood import analyze_mood
from services.embedding import get_embedding
from services.rag import get_relevant_context, _extract_keywords
from services.intent import detect_intent
from services.summary import generate_summary
from services.reminder import get_reminder_if_inactive
from services.tts import synthesize_to_mp3

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


def _is_rag_skip_follow_up(msg: str) -> bool:
    """True 表示本条不查 RAG，只按对话历史继续聊（避免「详细说说」误命中知识库而答成别的）。"""
    s = (msg or "").strip()
    if len(s) <= 2:
        return True
    if len(s) <= 6 and any(s == p or s.startswith(p) or p in s for p in _FOLLOW_UP_PHRASES):
        return True
    return False


# 知识库相关词：用户消息含这些才认为在「问参考」，否则检索到也不注入，让模型用自身知识答（如美国总统、马斯克）
_KB_TOPIC_HINTS = ("焦虑", "情绪", "压力", "心理", "书中", "参考", "文档", "自救", "缓解", "症状", "恐惧", "障碍")


def _should_inject_rag(user_message: str, rag_texts: list[str], keywords: list[str]) -> bool:
    """检索到了也未必注入：只有用户明显在问知识库相关，或检索内容与用户问题有关时才注入。"""
    if not rag_texts:
        return False
    msg = (user_message or "").strip()
    # 用户明确在问书/参考/焦虑等，注入
    if any(h in msg for h in _KB_TOPIC_HINTS):
        return True
    # 至少有一个「实词」关键词出现在检索结果里，才认为相关（避免「想知道」「是谁」命中无关文档）
    combined = " ".join(rag_texts)
    stop = {"想", "知道", "是", "的", "你", "我", "有", "吗", "什么", "怎么", "如何", "哪些", "告诉", "认识", "谁"}
    for kw in keywords:
        if not kw or len(kw) < 2 or kw in stop:
            continue
        if kw in combined:
            return True
    return False


# ---------- 请求体 ----------
class ChatRequest(BaseModel):
    user_id: int = 1
    session_id: int | None = None  # 不传则新建会话
    message: str


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
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ---------- 对话 ----------
async def _get_today_mood(conn, user_id: int) -> str:
    """按当天该用户所有会话的对话综合计算心情，以天为单位。"""
    mood_messages = []
    try:
        cursor = await conn.execute(
            """SELECT m.role, m.content FROM messages m
               JOIN sessions s ON s.id = m.session_id
               WHERE s.user_id = ? AND date(m.created_at) = date('now', 'localtime')
               ORDER BY m.id ASC""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        mood_messages = [{"role": r["role"], "content": (r["content"] or "")} for r in rows]
    except Exception:
        try:
            cursor = await conn.execute(
                """SELECT m.role, m.content FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.user_id = ? AND date(s.started_at) = date('now', 'localtime')
                   ORDER BY m.id ASC""",
                (user_id,),
            )
            rows = await cursor.fetchall()
            mood_messages = [{"role": r["role"], "content": (r["content"] or "")} for r in rows]
        except Exception:
            pass
    if not mood_messages:
        return "平静"
    try:
        return await analyze_mood(mood_messages) or "平静"
    except Exception:
        return "平静"


async def _run_chat(user_id: int, session_id: Optional[int], message: str, image_path: Optional[str] = None):
    """内部：执行一轮对话（可带图片），返回 (session_id, reply, reminder)。"""
    await ensure_user(user_id)
    conn = await get_db()
    try:
        cursor = await conn.execute("SELECT last_login_at FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        last_login = row["last_login_at"] if row else None
        reminder = await get_reminder_if_inactive(last_login)
        if session_id is None:
            await conn.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
            await conn.commit()
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            session_id = (await cursor.fetchone())["id"]
        try:
            cursor = await conn.execute(
                "SELECT role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 30",
                (session_id,),
            )
        except Exception:
            cursor = await conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 30",
                (session_id,),
            )
        rows = await cursor.fetchall()
        history = []
        for r in rows:
            role, content = r["role"], r["content"]
            img_path = r["image_path"] if "image_path" in r.keys() and r["image_path"] else None
            if role == "user" and img_path:
                content = _image_path_to_content_parts(content or "", img_path, UPLOAD_DIR)
            history.append({"role": role, "content": content})
        _kws = _extract_keywords(message)[:6]
        had_any = False
        if _is_rag_skip_follow_up(message):
            rag_texts, rag_mode = [], "keyword"
            intent = "chat"
        else:
            intent = await detect_intent(message)
            if any(h in (message or "") for h in _KB_TOPIC_HINTS):
                intent = "ask_kb"
            if intent == "ask_kb":
                rag_texts, rag_mode = await get_relevant_context(message, user_id)
                had_any = bool(rag_texts)
                if not _should_inject_rag(message, rag_texts, _kws):
                    rag_texts = []
            else:
                rag_texts, rag_mode = [], "keyword"
            RAG_STATS["total"] += 1 if intent == "ask_kb" else 0
            if rag_texts:
                RAG_STATS["hits"] += 1
        cursor = await conn.execute("SELECT anxiety_detected FROM sessions WHERE id = ?", (session_id,))
        srow = await cursor.fetchone()
        has_anxiety = bool(srow and srow["anxiety_detected"])
        extra_system, rag_user_prefix = build_chat_context(rag_texts, has_anxiety)
        current_user_content = (rag_user_prefix + message) if rag_user_prefix else message
        if rag_user_prefix:
            current_user_content += "\n\n（若用户问的是参考主题相关的内容，只依据上述内容回答且回复自然，勿出现「参考」「资料」等词；若是闲聊则正常回复。）"
        if image_path:
            current_user_content = _image_path_to_content_parts(current_user_content, image_path, UPLOAD_DIR)
        messages = [{"role": "system", "content": COMPANION_SYSTEM}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_content})
        try:
            reply = await chat(messages, extra_system=extra_system)
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
        try:
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, image_path) VALUES (?, ?, ?, ?)",
                (session_id, "user", message, image_path),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, image_path) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", reply, None),
            )
        except Exception:
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", message),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", reply),
            )
        await conn.commit()
        mood = await _get_today_mood(conn, user_id)
        return {"session_id": session_id, "reply": reply, "reminder": reminder, "mood": mood or "平静"}
    finally:
        await conn.close()


@app.get("/api/mood")
async def api_mood(user_id: int = 1):
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
async def api_chat(req: ChatRequest):
    """发送一条消息，返回小伴回复。若久未登录会先返回提醒再正常对话。"""
    if not (OPENAI_API_KEY or "").strip():
        raise HTTPException(status_code=503, detail="未配置 API 密钥。请在项目文件夹中创建 .env 文件，填写：OPENAI_API_KEY=你的密钥")
    return await _run_chat(req.user_id, req.session_id, req.message, None)


@app.post("/api/chat/send")
async def api_chat_send(
    message: str = Form(""),
    user_id: int = Form(1),
    session_id: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
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
async def api_chat_stream(req: ChatRequest):
    """流式回复（仅返回 assistant 内容，不写库；如需写库可先非流式写再这里只做展示）。"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
    await ensure_user(req.user_id)
    conn = await get_db()
    try:
        session_id = req.session_id
        if session_id is None:
            await conn.execute("INSERT INTO sessions (user_id) VALUES (?)", (req.user_id,))
            await conn.commit()
            cursor = await conn.execute("SELECT last_insert_rowid() AS id")
            session_id = (await cursor.fetchone())["id"]
        cursor = await conn.execute(
            """SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 30""",
            (session_id,),
        )
        history = [{"role": r["role"], "content": r["content"]} for r in await cursor.fetchall()]
        had_any = False
        if _is_rag_skip_follow_up(req.message):
            rag_texts, rag_mode = [], "keyword"
        else:
            rag_texts, rag_mode = await get_relevant_context(req.message, req.user_id)
            had_any = bool(rag_texts)
            if not _should_inject_rag(req.message, rag_texts, _extract_keywords(req.message)[:6]):
                rag_texts = []
            RAG_STATS["total"] += 1
            if rag_texts:
                RAG_STATS["hits"] += 1
        pct = round(100 * RAG_STATS["hits"] / RAG_STATS["total"]) if RAG_STATS["total"] else 0
        _kws = _extract_keywords(req.message)[:6]
        kw_str = ", ".join(_kws)
        mode_cn = "语义" if rag_mode == "semantic" else "关键词"
        if _is_rag_skip_follow_up(req.message):
            print(f"[RAG] 跳过（追问）| 用户: {req.message[:20]}…")
        elif had_any and not rag_texts:
            print(f"[RAG] 跳过（与知识库无关）| 用户: {req.message[:30]}…")
        elif rag_texts:
            print(f"[RAG] {mode_cn} | 关键词: {kw_str} | 本次检索到 {len(rag_texts)} 条 | 命中率 {RAG_STATS['hits']}/{RAG_STATS['total']} ({pct}%)")
        else:
            print(f"[RAG] {mode_cn} | 关键词: {kw_str} | 未检索到 | 命中率 {RAG_STATS['hits']}/{RAG_STATS['total']} ({pct}%)")
        cursor = await conn.execute("SELECT anxiety_detected FROM sessions WHERE id = ?", (session_id,))
        srow = await cursor.fetchone()
        has_anxiety = bool(srow and srow["anxiety_detected"])
        extra_system, rag_user_prefix = build_chat_context(rag_texts, has_anxiety)
        current_user_content = (rag_user_prefix + req.message) if rag_user_prefix else req.message
        if rag_user_prefix:
            current_user_content += "\n\n（若用户问的是参考主题相关的内容，只依据上述内容回答且回复自然，勿出现「参考」「资料」等词；若是闲聊则正常回复。）"
        if rag_texts:
            preview = (rag_texts[0][:400] + "…") if len(rag_texts[0]) > 400 else rag_texts[0]
            print(f"[RAG] 注入参考预览: {preview}")
        messages = [{"role": "system", "content": COMPANION_SYSTEM}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_content})

        async def gen():
            full = []
            async for chunk in chat_stream(messages, extra_system=extra_system):
                full.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            # 流式结束后可异步写库（此处简化，仅做演示）
            content = "".join(full)
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", req.message),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", content),
            )
            await conn.commit()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    finally:
        await conn.close()


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
        await conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), summary = ?, anxiety_detected = ?, mood = ? WHERE id = ?",
            (summary, 1 if anxiety else 0, mood, req.session_id),
        )
        await conn.commit()
        return {"session_id": req.session_id, "summary": summary, "anxiety_detected": anxiety, "mood": mood}
    finally:
        await conn.close()


# ---------- 会话列表（左侧栏：不同话题，可继续上次对话）----------
@app.get("/api/sessions")
async def api_sessions(user_id: int = 1, limit: int = 50):
    """获取用户的会话列表（含未结束的），按开始时间倒序。用于左侧栏展示、点击后加载该会话消息。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, started_at, ended_at, summary, title
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
            })
        return out
    finally:
        await conn.close()


@app.get("/api/sessions/{session_id}/messages")
async def api_session_messages(session_id: int, user_id: int = 1):
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


@app.patch("/api/sessions/{session_id}")
async def api_rename_session(session_id: int, req: RenameSessionRequest, user_id: int = 1):
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
async def api_delete_session(session_id: int, user_id: int = 1):
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
async def api_summaries(user_id: int = 1, limit: int = 20):
    """获取用户的历史会话总结（生活记录），含该会话中用户分享的图片列表。"""
    conn = await get_db()
    try:
        cursor = await conn.execute(
            """SELECT id, started_at, ended_at, summary, anxiety_detected, mood
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
    rag_texts, mode = await get_relevant_context(q.strip(), user_id=1, limit=8)
    return {
        "query": q.strip(),
        "mode": mode,
        "count": len(rag_texts),
        "snippets": [s[:800] + ("…" if len(s) > 800 else "") for s in rag_texts],
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
    conn = await get_db()
    try:
        text_for_emb = (req.title or "") + "\n" + (req.content or "")
        emb = await get_embedding(text_for_emb)
        emb_json = json.dumps(emb) if emb else None
        try:
            await conn.execute(
                "INSERT INTO knowledge (title, content, source_url, embedding) VALUES (?, ?, ?, ?)",
                (req.title or "", req.content.strip(), req.source_url or "", emb_json),
            )
        except Exception as e:
            if "no such column" in str(e).lower() or "embedding" in str(e).lower():
                await conn.execute(
                    "INSERT INTO knowledge (title, content, source_url) VALUES (?, ?, ?)",
                    (req.title or "", req.content.strip(), req.source_url or ""),
                )
            else:
                raise
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() AS id")
        kid = (await cursor.fetchone())["id"]
        return {"id": kid}
    finally:
        await conn.close()


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
    """根据文件名后缀解析文本：支持 .txt、.pdf。"""
    name = (file.filename or "").lower()
    raw = file.file.read()
    if name.endswith(".txt"):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("gbk", errors="replace")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n\n".join(parts) if parts else ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")
    raise HTTPException(status_code=400, detail="仅支持 .txt 或 .pdf 文件")


@app.post("/api/knowledge/upload")
async def api_upload_knowledge(
    file: UploadFile = File(...),
    title: str = Form(""),
    source_url: str = Form(""),
):
    """上传文件到知识库：支持 .txt、.pdf，自动解析正文后入库。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="请选择文件")
    text = _extract_text_from_file(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析出文字")
    use_title = (title or "").strip() or (file.filename or "上传文件")
    content = text.strip()
    text_for_emb = use_title + "\n" + content
    emb = await get_embedding(text_for_emb)
    emb_json = json.dumps(emb) if emb else None
    conn = await get_db()
    try:
        try:
            await conn.execute(
                "INSERT INTO knowledge (title, content, source_url, embedding) VALUES (?, ?, ?, ?)",
                (use_title, content, (source_url or "").strip(), emb_json),
            )
        except Exception as e:
            if "no such column" in str(e).lower() or "embedding" in str(e).lower():
                await conn.execute(
                    "INSERT INTO knowledge (title, content, source_url) VALUES (?, ?, ?)",
                    (use_title, content, (source_url or "").strip()),
                )
            else:
                raise
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() AS id")
        kid = (await cursor.fetchone())["id"]
        return {"id": kid, "title": use_title}
    finally:
        await conn.close()


# ---------- 健康检查与久未登录提醒 ----------
@app.get("/api/check-in")
async def api_check_in(user_id: int = 1):
    """用户打开应用时调用：先根据上次登录判断是否久未登录并返回提醒，再更新 last_login。"""
    conn = await get_db()
    try:
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


if __name__ == "__main__":
    import uvicorn
    port = 8000
    host = "0.0.0.0"
    print()
    print("  小伴 · 情感陪伴 已启动")
    print("  在浏览器中打开:  http://127.0.0.1:%s" % port)
    print("  本机访问也可用:  http://localhost:%s" % port)
    print()
    uvicorn.run(app, host=host, port=port)
