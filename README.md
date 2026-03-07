# 树洞日记 · 情感陪伴 AI

一款情感陪伴类产品：以**树洞**为形象，陪聊天、倾听生活、支持图片分享；内置**意图识别 + RAG 知识库**，可引用你上传的书/文章；每次对话可生成**生活记录**总结，并识别**当日心情**与**每条日记当时的心情**；久未登录时会主动问候。

前端为「树洞日记」风格：左侧会话列表（多话题、可新建/重命名/删除）、中间说话区、右侧日记页带日历与详情查看。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **日常陪聊** | 树洞以温暖、口语化方式回复；支持文字与图片，多轮对话。 |
| **多会话管理** | 左侧栏展示历史话题，可新建对话、重命名、删除；切换会话继续上次对话。 |
| **意图识别 + RAG** | 自动判断「闲聊」或「问知识库」；问知识库时从后台检索你上传的文档并引用，用户无感知。 |
| **焦虑分析** | 会话结束时分析是否存在焦虑/压力，并在该条日记上标记，后续回复更偏共情。 |
| **心情识别** | 按**当天**所有对话综合显示「今日心情」（聊天页右上角）；每条**日记卡片**右下角显示该次对话的当时心情。 |
| **生活记录（日记）** | 会话结束后生成简短总结；日记页右侧有日历，点击有记录的日期可定位，支持「查看详情」打开当天聊天记录。 |
| **语音与朗读** | 支持浏览器语音输入；回复可点击「朗读」（浏览器 TTS；可选配置 DashScope 高品质 TTS）。 |
| **久未登录提醒** | 超过设定天数未访问时，再次打开会收到树洞的问候/分享。 |

---

## 快速开始

### 1. 环境

- Python 3.10+
- 可用的 OpenAI 或兼容 API（如 DeepSeek、通义、Kimi 等）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API

复制环境变量示例并填写你的 API Key：

```bash
cp .env.example .env
```

在 `.env` 中设置：

- `OPENAI_API_KEY`：必填。
- `OPENAI_BASE_URL`：用哪家服务就填哪家的地址（见下表）。
- `OPENAI_MODEL`：用哪家就填哪家的模型名（见下表）。

**常见服务商：**

| 服务商 | OPENAI_BASE_URL | OPENAI_MODEL |
|--------|-----------------|---------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` / `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里 通义 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` / `qwen-plus` |
| 月之暗面 Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 ChatGLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` / `glm-4` |

**RAG 语义检索（可选）**  
在 `.env` 中增加 `OPENAI_EMBEDDING_MODEL`（如通义 `text-embedding-v3`、OpenAI `text-embedding-3-small`），并重新上传知识库文档后，RAG 会按语义检索；未配置时使用关键词检索。

**TTS（可选）**  
如需高品质朗读，可配置 `DASHSCOPE_API_KEY`、`DASHSCOPE_TTS_MODEL`、`DASHSCOPE_TTS_VOICE`；不配置时使用浏览器自带朗读。

### 4. 启动服务

在项目目录下执行：

```bash
python main.py
```

或双击 **`run.bat`**（Windows）。启动成功后终端会显示访问地址。

在浏览器打开 **http://127.0.0.1:8000** 或 **http://localhost:8000** 即可使用。（请通过该地址访问，不要直接打开 `static/index.html`，否则接口会失败。）

---

## 项目结构

```
├── main.py              # FastAPI 入口与路由（聊天、会话、日记、知识库、上传等）
├── config.py            # 配置（API、数据库、久未登录天数、可选 TTS）
├── db.py                # SQLite 初始化：用户、会话、消息、知识库、上传图片等
├── prompts.py           # 人设、焦虑分析、总结、意图识别、心情分析等提示词
├── services/
│   ├── llm.py           # 调用 OpenAI 兼容 API
│   ├── embedding.py     # 向量嵌入（RAG 语义检索）
│   ├── intent.py        # 意图识别（闲聊 vs 问知识库）
│   ├── rag.py           # 知识库检索（关键词 + 语义）
│   ├── anxiety.py       # 焦虑/压力检测
│   ├── mood.py          # 心情识别（当日 / 单次会话）
│   ├── summary.py       # 会话总结生成
│   ├── reminder.py      # 久未登录问候
│   └── tts.py           # 可选 DashScope TTS
├── static/
│   └── index.html       # 前端：树洞日记 UI（说话 / 日记、侧栏、日历、详情弹窗等）
├── .env.example         # 环境变量示例（勿提交 .env）
└── requirements.txt
```

---

## API 说明

| 接口 | 说明 |
|------|------|
| `GET /` | 前端页面（树洞日记）。 |
| `GET /api/check-in?user_id=1` | 打开应用时调用；久未登录则返回 `reminder`，并更新上次登录时间。 |
| `GET /api/mood?user_id=1` | 获取当日综合心情（按当天所有对话计算），用于聊天页右上角展示。 |
| `POST /api/chat` | 发送文字消息。请求体：`{ "user_id", "session_id", "message" }`。返回：`{ "session_id", "reply", "reminder?", "mood?" }`。 |
| `POST /api/chat/send` | 发送消息并可带一张图片（multipart：`message`, `user_id`, `session_id`, `image`）。返回含 `mood`。 |
| `GET /api/uploads/{filename}` | 获取用户上传的对话图片。 |
| `POST /api/session/end` | 结束当前会话：焦虑分析 + 总结 + 当时心情写入。请求体：`{ "user_id", "session_id" }`。返回：`{ "session_id", "summary", "anxiety_detected", "mood" }`。 |
| `GET /api/sessions?user_id=1&limit=50` | 会话列表（左侧栏），按开始时间倒序。 |
| `GET /api/sessions/{session_id}/messages?user_id=1` | 某会话的全部消息（含图片路径），用于加载历史与「查看详情」。 |
| `PATCH /api/sessions/{session_id}` | 重命名会话。请求体：`{ "title": "新标题" }`。 |
| `DELETE /api/sessions/{session_id}` | 删除该会话及全部消息。 |
| `GET /api/summaries?user_id=1&limit=20` | 历史会话总结（生活记录列表），每条含 `mood`、`images`；旧记录无 mood 时接口会尝试补全。 |
| `POST /api/tts` | 文本转语音（可选 DashScope TTS）。请求体：`{ "text": "..." }`。 |
| `GET /upload` | 知识库上传页（仅管理员）。上传 .txt / .pdf 后，RAG 在聊天时自动引用。 |
| `POST /api/knowledge/upload` | 上传文件到知识库（multipart：file、title、source_url）。 |
| `GET /api/knowledge` | 列出知识库条目。 |
| `POST /api/knowledge` | 添加知识。请求体：`{ "title", "content", "source_url" }`。 |
| `DELETE /api/knowledge/{kid}` | 删除某条知识。 |
| `GET /api/debug/rag?q=...` | 调试：查看问题 `q` 会检索到的 RAG 内容。 |

---

## 配置项

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | 必填 |
| `OPENAI_BASE_URL` | API 基地址 | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | 嵌入模型（RAG 语义检索） | 空则仅关键词 |
| `DB_PATH` | SQLite 库路径 | `companion.db` |
| `INACTIVE_DAYS_FOR_REMINDER` | 超过几天未登录触发问候 | `2` |
| `DASHSCOPE_API_KEY` | 通义 TTS（可选） | 空则用浏览器朗读 |
| `DASHSCOPE_TTS_MODEL` / `DASHSCOPE_TTS_VOICE` | TTS 模型与音色 | 可选 |

---

## 使用说明

1. **说话**：在输入框输入或上传图片发送。系统会先做**意图识别**：判定为「问知识库」则检索 RAG 并引用；「闲聊」则直接用模型知识回复。
2. **日记**：切到「日记」标签可看历史总结；每条卡片右下角为该次对话的**当时心情**；点击「查看详情」可打开该次完整聊天记录；右侧日历可点击有记录的日期快速定位。
3. **心情**：聊天页右上角为**当日综合心情**（按当天所有对话计算）；切换会话不会变，以天为单位。
4. **知识库**：访问 http://127.0.0.1:8000/upload 上传书或文章，聊天时自动在后端引用，用户端不展示该入口。
5. **久未登录**：超过配置天数未访问时，下次打开会请求 `GET /api/check-in`，满足条件则展示树洞问候。

---

## 意图识别与 RAG

每条用户消息会先做**意图识别**：判断是「闲聊」还是「问知识库」。仅当「问知识库」时才做 RAG 检索并注入参考。控制台会打印 `[意图] 闲聊` 或 `[意图] 问知识库` 及 RAG 检索情况。  
若配置了 `OPENAI_EMBEDDING_MODEL` 并已上传文档，会使用语义检索；否则使用关键词检索。出现 `[Embedding] API 调用失败` 时请检查嵌入模型与 Key；知识库无向量时请到 `/upload` 重新上传。

---

## 许可证

MIT（或按需修改）。
