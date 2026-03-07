# 小伴 · 情感陪伴 AI

一款以**卡通形象小伴**为核心的情感陪伴产品：陪聊天、倾听生活琐事、分析焦虑倾向、接 RAG/外接文章、每次对话后生成生活记录总结，并在久未登录时主动问候。

---

## 角色设定

- **小伴**：陪聊型 AI，外形为卡通形象，像生活中的好朋友。
- 悉心倾听用户生活中的各种事情，处处心系用户。
- 可做用户的**生活记录搭子**：根据对话内容生成总结笔记。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **日常陪聊** | 用户输入对话，小伴以温暖、口语化的方式回复。 |
| **焦虑分析** | 在会话结束后分析对话中是否存在焦虑/压力，并影响后续回复的共情程度。 |
| **RAG / 知识库** | 后端知识库：管理员上传书/文章后，聊天时小伴自动在后台检索并引用，用户端不显示。 |
| **对话总结** | 每次会话结束后生成一篇简短「生活记录」总结，可在「生活记录」页查看。 |
| **久未登录提醒** | 超过 2～3 天未登录时，再次打开会收到小伴的问候/分享。 |

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

- `OPENAI_API_KEY`：必填，填你从对应平台获取的 API Key。
- `OPENAI_BASE_URL`：**用哪家服务就填哪家的地址**，见下表。
- `OPENAI_MODEL`：**用哪家就填哪家的模型名**，见下表。

**常见服务商对照表（BASE_URL 和 MODEL 必须与 Key 所属平台一致）：**

| 服务商 | OPENAI_BASE_URL | OPENAI_MODEL |
|--------|-----------------|--------------|
| OpenAI 官方 | `https://api.openai.com/v1` | `gpt-4o-mini` 或 `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` 或 `deepseek-reasoner` |
| 阿里 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` 或 `qwen-plus` |
| 月之暗面 Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` 或 `moonshot-v1-32k` |
| 智谱 ChatGLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` 或 `glm-4` |
| 字节 豆包 | `https://ark.cn-beijing.volces.com/api/v3` | 需在控制台查模型 ID（如 `ep-xxx`） |
| 腾讯 混元 | `https://api.hunyuan.tencent.com/v1` | `hunyuan-lite` 或 `hunyuan-standard` |
| 讯飞 星火 | 需用其 SDK 或专用接口，非标准 OpenAI 兼容 | — |
| 国内中转 / 代理 | 填你购买时给的 base URL | 填对方提供的模型名 |

若你用的是 **DeepSeek**，`.env` 里应类似：
```env
OPENAI_API_KEY=sk-你的DeepSeek密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```
若你用的是 **通义**，应类似：
```env
OPENAI_API_KEY=你的通义Key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo
```

**RAG 语义检索（可选）**  
若希望按「意思」检索知识库（而不只是关键词），在 `.env` 中增加：

- `OPENAI_EMBEDDING_MODEL`：嵌入模型名。  
  - 通义：`text-embedding-v3`  
  - OpenAI：`text-embedding-3-small`  

配置后需**重新上传**知识库文档，才会为每条知识生成向量；未配置或未重新上传时仍使用关键词检索。

### 4. 启动服务

在项目目录下执行：

```bash
python main.py
```

或双击运行 **`run.bat`**（Windows）。启动成功后终端会显示访问地址。

在浏览器地址栏输入并访问：

- **http://127.0.0.1:8000**  
或  
- **http://localhost:8000**

即可打开聊天界面。（请务必通过上述地址访问，不要直接双击打开 `static/index.html` 文件，否则接口会请求失败。）

### 浏览器访问不了时

1. **确认服务已启动**：运行 `python main.py` 后，终端应显示「小伴 · 情感陪伴 已启动」和访问地址，且无报错退出。
2. **用对地址**：在浏览器地址栏输入 `http://127.0.0.1:8000` 或 `http://localhost:8000`（不要用 `https`，不要漏掉端口 `:8000`）。
3. **端口被占用**：若 8000 端口被占用，可修改 `main.py` 末尾的 `port = 8000` 为其他端口（如 8080），再访问 `http://127.0.0.1:8080`。
4. **防火墙**：本机访问一般不受影响；若需局域网访问，请放行对应端口的入站规则。

---

## 项目结构

```
├── main.py           # FastAPI 入口与路由
├── config.py         # 配置（API、数据库、久未登录天数）
├── db.py             # SQLite 初始化与用户/会话/消息/知识库表
├── prompts.py        # 小伴人设、焦虑分析、总结、久未登录问候等提示词
├── services/
│   ├── llm.py        # 调用 OpenAI 兼容 API
│   ├── anxiety.py    # 焦虑/压力检测
│   ├── rag.py        # 知识库与近期总结检索（RAG）
│   ├── summary.py    # 会话总结生成
│   └── reminder.py   # 久未登录问候
├── static/
│   └── index.html    # 前端：卡通形象 + 聊天 + 生活记录
├── .env.example      # 环境变量示例
└── requirements.txt
```

---

## API 说明

- **`GET /`**  
  前端页面（聊天 + 生活记录）。

- **`GET /api/check-in?user_id=1`**  
  用户打开应用时调用：若久未登录则返回 `reminder`（小伴问候），并更新上次登录时间。

- **`POST /api/chat`**  
  发送一条消息，返回小伴回复。  
  请求体：`{ "user_id": 1, "session_id": null | number, "message": "..." }`  
  返回：`{ "session_id", "reply", "reminder" }`（首次或久未登录时可能有 `reminder`）。

- **`POST /api/chat/send`**  
  发送消息并可附带一张图片（multipart/form-data：`message`、`user_id`、`session_id`、`image`）。模型会看到图片内容并回复；生活记录总结中会记为「用户分享了图片」。

- **`GET /api/uploads/{filename}`**  
  获取用户上传的对话图片（仅单层文件名，用于前端展示）。

- **`POST /api/session/end`**  
  结束当前会话：分析焦虑并生成总结写入数据库。  
  请求体：`{ "user_id": 1, "session_id": number }`  
  返回：`{ "session_id", "summary", "anxiety_detected" }`。

- **`GET /api/sessions?user_id=1&limit=50`**  
  获取该用户的会话列表（含未结束的），用于左侧栏「不同话题」展示；按开始时间倒序。

- **`GET /api/sessions/{session_id}/messages?user_id=1`**  
  获取某会话的全部消息，用于点击某条会话后加载历史并继续对话。

- **`PATCH /api/sessions/{session_id}?user_id=1`**  
  重命名会话。请求体：`{ "title": "新标题" }`。

- **`DELETE /api/sessions/{session_id}?user_id=1`**  
  删除该会话及其全部消息。

- **`GET /api/summaries?user_id=1&limit=20`**  
  获取该用户的历史会话总结（生活记录列表）。

- **`GET /upload`**  
  **仅管理员**：知识库上传页。上传 .txt / .pdf 书或文章后，RAG 在聊天时自动引用，用户无感知。不对外展示，仅你知道该地址即可。

- **`POST /api/knowledge/upload`**  
  上传文件到知识库（multipart：file、title、source_url）。也可用接口批量导入。

- **`POST /api/knowledge`**  
  添加 RAG 知识/外接文章。  
  请求体：`{ "title": "", "content": "...", "source_url": "" }`。

- **`POST /api/chat/stream`**  
  流式回复（SSE），请求体同 `/api/chat`。

---

## 配置项

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | 必填 |
| `OPENAI_BASE_URL` | API 基地址 | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |
| `DB_PATH` | SQLite 库路径 | `companion.db` |
| `INACTIVE_DAYS_FOR_REMINDER` | 超过几天未登录触发问候 | `2` |

---

## 使用说明

1. **聊天**：在输入框输入内容发送。系统会先做**意图识别**（闲聊 vs 问知识库）：若判定为「问知识库」则检索并注入 RAG；若为「闲聊」则直接用模型原有知识回复，不查知识库。
2. **生活记录**：切换至「生活记录」标签可查看历史会话总结；关闭页面或结束会话时会自动调用「结束会话」并生成当次总结。
3. **焦虑分析**：在「结束会话」时自动执行，结果会体现在该条生活记录是否带「焦虑/压力」标记，并在后续对话中让回复更偏共情。
4. **外接文章**：通过 `POST /api/knowledge` 写入标题、内容与链接，对话时 RAG 会按关键词检索并注入上下文。
5. **久未登录**：超过配置天数未访问时，下次打开页面会先请求 `GET /api/check-in`，若满足条件会展示小伴的问候/分享。

---

## 意图识别与 RAG

每条用户消息会先经过**意图识别**（调用一次小模型）：判断是「闲聊」还是「问知识库」。只有判定为「问知识库」时才会做 RAG 检索并注入参考；闲聊直接不查库，避免误注入无关内容。控制台会打印 `[意图] 闲聊 -> 不查 RAG` 或 `[意图] 问知识库 | [RAG] …`。

## RAG 还是不行时

1. **看控制台日志**  
   每次发消息后，运行服务的终端会打印 `[意图]` 与 `[RAG]`：  
   - 若显示 **「语义」**：说明已用向量检索；若仍答不好，多半是知识库内容或 prompt 问题。  
   - 若显示 **「关键词」**：说明未用语义检索。请检查是否在 `.env` 中配置了 `OPENAI_EMBEDDING_MODEL`，并**重新上传**知识库文档（只有新上传的才会写向量）。

2. **出现 `[Embedding] API 调用失败`**  
   说明嵌入接口报错（如模型名错误、该平台不支持、或 Key 无权限）。通义请用 `text-embedding-v3`，并确认与对话使用同一套 Key 和 Base URL。

3. **出现「知识库暂无向量，请重新上传」**  
   说明已配置嵌入模型且调用成功，但库里没有带向量的文档。到 **http://127.0.0.1:8000/upload** 重新上传需要参与检索的文档即可。

---

## 发布到 GitHub

### 发布前必查（安全）

1. **绝不能提交的内容**（已写在 `.gitignore`，提交前请再确认）：
   - **`.env`**：里面是 API Key，一旦泄露需立即在对应平台重置密钥。
   - **`*.db`**：数据库里有对话、总结等用户数据，只应留在本机。
   - **`uploads/`**：用户上传的图片，隐私相关。

2. **提交前自检**（在项目目录下执行）：
   ```bash
   git status
   ```
   确认列表里**没有** `.env`、`companion.db`、`uploads/` 等。若出现，说明未被忽略，不要 `git add` 它们。

3. **建议保留在仓库里的**：
   - `.env.example`：示例配置，不含真实密钥，方便别人克隆后 `cp .env.example .env` 再填自己的 Key。

### 操作步骤

1. **在 GitHub 上新建仓库**
   - 打开 [github.com/new](https://github.com/new)，登录后点 “New repository”。
   - 仓库名自定（如 `ai-companion` 或 `树洞日记`）。
   - 选 Public，**不要**勾选 “Add a README”（本地已有项目）。
   - 创建后记下仓库地址，例如：`https://github.com/你的用户名/仓库名.git`。

2. **在本地初始化并推送**（在项目根目录 `AI情感陪伴` 下执行）：
   ```bash
   git init
   git add .
   git status
   ```
   再次确认没有 `.env`、`*.db`、`uploads/` 被加入。若有，用 `git reset HEAD .env` 等取消，并检查 `.gitignore`。

   ```bash
   git commit -m "Initial commit: 情感陪伴 AI 项目"
   git branch -M main
   git remote add origin https://github.com/你的用户名/仓库名.git
   git push -u origin main
   ```

3. **别人克隆后如何运行**
   - `git clone https://github.com/你的用户名/仓库名.git`
   - `cd 仓库名`
   - `cp .env.example .env`，编辑 `.env` 填入自己的 `OPENAI_API_KEY` 等。
   - `pip install -r requirements.txt`
   - `python main.py` 或双击 `run.bat`。

### 后续更新代码

```bash
git add .
git status
git commit -m "简短描述本次修改"
git push
```

## 许可证

MIT（或按你项目需求修改）。
