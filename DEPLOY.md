# 部署说明：Vercel + Render

本项目支持两种部署方式：

- **Render**：部署完整后端（FastAPI + 前端静态资源），一站搞定。
- **Vercel + Render**：后端放在 Render，前端放在 Vercel（可选，用于加速或独立前端域名）。

---

## 一、部署到 Render（推荐，全栈）

Render 会运行 FastAPI，同时提供前端页面和所有 API。

### 1. 连接仓库

1. 登录 [Render](https://render.com)，点击 **New → Blueprint**。
2. 连接你的 GitHub 仓库（包含本项目的仓库）。
3. Render 会识别根目录的 `render.yaml`，按其中配置创建 Web Service。

### 2. 配置环境变量

在 Render 的 **Environment** 中至少添加：

| 变量 | 说明 | 必填 |
|------|------|------|
| `OPENAI_API_KEY` | 你的 LLM API Key（OpenAI / DeepSeek / 通义等） | ✅ |
| `OPENAI_BASE_URL` | API 基地址（见 README 常见服务商表） | 可选 |
| `OPENAI_MODEL` | 模型名 | 可选 |

其他可选变量：`OPENAI_EMBEDDING_MODEL`、`DB_PATH`、`INACTIVE_DAYS_FOR_REMINDER`、`DASHSCOPE_*` 等，见 README「配置项」。

### 3. 部署

- 若用 **Blueprint**：保存后 Render 会自动按 `render.yaml` 创建服务并部署。
- 若**手动**创建：选择 **Web Service**，Runtime 选 **Python**，Build 填 `pip install -r requirements.txt`，Start 填 `python main.py`。

### 4. 注意

- Render 的实例**磁盘非持久**：重启或重新部署后 SQLite 数据会清空。若需要持久化，可改用 Render 的 PostgreSQL 或外部数据库（需改代码）。
- 免费实例一段时间无访问会休眠，首次访问可能较慢。

部署完成后，访问 Render 提供的 URL（如 `https://ai-companion-diary.onrender.com`）即可使用。

---

## 二、前端部署到 Vercel（可选）

若希望前端单独放在 Vercel（例如用 Vercel 的 CDN 或自定义域名），可只把**静态前端**部署到 Vercel，API 仍请求 Render 上的后端。

### 1. 先完成 Render 部署

确保后端已在 Render 正常运行，并记下地址，例如：  
`https://ai-companion-diary.onrender.com`

### 2. 在 Vercel 连接同一仓库

1. 登录 [Vercel](https://vercel.com)，**Import** 本项目的 GitHub 仓库。
2. 在项目设置中配置：
   - **Framework Preset**：Other（或 None）。
   - **Build Command**：已由 `vercel.json` 指定为 `node scripts/write-config.js`。
   - **Output Directory**：`static`（已写在 `vercel.json`）。

### 3. 配置环境变量

在 Vercel 项目 **Settings → Environment Variables** 中添加：

| 变量 | 值 | 说明 |
|------|-----|------|
| `API_URL` | `https://你的Render应用.onrender.com` | 后端地址，**不要**末尾斜杠 |

例如：`API_URL` = `https://ai-companion-diary.onrender.com`

### 4. 部署

保存后重新 **Deploy**。构建时会执行 `node scripts/write-config.js`，把 `API_URL` 写入 `static/config.js`，前端会请求你填写的 Render 地址。

### 5. 若未配置 API_URL

未设置 `API_URL` 时，前端会使用空字符串（同源）。而 Vercel 上只有静态资源，没有 `/api`，因此所有接口会 404。**部署到 Vercel 时务必配置 `API_URL` 为你的 Render 地址。**

---

## 三、仅用 Render（不用 Vercel）

不配置 Vercel 时，只部署到 Render 即可：

- 在 Render 创建 Web Service，按上面「一、部署到 Render」操作。
- 访问 Render 给的域名即可，前端和后端同源，无需改 `config.js`。

---

## 四、本地与部署差异

| 项目 | 本地 | Render | Vercel 前端 |
|------|------|--------|-------------|
| 后端 | `python main.py`（端口 8000） | Render 自动注入 `PORT` | 无，请求 Render |
| 前端 API 地址 | 同源 `''` 或 file 时 `http://127.0.0.1:8000` | 同源 `''` | 由 `API_URL` 写入 `config.js` |
| 数据库 | 本地 `companion.db` | Render 实例内 SQLite（非持久） | - |

---

## 五、故障排查

- **Render 部署失败**：检查 Build Command 是否为 `pip install -r requirements.txt`，Start Command 是否为 `python main.py`，并确认已设置 `OPENAI_API_KEY`。
- **Vercel 页面能打开但接口报错**：检查 Vercel 的 `API_URL` 是否等于 Render 的完整地址（含 `https://`、无末尾 `/`），且 Render 服务已运行。
- **CORS 报错**：当前后端已设置 `allow_origins=["*"]`，一般不会出现；若你改了域名，可在 `main.py` 的 CORS 中间件里核对来源。

如有问题，可先看 Render / Vercel 的构建与运行日志。
