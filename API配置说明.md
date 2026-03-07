# API 配置说明（BASE_URL 与 MODEL）

你的 Key 填好了仍报错，多半是 **OPENAI_BASE_URL** 或 **OPENAI_MODEL** 和当前使用的 Key 所属平台不一致。请按下面对照修改 `.env`。

---

## 常见服务商

| 服务商 | OPENAI_BASE_URL | OPENAI_MODEL |
|--------|-----------------|--------------|
| **OpenAI 官方** | `https://api.openai.com/v1` | `gpt-4o-mini` 或 `gpt-4o` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` |
| **阿里 通义千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` 或 `qwen-plus` |
| **月之暗面 Kimi** | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` 或 `moonshot-v1-32k` |
| **智谱 ChatGLM** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` 或 `glm-4` |
| **腾讯 混元** | `https://api.hunyuan.tencent.com/v1` | `hunyuan-lite` 或 `hunyuan-standard` |

---

## 示例：DeepSeek

```env
OPENAI_API_KEY=sk-你的DeepSeek密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

---

## 示例：通义千问

```env
OPENAI_API_KEY=你的通义Key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo
```

---

## 示例：Kimi（月之暗面）

```env
OPENAI_API_KEY=你的Kimi密钥
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_MODEL=moonshot-v1-8k
```

---

## 示例：OpenAI 官方

```env
OPENAI_API_KEY=sk-你的OpenAI密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

---

**注意：**

1. 修改 `.env` 后要**重新启动** `run.bat` 或 `python main.py` 才会生效。
2. 若使用国内中转/代理，请按对方文档填写 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。
3. 智谱、字节等部分厂商的接口地址或模型名可能随版本更新，以官方文档为准。
