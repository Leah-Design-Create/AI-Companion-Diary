# -*- coding: utf-8 -*-
"""调用 OpenAI 兼容 API"""
import json
import os
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


def get_client():
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
        base_url=OPENAI_BASE_URL,
    )


# ---------- 工具定义 ----------
SEARCH_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "搜索专业心理健康知识库。"
            "当用户提出与焦虑、情绪、心理、压力、恐惧、性格等相关的知识性或概念性问题时调用，"
            "例如：焦虑有哪些症状、焦虑是性格吗、怎么缓解压力、什么是恐惧症。"
            "以下场景不要调用：闲聊打招呼、问你的名字或感受、聊食物/娱乐/天气/个人经历/偏好等生活话题、"
            "以及用户只是在倾诉情绪而没有提出知识性问题的情况。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，提取用户问题的核心内容",
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_RESULT_PREFIX = (
    "以下是知识库中检索到的内容，若与用户问题相关请优先参考，用口语化方式自然表达，"
    "不要出现「参考资料」「书中」「资料显示」等字样。"
    "若内容与用户问题无关，按你自己的知识正常回答即可。\n\n"
)


def _build_msgs(messages: list[dict], extra_system: str) -> list[dict]:
    """合并 extra_system 到第一条 system message。"""
    system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    if extra_system:
        system = extra_system.rstrip() + "\n\n" + (system or "")
    if system and messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": system}] + list(messages[1:])
    elif system:
        return [{"role": "system", "content": system}] + list(messages)
    return list(messages)


async def chat(messages: list[dict], extra_system: str = "") -> str:
    client = get_client()
    msgs = _build_msgs(messages, extra_system)
    temp = 0.15 if extra_system else 0.9
    r = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=temp,
        max_tokens=1024,
        extra_body={"enable_thinking": False},
    )
    return (r.choices[0].message.content or "").strip()


async def chat_with_knowledge(
    messages: list[dict],
    extra_system: str = "",
    rag_fn=None,  # async (query: str) -> list[str]
    force_tool: bool = False,
) -> tuple[str, int]:
    """带 search_knowledge 工具的对话。
    若模型调用工具，执行 rag_fn 获取结果后二次调用得到最终回复。
    rag_fn 为 None 时退化为普通 chat。
    返回 (reply, total_tokens)。
    """
    client = get_client()
    msgs = _build_msgs(messages, extra_system)

    _no_think = {"enable_thinking": False}

    if rag_fn is None:
        r = await client.chat.completions.create(
            model=OPENAI_MODEL, messages=msgs, temperature=0.9, max_tokens=1024,
            extra_body=_no_think,
        )
        tokens = r.usage.total_tokens if r.usage else 0
        return (r.choices[0].message.content or "").strip(), tokens

    tc = {"type": "function", "function": {"name": "search_knowledge"}} if force_tool else "auto"
    r = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=0.9,
        max_tokens=1024,
        tools=[SEARCH_KNOWLEDGE_TOOL],
        tool_choice=tc,
        extra_body=_no_think,
    )
    msg = r.choices[0].message
    tokens = r.usage.total_tokens if r.usage else 0

    if not msg.tool_calls:
        print(f"[FunctionCall] 模型直接回答（未调用工具）finish_reason={r.choices[0].finish_reason}")
        return (msg.content or "").strip(), tokens

    # 模型决定调用工具
    tool_call = msg.tool_calls[0]
    try:
        args = json.loads(tool_call.function.arguments)
        query = args.get("query", "")
    except Exception:
        query = ""

    print(f"[FunctionCall] search_knowledge(query={query!r})")
    rag_texts = await rag_fn(query) if query else []
    tool_content = _TOOL_RESULT_PREFIX + "\n---\n".join(rag_texts[:6]) if rag_texts else "知识库中未找到相关内容。"

    msgs_with_result = msgs + [
        {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        },
    ]

    r2 = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs_with_result,
        temperature=0.15,
        max_tokens=1024,
        extra_body={"enable_thinking": False},
    )
    tokens += r2.usage.total_tokens if r2.usage else 0
    return (r2.choices[0].message.content or "").strip(), tokens


async def chat_stream(messages: list[dict], extra_system: str = ""):
    client = get_client()
    msgs = _build_msgs(messages, extra_system)
    temp = 0.15 if extra_system else 0.9
    stream = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=temp,
        max_tokens=1024,
        stream=True,
        extra_body={"enable_thinking": False},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def chat_stream_with_knowledge(
    messages: list[dict],
    extra_system: str = "",
    rag_fn=None,  # async (query: str) -> list[str]
    force_tool: bool = False,
):
    """带 search_knowledge 工具的流式对话。
    第一次调用非流式（检测工具调用），有工具调用则执行 RAG 后再流式输出。
    """
    client = get_client()
    msgs = _build_msgs(messages, extra_system)

    if rag_fn is None:
        async for chunk in chat_stream(messages, extra_system):
            yield chunk
        return

    tc = {"type": "function", "function": {"name": "search_knowledge"}} if force_tool else "auto"
    r = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=0.8,
        max_tokens=1024,
        tools=[SEARCH_KNOWLEDGE_TOOL],
        tool_choice=tc,
        extra_body={"enable_thinking": False},
    )
    msg = r.choices[0].message

    if not msg.tool_calls:
        content = (msg.content or "").strip()
        print(f"[FunctionCall] 未返回工具调用（finish_reason={r.choices[0].finish_reason}），直接回答")
        if content:
            yield content
        return

    # 模型调用了工具
    tool_call = msg.tool_calls[0]
    try:
        args = json.loads(tool_call.function.arguments)
        query = args.get("query", "")
    except Exception:
        query = ""

    print(f"[FunctionCall] search_knowledge(query={query!r})")
    rag_texts = await rag_fn(query) if query else []
    tool_content = _TOOL_RESULT_PREFIX + "\n---\n".join(rag_texts[:6]) if rag_texts else "知识库中未找到相关内容。"

    msgs_with_result = msgs + [
        {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        },
    ]

    # 第二次调用：流式输出最终回复
    stream = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs_with_result,
        temperature=0.15,
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
