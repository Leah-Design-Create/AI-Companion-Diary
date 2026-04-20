# -*- coding: utf-8 -*-
"""DashScope 文生图：根据对话摘要+情绪生成心情画（REST API，异步任务模式）。"""
import asyncio
import uuid
from pathlib import Path

import httpx

from config import DASHSCOPE_API_KEY, IMAGE_GEN_MODEL, IMAGE_GEN_SIZE

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

_API_SUBMIT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
_API_TASK   = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

_PROMPT_GEN_SYSTEM = """\
你是一位擅长将情绪转化为自然意象的动画背景画师，风格融合吉卜力手绘温暖感与新海诚的光影细腻感。
根据用户今天的情绪和对话摘要，写一段中文的文生图 prompt。

核心原则：
- 用自然景象表达情绪（光线、云、草地、小镇、花田、海边、森林、季节等），不要出现手机、电脑等现代电子设备
- 不要直接描述对话内容，提炼情绪的"氛围"，用有故事感的自然场景呈现
- 风格：吉卜力 × 新海诚动画背景风格，手绘笔触，光感丰富，色彩温暖明亮，有生活气息和故事感
- 光线是核心：黄金时刻的光晕、穿透树叶的斑驳光、雨后清透的光，让画面有呼吸感
- 色调饱和度适中，不过艳，有温度感；不要出现人物面孔、文字
- 长度：30～50个中文字
- 只输出 prompt 本身，不要有任何解释"""

# 情绪词兜底 prompt（LLM 失败时使用）
_FALLBACK_PROMPTS: dict[str, str] = {
    "平静": "清晨阳光穿过森林树叶，斑驳光影洒在青草地，远处小鹿，吉卜力动画背景风格，暖绿与金色光晕",
    "疲惫": "黄昏小镇的石板路，路灯刚刚亮起，橘色暮光拉长影子，新海诚光感，暖橘与蓝紫渐变天空",
    "委屈": "雨天窗玻璃上的水珠，窗外模糊的绿意，室内一盏暖灯，吉卜力风格，蓝灰与暖黄对比",
    "焦虑": "大风吹过的草原，云影快速移动，远处一棵孤树弯腰，新海诚天空风格，蓝白云与翠绿草地",
    "烦躁": "海边礁石，浪打礁石溅起水花，傍晚橙红天空倒映海面，动画背景风格，橙红与深蓝强对比",
    "难过": "秋日黄昏的落叶小径，金黄树叶铺满地面，光线低斜温柔，吉卜力手绘感，暖金与棕褐色调",
    "低落": "阴天的山间小屋，炊烟缓缓升起，远山笼罩薄雾，吉卜力风格，灰蓝与苔绿，静谧低沉",
    "开心": "春日花田，粉色樱花漫天飞舞，阳光明媚，新海诚光感，粉白与天蓝，轻盈欢快",
    "兴奋": "夏夜祭典，灯笼光芒倒映河面，烟火绽放夜空，吉卜力 × 新海诚，暖金与深靛蓝，热烈喜悦",
    "期待": "清晨薄雾山谷，一列小火车从远处驶来，晨光穿透云层，新海诚光感，橙粉朝霞与薄雾",
    "放松": "午后草坡上，阳光懒洋洋，蒲公英随风飘散，吉卜力动画风格，暖黄绿与蓝天白云",
    "安心": "傍晚小屋亮起暖灯，窗外初雪飘落，壁炉光从窗缝透出，吉卜力手绘温暖感，暖琥珀与冷蓝雪色",
    "困惑": "黄昏森林中的岔路口，光线从树隙射入，远处隐约有光，新海诚光影，暖金光与深绿阴影",
}
_DEFAULT_FALLBACK = "黄昏时分的山间小镇，暖光从窗户透出，远山与云霞，吉卜力 × 新海诚动画背景风格，手绘温暖感，金橙与蓝紫"


async def _generate_image_prompt(mood: str, summary: str) -> str:
    """用 LLM 根据摘要+情绪生成定制化的图片 prompt。"""
    try:
        from services.llm import get_client
        from config import OPENAI_MODEL
        client = get_client()
        user_content = f"今天的情绪：{mood}\n\n对话摘要：{summary[:400]}"
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_GEN_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.9,
            max_tokens=120,
            extra_body={"enable_thinking": False},
        )
        prompt = (resp.choices[0].message.content or "").strip()
        if prompt:
            print(f"[ImageGen] 生成 prompt: {prompt[:80]}...")
            return prompt
    except Exception as e:
        print(f"[ImageGen] prompt 生成失败，使用兜底: {e}")
    return _FALLBACK_PROMPTS.get(mood, _DEFAULT_FALLBACK)


async def generate_mood_image(mood: str, summary: str | None = None) -> str | None:
    """
    根据对话摘要+情绪词生成心情画，返回图片 URL。
    失败或未配置时返回 None。
    """
    if not DASHSCOPE_API_KEY or not IMAGE_GEN_MODEL:
        print("[ImageGen] 未配置 DASHSCOPE_API_KEY 或 IMAGE_GEN_MODEL，跳过生图")
        return None

    if summary:
        art_prompt = await _generate_image_prompt(mood, summary)
    else:
        art_prompt = _FALLBACK_PROMPTS.get(mood, _DEFAULT_FALLBACK)

    full_prompt = f"{art_prompt}, Studio Ghibli anime background art style, Makoto Shinkai lighting, hand-painted texture, warm cinematic atmosphere, no text, no faces, high quality"
    size = IMAGE_GEN_SIZE or "1024*1024"

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": IMAGE_GEN_MODEL,
        "input": {"prompt": full_prompt},
        "parameters": {"size": size, "n": 1},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(_API_SUBMIT, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            task_id = data.get("output", {}).get("task_id")
            if not task_id:
                print(f"[ImageGen] 提交失败: {data}")
                return None
            print(f"[ImageGen] 任务已提交 task_id={task_id}")

            poll_headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
            for _ in range(9):
                await asyncio.sleep(3)
                pr = await client.get(_API_TASK.format(task_id=task_id), headers=poll_headers)
                pr.raise_for_status()
                pd = pr.json()
                status = pd.get("output", {}).get("task_status", "")
                if status == "SUCCEEDED":
                    results = pd.get("output", {}).get("results", [])
                    if results:
                        remote_url = results[0].get("url")
                        print(f"[ImageGen] 生图成功 mood={mood!r}，下载到本地...")
                        # 下载图片到本地 uploads，避免 DashScope URL 过期
                        local_url = await _download_to_local(client, remote_url)
                        return local_url or remote_url
                    return None
                if status in ("FAILED", "CANCELED"):
                    print(f"[ImageGen] 任务失败 status={status} detail={pd}")
                    return None
            print("[ImageGen] 超时，未获取到结果")
            return None

    except Exception as e:
        print(f"[ImageGen] 生图异常: {e}")
        return None


async def _download_to_local(client: httpx.AsyncClient, url: str) -> str | None:
    """把 DashScope 临时图片 URL 下载到本地 uploads/，返回本地访问路径。"""
    try:
        r = await client.get(url, timeout=30)
        r.raise_for_status()
        ext = "jpg"
        ct = r.headers.get("content-type", "")
        if "png" in ct:
            ext = "png"
        filename = f"mood_{uuid.uuid4().hex}.{ext}"
        filepath = UPLOADS_DIR / filename
        filepath.write_bytes(r.content)
        print(f"[ImageGen] 已保存到本地: {filename}")
        return f"/api/uploads/{filename}"
    except Exception as e:
        print(f"[ImageGen] 下载到本地失败，使用原始 URL: {e}")
        return None
