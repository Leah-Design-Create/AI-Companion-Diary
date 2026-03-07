import os
import dashscope
from dashscope.audio.qwen_tts import SpeechSynthesizer

# 直接在代码里写 Key（本地测试可以，别传给别人）
dashscope.api_key = "sk-你在阿里云看到的那串Key"

text = "你好，我是小伴，很高兴认识你。"
response = SpeechSynthesizer.call(
    model="qwen3-tts-flash",
    api_key=dashscope.api_key,
    text=text,
    voice="Cherry",
)
print(response)