# -*- coding: utf-8 -*-
"""天气服务：通过 wttr.in（无需 API Key）获取当前天气。"""
import httpx

# WMO 天气码 → (中文描述, emoji)
_WMO: dict[int, tuple[str, str]] = {
    113: ("晴", "☀️"),
    116: ("晴间多云", "🌤️"),
    119: ("多云", "⛅"),
    122: ("阴天", "☁️"),
    143: ("有雾", "🌫️"),
    176: ("局部小雨", "🌦️"),
    179: ("局部小雪", "🌨️"),
    182: ("冻雨", "🌧️"),
    185: ("冻毛毛雨", "🌧️"),
    200: ("雷阵雨", "⛈️"),
    227: ("飘雪", "🌨️"),
    230: ("暴雪", "❄️"),
    248: ("有雾", "🌫️"),
    260: ("冻雾", "🌫️"),
    263: ("毛毛雨", "🌦️"),
    266: ("毛毛雨", "🌦️"),
    281: ("冻毛毛雨", "🌧️"),
    284: ("冻毛毛雨", "🌧️"),
    293: ("小雨", "🌧️"),
    296: ("小雨", "🌧️"),
    299: ("中雨", "🌧️"),
    302: ("中雨", "🌧️"),
    305: ("大雨", "🌧️"),
    308: ("暴雨", "🌧️"),
    311: ("冻雨", "🌧️"),
    314: ("冻雨", "🌧️"),
    317: ("雨夹雪", "🌨️"),
    320: ("雨夹雪", "🌨️"),
    323: ("小雪", "🌨️"),
    326: ("小雪", "🌨️"),
    329: ("中雪", "❄️"),
    332: ("中雪", "❄️"),
    335: ("大雪", "❄️"),
    338: ("暴雪", "❄️"),
    350: ("冰粒", "🌨️"),
    353: ("阵雨", "🌦️"),
    356: ("强阵雨", "⛈️"),
    359: ("暴雨", "⛈️"),
    362: ("雨夹雪", "🌨️"),
    365: ("雨夹雪", "🌨️"),
    368: ("阵雪", "🌨️"),
    371: ("强阵雪", "❄️"),
    374: ("冰粒阵雨", "🌨️"),
    377: ("冰粒阵雨", "🌨️"),
    386: ("雷阵雨", "⛈️"),
    389: ("强雷阵雨", "⛈️"),
    392: ("雷阵雪", "⛈️"),
    395: ("强雷阵雪", "⛈️"),
}


async def get_weather(lat: float, lon: float) -> dict:
    """调用 wttr.in 获取当前天气（不需要 API Key）。"""
    url = f"https://wttr.in/{lat},{lon}?format=j1"

    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(url, headers={"User-Agent": "treeholediary/1.0"})
        resp.raise_for_status()
        data = resp.json()

    current = (data.get("current_condition") or [{}])[0]
    temp = int(current.get("temp_C", 0))
    feels = int(current.get("FeelsLikeC", temp))
    humidity = current.get("humidity", "")
    code = int(current.get("weatherCode", 113))

    area = (data.get("nearest_area") or [{}])[0]
    city = ""
    area_names = area.get("areaName") or []
    if area_names:
        city = area_names[0].get("value", "")

    desc, emoji = _WMO.get(code, ("未知天气", "🌡️"))
    parts = [p for p in [city, desc, f"{temp}°C", f"体感 {feels}°C"] if p]
    summary = f"{emoji} " + "，".join(parts)

    return {
        "desc": desc,
        "emoji": emoji,
        "temp": temp,
        "feels_like": feels,
        "humidity": humidity,
        "city": city,
        "summary": summary,
    }
