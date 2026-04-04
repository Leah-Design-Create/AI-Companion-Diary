# -*- coding: utf-8 -*-
"""天气服务：通过 open-meteo.com（无需 API Key）获取当前天气。"""
import httpx

# WMO 天气码 → (中文描述, emoji)
_WMO: dict[int, tuple[str, str]] = {
    0:  ("晴", "☀️"),
    1:  ("晴间多云", "🌤️"),
    2:  ("多云", "⛅"),
    3:  ("阴天", "☁️"),
    45: ("有雾", "🌫️"),
    48: ("有雾", "🌫️"),
    51: ("毛毛雨", "🌦️"),
    53: ("小雨", "🌧️"),
    55: ("中雨", "🌧️"),
    61: ("小雨", "🌧️"),
    63: ("中雨", "🌧️"),
    65: ("大雨", "🌧️"),
    71: ("小雪", "🌨️"),
    73: ("中雪", "❄️"),
    75: ("大雪", "❄️"),
    77: ("冰粒", "🌨️"),
    80: ("阵雨", "🌦️"),
    81: ("中等阵雨", "🌧️"),
    82: ("强阵雨", "⛈️"),
    85: ("阵雪", "🌨️"),
    86: ("强阵雪", "❄️"),
    95: ("雷阵雨", "⛈️"),
    96: ("雷阵雨伴冰雹", "⛈️"),
    99: ("强雷阵雨", "⛈️"),
}


async def get_weather(lat: float, lon: float) -> dict:
    """调用 open-meteo.com 获取当前天气（不需要 API Key）。"""
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,weathercode,relative_humidity_2m"
        "&timezone=auto"
    )
    city_url = (
        f"https://nominatim.openstreetmap.org/reverse"
        f"?lat={lat}&lon={lon}&format=json&accept-language=zh-CN"
    )

    async with httpx.AsyncClient(timeout=10) as client:
        w_resp = await client.get(weather_url)
        w_resp.raise_for_status()
        w_data = w_resp.json()

        city = ""
        try:
            c_resp = await client.get(city_url, headers={"User-Agent": "treeholediary/1.0"})
            if c_resp.status_code == 200:
                addr = c_resp.json().get("address", {})
                city = addr.get("city") or addr.get("town") or addr.get("county") or ""
        except Exception:
            pass

    current = w_data.get("current", {})
    code = int(current.get("weathercode", 0))
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")

    desc, emoji = _WMO.get(code, ("未知天气", "🌡️"))
    temp_str = f"{round(temp)}°C" if temp is not None else ""
    feels_str = f"体感 {round(feels)}°C" if feels is not None else ""

    parts = [p for p in [city, desc, temp_str, feels_str] if p]
    summary = f"{emoji} " + "，".join(parts)

    return {
        "desc": desc,
        "emoji": emoji,
        "temp": round(temp) if temp is not None else None,
        "feels_like": round(feels) if feels is not None else None,
        "humidity": humidity,
        "city": city,
        "summary": summary,
    }
