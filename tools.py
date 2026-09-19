"""Tool definitions (OpenAI function-calling schema) + their implementations.
Three tools demonstrate the core agent abilities the JD asks for:
  - get_weather      : live tool call to a real external API (open-meteo, no key)
  - search_attractions: structured lookup over local data (data/pois.json)
  - remember_preference: writes to long-term memory (memory.py)
"""
import itertools
import json
import math
import os
import requests

from memory import save_pref

_GEOCACHE = os.path.join(os.path.dirname(__file__), "data", "geocache.json")

_POIS_PATH = os.path.join(os.path.dirname(__file__), "data", "pois.json")

# --- OpenAI-compatible tool schemas ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市未来几天的天气，用于安排行程（室内/室外）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市英文名，如 London / Paris / Rome"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "按城市和兴趣标签检索景点，返回名称、简介、是否适合带小孩、预算等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市英文名，如 London"},
                    "interest": {
                        "type": "string",
                        "description": "兴趣标签，如 历史 / 公园 / 博物馆 / 亲子 / 美食；留空则返回全部",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "把用户透露的长期偏好按字段记下来，供以后推荐使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string",
                              "description": "字段：出行类型 / 预算 / 兴趣 / 不喜欢 / 同行 / 红线 / 其他"},
                    "value": {"type": "string", "description": "具体内容，如 '带7岁孩子' 或 '不喜欢博物馆'"},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_route",
            "description": "按真实经纬度计算一组景点最省脚程的游览顺序与总步行距离。排某一天的景点时调用它拿到优化顺序。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市英文名，如 London"},
                    "places": {"type": "array", "items": {"type": "string"},
                               "description": "这一天计划去的景点名（中文或英文均可）"},
                },
                "required": ["city", "places"],
            },
        },
    },
]

_WEATHER_CODES = {
    0: "晴", 1: "多云转晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇", 51: "小毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨", 95: "雷雨",
}


def _get_weather(city: str) -> dict:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en"}, timeout=10,
        ).json()
        if not geo.get("results"):
            return {"error": f"找不到城市：{city}"}
        loc = geo["results"][0]
        fc = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": 3, "timezone": "auto",
            }, timeout=10,
        ).json()["daily"]
        days = []
        for i, date in enumerate(fc["time"]):
            days.append({
                "date": date,
                "weather": _WEATHER_CODES.get(fc["weather_code"][i], "未知"),
                "temp": f"{fc['temperature_2m_min'][i]}~{fc['temperature_2m_max'][i]}°C",
            })
        return {"city": loc["name"], "forecast": days}
    except Exception as e:  # noqa: BLE001 - demo: surface any network error to the model
        return {"error": f"天气查询失败：{e}"}


def _search_attractions(city: str, interest: str = "") -> dict:
    with open(_POIS_PATH, encoding="utf-8") as f:
        pois = json.load(f)
    city_key = city.strip().lower()
    hits = [p for p in pois if p["city"].lower() == city_key]
    if interest:
        hits = [p for p in hits if interest in p["tags"]]
    if not hits:
        return {"note": f"{city} 暂无匹配景点数据（demo 数据集较小）", "results": []}
    return {"results": hits}


def _haversine(a, b):
    """Great-circle distance in km between (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _route_len(order):
    return sum(_haversine(order[i][1], order[i + 1][1]) for i in range(len(order) - 1))


def _load_geocache():
    try:
        with open(_GEOCACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _geocode(place, city):
    """Free geocoding via OpenStreetMap Nominatim, cached to disk. Returns (lat, lon) or None."""
    key = f"{place}|{city}"
    cache = _load_geocache()
    if key in cache:
        return tuple(cache[key]) if cache[key] else None
    coord = None
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{place}, {city}", "format": "json", "limit": 1},
            headers={"User-Agent": "TravelAgentDemo/1.0 (portfolio)"}, timeout=10,
        ).json()
        if r:
            coord = (float(r[0]["lat"]), float(r[0]["lon"]))
    except Exception:  # noqa: BLE001 - geocoding is best-effort
        coord = None
    cache[key] = list(coord) if coord else None
    try:
        os.makedirs(os.path.dirname(_GEOCACHE), exist_ok=True)
        with open(_GEOCACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except OSError:
        pass
    return coord


def _optimize_route(city, places):
    """Order the given places for least walking. Known POIs use the local DB;
    unknown ones are geocoded for free via OSM. Optimal by brute force for small
    N (<=7), else nearest-neighbour."""
    with open(_POIS_PATH, encoding="utf-8") as f:
        pois = json.load(f)
    known = {p["name"]: (p["lat"], p["lon"])
             for p in pois if p["city"].lower() == city.strip().lower() and "lat" in p}
    matched, seen = [], set()
    for pl in places:
        if not pl:
            continue
        hit = next((name for name in known if pl in name or name in pl), None)
        label = hit or pl
        coord = known[hit] if hit else _geocode(pl, city)
        if coord and label not in seen:
            matched.append((label, coord)); seen.add(label)
    if len(matched) < 2:
        return {"note": "可定位的景点不足 2 个，未做路线优化。", "ordered": [m[0] for m in matched]}
    original_km = _route_len(matched)
    if len(matched) <= 7:
        best = min(itertools.permutations(matched), key=_route_len)
        route, method = list(best), "全局最优"
    else:
        remaining, route = matched[:], [matched[0]]
        remaining.pop(0)
        while remaining:
            last = route[-1][1]
            nxt = min(range(len(remaining)), key=lambda i: _haversine(last, remaining[i][1]))
            route.append(remaining.pop(nxt))
        method = "最近邻(近似)"
    opt_km = _route_len(route)
    return {"ordered": [m[0] for m in route], "method": method,
            "optimized_walk_km": round(opt_km, 1),
            "original_order_km": round(original_km, 1),
            "saved_km": round(max(original_km - opt_km, 0), 1)}


def dispatch(name: str, args: dict) -> str:
    """Run a tool by name and return a JSON string (fed back to the model)."""
    if name == "get_weather":
        result = _get_weather(args.get("city", ""))
    elif name == "search_attractions":
        result = _search_attractions(args.get("city", ""), args.get("interest", ""))
    elif name == "remember_preference":
        result = {"ok": save_pref(args.get("field", "其他"), args.get("value", ""))}
    elif name == "optimize_route":
        result = _optimize_route(args.get("city", ""), args.get("places", []))
    else:
        result = {"error": f"未知工具：{name}"}
    return json.dumps(result, ensure_ascii=False)
