"""Web front-end for the travel-planning agent.

Wraps the same agent (tools + RAG + memory) in a small Flask server so it can be
used in a browser. Reuses tools.py / rag.py / memory.py unchanged.
Run:  pip install flask  →  python app.py  →  open http://127.0.0.1:5000
"""
import datetime
import json
import os
import time
from collections import defaultdict

import requests
from flask import Flask, request, jsonify, send_file

import config
import memory
from tools import TOOLS, dispatch
from rag import retrieve

app = Flask(__name__)
_CHAT_URL = config.BASE_URL.rstrip("/") + "/chat/completions"


def _llm(messages):
    """Call the OpenAI-compatible chat endpoint directly via requests.
    Avoids httpx's ascii-only header path that broke on the deploy host."""
    r = requests.post(
        _CHAT_URL,
        headers={"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"},
        json={"model": config.MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

SYSTEM_TEMPLATE = (
    "你是一个懂『约束』、会『应变』、因人而异的旅行规划助手，帮用户安排欧洲城市的行程。"
    "需要实时或事实性信息时，主动调用工具（查天气、查景点）；"
    "用户透露长期偏好时，按字段调用 remember_preference（field=出行类型/预算/兴趣/不喜欢/同行/红线，value=具体内容）记下来。\n"
    "除了排出可执行行程，你还要做到几件多数助手不做的事：\n"
    "1. 预算护栏：若用户给了预算，估算每项花费、把行程控制在预算内；接近或超支时主动提示，并给更省的替代。\n"
    "2. 精力节奏（尤其带小孩）：给每一天标注『强度：轻 / 中 / 高』，不要把累的活动堆在一起，"
    "主动安排午休和缓冲时间，别把一天排满。\n"
    "3. 行程小结：每份行程末尾用一小段给出『预计总花费』『每日强度』，以及你为满足用户约束"
    "（出行类型 / 预算 / 天气）所做的关键取舍。\n"
    "4. 因人而异：先判断出行类型（独行 / 情侣 / 带娃 / 银发 / 商务 / 打卡特种兵 等），"
    "用不同策略——独行重安全与灵活，情侣重氛围与慢节奏，银发少走路、重无障碍，"
    "打卡型讲效率与最短路线，商务讲省心与就近；拿不准就先用一句话问清。\n"
    "5. 随时应变：当用户在途中报告变化（下雨、航班延误、景点关门、时间或体力变化），"
    "不要推倒重来，只在现有行程上做最小改动，并说明『改了哪里、为什么』。\n"
    "6. 路线可算不可猜：排定某一天的景点后，调用 optimize_route 得到最省脚程的顺序，"
    "并在行程中标出『已按地理位置优化，全天步行约 X 公里』；不要自己臆测距离。\n"
    "7. 红线优先：用户设定的『红线』（如怕早起、恨排队、一天最多 N 个点、必须午休、忌口）是不可违反的硬约束，"
    "规划必须满足，并在小结里逐条回应『红线：X → 已满足/如何满足』。\n"
    "8. 已订即锁定：当用户告知已预订的项目（机票/门票/酒店，通常含日期时间），把它作为固定锚点写入行程对应时间、"
    "标注为已订，围绕它安排其它活动，不要改动或删除它。\n"
    "回答要具体、可执行，说明『为什么这么排』，用 Markdown 分日组织。\n"
    "当你给出或更新一份行程时，在回答的最末尾附一个 ```plan 代码块，内含 JSON（供界面渲染成可编辑卡片），"
    "格式：{{\"days\":[{{\"day\":1,\"title\":\"当天主题\",\"intensity\":\"轻|中|高\","
    "\"items\":[{{\"time\":\"上午|中午|下午|晚上\",\"name\":\"地点或活动\",\"note\":\"简短说明，如免费/时长/门票\",\"fixed\":false}}]}}]}}"
    "（用户已预订/锁定的项把 fixed 设为 true）。"
    "代码块之外照常用中文讲解，不要在正文里重复 JSON。\n"
    "【已知的用户偏好】{profile}\n"
    "【参考资料（可能相关，酌情使用）】\n{context}"
)

# Per-conversation state so multiple visitors on a public deploy don't collide.
histories = defaultdict(list)

# Simple abuse / cost guards (in-memory, single process) for the public demo.
DAILY_CAP = int(os.getenv("DAILY_CAP", "200"))
PER_IP_HOUR = int(os.getenv("PER_IP_HOUR", "20"))
_day = {"date": None, "count": 0}
_ip_hits = defaultdict(list)


def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def _rate_limited(ip):
    """Return a friendly message if the request should be blocked, else None."""
    today = datetime.date.today().isoformat()
    if _day["date"] != today:
        _day["date"], _day["count"] = today, 0
    if _day["count"] >= DAILY_CAP:
        return "今天的体验额度用完啦（这是作者自费的 Demo，设了每日上限）。欢迎看代码，或明天再来 🙌"
    now = time.time()
    hits = [t for t in _ip_hits[ip] if now - t < 3600]
    if len(hits) >= PER_IP_HOUR:
        return "你这一小时试得有点多啦，歇会儿再来～（Demo 限流）"
    hits.append(now)
    _ip_hits[ip] = hits
    _day["count"] += 1
    return None


def build_system(user_query):
    return {"role": "system",
            "content": SYSTEM_TEMPLATE.format(profile=memory.profile_text(), context=retrieve(user_query, k=3))}


def run_turn(history, user_input):
    """Run one user turn on the given conversation; return (final_reply, [tool steps])."""
    steps = []
    history.append({"role": "user", "content": user_input})
    while True:
        messages = [build_system(user_input)] + history
        msg = _llm(messages)
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        history.append(assistant)

        if not tool_calls:
            return assistant["content"], steps
        for tc in tool_calls:
            fn = tc["function"]
            args = json.loads(fn.get("arguments") or "{}")
            steps.append({"name": fn["name"], "args": args})
            result = dispatch(fn["name"], args)
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "chat.html"))


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = data.get("message", "").strip()
    cid = (data.get("cid") or "local")[:64]
    if not user_input:
        return jsonify({"error": "empty"}), 400
    limited = _rate_limited(_client_ip())
    if limited:
        return jsonify({"reply": limited, "steps": []})
    memory.set_context(cid)
    try:
        reply, steps = run_turn(histories[cid], user_input)
        return jsonify({"reply": reply, "steps": steps})
    except Exception as e:  # noqa: BLE001 - surface any error to the UI for the demo
        import traceback
        tb = traceback.format_exc()
        traceback.print_exc()
        files = [ln.strip() for ln in tb.splitlines()
                 if ln.strip().startswith('File "') and ("/app/" in ln or "requests" in ln)]
        return jsonify({"error": f"{type(e).__name__}: {e} @@ " + " ⟶ ".join(files[:6])}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    cid = ((request.json or {}).get("cid") or "local")[:64]
    histories[cid].clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"旅行规划 Agent 网页版 → http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
