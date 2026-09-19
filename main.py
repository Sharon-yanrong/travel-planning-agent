"""Travel-planning agent — a minimal CLI demo.

Shows the core building blocks an AI PM should understand hands-on:
  - tool calling  : the model decides when to call get_weather / search_attractions
  - RAG           : relevant guide passages are retrieved and injected each turn
  - memory        : user preferences persist across turns and across runs
Provider-agnostic: talks to any OpenAI-compatible endpoint (see config.py / .env).
"""
import json

from openai import OpenAI

import config
from tools import TOOLS, dispatch
from rag import retrieve
from memory import profile_text

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

SYSTEM_TEMPLATE = (
    "你是一个旅行规划助手，帮用户安排欧洲城市的行程。"
    "需要实时或事实性信息时，主动调用工具（查天气、查景点）。"
    "当用户透露长期偏好（带小孩、预算、兴趣）时，调用 remember_preference 记下来。"
    "回答要具体、可执行，并说明为什么这样安排。\n"
    "【已知的用户偏好】{profile}\n"
    "【参考资料（可能相关，酌情使用）】\n{context}"
)


def build_system(user_query: str) -> dict:
    content = SYSTEM_TEMPLATE.format(profile=profile_text(), context=retrieve(user_query, k=3))
    return {"role": "system", "content": content}


def run_turn(history: list, user_input: str) -> list:
    """One user turn: may loop through several tool calls before the final answer."""
    history.append({"role": "user", "content": user_input})
    while True:
        messages = [build_system(user_input)] + history
        resp = client.chat.completions.create(
            model=config.MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = resp.choices[0].message
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        history.append(assistant)

        if not msg.tool_calls:
            print("\n助手>", msg.content, "\n")
            return history

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"  [调用工具] {tc.function.name}({args})")
            result = dispatch(tc.function.name, args)
            history.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def main():
    print("旅行规划 Agent（输入 exit 退出）")
    print("试试：我下周带7岁孩子去伦敦玩3天，预算有限，帮我安排\n")
    history = []
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        history = run_turn(history, user_input)


if __name__ == "__main__":
    main()
