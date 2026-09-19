"""Per-conversation traveller profile that persists to disk, so the agent
'remembers' each user (type / budget / interests / dislikes / red-lines) without
different visitors on a public deploy contaminating each other's memory."""
import json
import os

_DIR = os.path.join(os.path.dirname(__file__), "data", "profiles")
_SINGLE = {"出行类型", "预算", "同行"}          # these overwrite; others append
_CURRENT = {"cid": "local"}


def set_context(cid: str):
    """Select whose profile subsequent calls read/write (call once per turn)."""
    _CURRENT["cid"] = (cid or "local")[:64]


def _path() -> str:
    safe = "".join(c for c in _CURRENT["cid"] if c.isalnum() or c in "-_") or "local"
    return os.path.join(_DIR, safe + ".json")


def load_profile() -> dict:
    try:
        with open(_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_pref(field: str, value: str) -> str:
    field = (field or "其他").strip()
    value = (value or "").strip()
    if not value:
        return "（空偏好，未记录）"
    profile = load_profile()
    if field in _SINGLE:
        profile[field] = value
    else:
        lst = profile.setdefault(field, [])
        if value not in lst:
            lst.append(value)
    os.makedirs(_DIR, exist_ok=True)
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return f"已记住 {field}：{value}"


def profile_text() -> str:
    profile = load_profile()
    if not profile:
        return "（暂无已知偏好）"
    return "；".join(f"{k}：{'、'.join(v)}" if isinstance(v, list) else f"{k}：{v}"
                     for k, v in profile.items())
