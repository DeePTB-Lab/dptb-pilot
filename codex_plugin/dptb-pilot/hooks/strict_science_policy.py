from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


SCIENCE_RE = re.compile(
    r"(?:"
    r"materials?|physics|electronic\s+structure|band\s*(?:structure|gap)|density\s+of\s+states|\bDOS\b|"
    r"transport|NEGF|molecular\s+dynamics|DFT|abacus|deeptb|tight[- ]binding|hamiltonian|"
    r"phonon|relaxation|geometry\s+optim|simulation|numerical\s+(?:calculation|analysis)|"
    r"材料|物理|电子结构|能带|态密度|输运|分子动力学|第一性原理|哈密顿|声子|结构优化|数值计算|科学计算"
    r")",
    re.IGNORECASE,
)

SHELL_TOOL_RE = re.compile(r"(?:exec|shell|terminal|bash|powershell|cmd|command)", re.IGNORECASE)
COMPUTE_COMMAND_RE = re.compile(
    r"(?:\bpython(?:3|\.exe)?\b|\bipython\b|\bjupyter\b|\bmatlab\b|\bjulia\b|\bRscript\b|"
    r"numpy|scipy|pymatgen|ase\b|torch|jax|tensorflow|calculator|wolfram|mathematica)",
    re.IGNORECASE,
)
COMPUTE_TOOL_RE = re.compile(
    r"(?:calculator|compute|simulation|materials?|physics|chemistry|numpy|scipy|pymatgen|"
    r"jupyter|python|matlab|wolfram|electronic|molecular|quantum)",
    re.IGNORECASE,
)
CODE_WRITE_RE = re.compile(r"(?:apply_patch|write_file|create_file|edit_file)", re.IGNORECASE)


def _read_payload() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_all_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_all_strings(item))
        return result
    return []


def _session_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("session_id") or payload.get("conversation_id") or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:128] or "unknown"


def _data_root() -> Path:
    configured = os.getenv("PLUGIN_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.dptb-pilot-codex/hook-data").expanduser().resolve()


def _state_path(payload: dict[str, Any]) -> Path:
    return _data_root() / "strict-science" / f"{_session_id(payload)}.json"


def _write_state(payload: dict[str, Any], scientific: bool, prompt: str) -> None:
    path = _state_path(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "session_id": _session_id(payload),
        "scientific": scientific,
        "timestamp": time.time(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_state(payload: dict[str, Any]) -> dict[str, Any]:
    path = _state_path(payload)
    if not path.is_file():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(state, dict) or time.time() - float(state.get("timestamp") or 0) > 1800:
        return {}
    return state


def _allow_dptb(tool_name: str) -> bool:
    normalized = tool_name.lower().replace("_", "-")
    return "dptb-pilot" in normalized or normalized.startswith("dptb-") or "dptb-route-scientific-task" in normalized


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=True))


def _handle_user_prompt(payload: dict[str, Any]) -> None:
    prompt = "\n".join(_all_strings({key: payload.get(key) for key in ("prompt", "user_prompt", "message", "content")}))
    if not prompt:
        prompt = "\n".join(_all_strings(payload))
    _write_state(payload, bool(SCIENCE_RE.search(prompt)), prompt)
    print("{}")


def _handle_pre_tool(payload: dict[str, Any]) -> None:
    state = _load_state(payload)
    if not state.get("scientific"):
        print("{}")
        return

    tool_name = str(payload.get("tool_name") or "")
    if _allow_dptb(tool_name):
        print("{}")
        return

    tool_text = "\n".join(_all_strings(payload.get("tool_input") or {}))
    blocks_shell = bool(SHELL_TOOL_RE.search(tool_name) and COMPUTE_COMMAND_RE.search(tool_text))
    blocks_compute_tool = bool(COMPUTE_TOOL_RE.search(tool_name))
    blocks_custom_code = bool(CODE_WRITE_RE.search(tool_name) and COMPUTE_COMMAND_RE.search(tool_text))
    if blocks_shell or blocks_compute_tool or blocks_custom_code:
        _deny(
            "Scientific calculations in this task must run through DPTB Pilot. "
            "Python, NumPy, pymatgen, custom formulas, and non-DPTB compute tools are blocked. "
            "Use dptb_route_scientific_task; if Pilot is unavailable, stop instead of falling back."
        )
        return
    print("{}")


def main() -> int:
    payload = _read_payload()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "user-prompt":
        _handle_user_prompt(payload)
        return 0
    if mode == "pre-tool":
        _handle_pre_tool(payload)
        return 0
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
