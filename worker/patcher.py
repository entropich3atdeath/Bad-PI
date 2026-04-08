"""
worker/patcher.py

Applies a config_delta dict to train.py.
Uses regex to replace top-level constant assignments.
Supports: int, float, str, bool values.

Example delta: {"DEPTH": 12, "learning_rate": 3e-4, "WINDOW_PATTERN": "SSL"}
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Any


def _fmt(val: Any) -> str:
    """Format a Python literal for insertion into source."""
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, float):
        # Preserve scientific notation for small numbers
        if abs(val) < 1e-2 and val != 0:
            return f"{val:.2e}"
        return repr(val)
    return repr(val)


def apply_delta(train_py: Path, delta: dict[str, Any], backup: bool = True) -> dict[str, Any]:
    """
    Apply delta to train_py in-place.
    Returns {key: old_value} for every key that was changed (for logging).
    """
    if backup:
        shutil.copy(train_py, train_py.with_suffix(".py.bak"))

    src = train_py.read_text()
    applied = {}

    for key, new_val in delta.items():
        # Match:  KEY = <anything up to newline or comment>
        pattern = rf"^({re.escape(key)}\s*=\s*)(.+?)(\s*(?:#.*)?)$"
        new_src, n = re.subn(
            pattern,
            lambda m: m.group(1) + _fmt(new_val) + m.group(3),
            src,
            flags=re.MULTILINE,
        )
        if n > 0:
            # Extract old value for logging
            m = re.search(pattern, src, re.MULTILINE)
            if m:
                applied[key] = m.group(2).strip()
            src = new_src
        else:
            print(f"  [patcher] WARNING: key '{key}' not found in {train_py.name}")

    train_py.write_text(src)
    return applied


def restore_backup(train_py: Path):
    bak = train_py.with_suffix(".py.bak")
    if bak.exists():
        shutil.copy(bak, train_py)


def read_current_config(train_py: Path, keys: list[str]) -> dict[str, Any]:
    """Extract current values of the given keys from train.py."""
    src = train_py.read_text()
    result = {}
    for key in keys:
        m = re.search(rf"^{re.escape(key)}\s*=\s*(.+?)(?:\s*#.*)?$", src, re.MULTILINE)
        if m:
            try:
                result[key] = eval(m.group(1).strip())  # safe for literals
            except Exception:
                result[key] = m.group(1).strip()
    return result
