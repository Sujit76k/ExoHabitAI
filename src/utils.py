"""
=====================================================
🚀 ExoHabitAI — Utility Functions (Production Version)
Central helpers used across training, API, dashboard.
=====================================================
"""

import os
import json
import datetime
from typing import Any, Dict


# =====================================================
# 📁 DIRECTORY HELPERS
# =====================================================

def ensure_dir_exists(path: str) -> None:
    """
    Safely create directory if it does not exist.

    Works even if nested folders missing.
    """
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Failed creating directory {path}: {e}")


def normalize_path(path: str) -> str:
    """
    Convert path to OS-safe absolute path.
    """
    return os.path.abspath(os.path.expanduser(path))


# =====================================================
# 🕒 TIME HELPERS
# =====================================================

def get_timestamp() -> str:
    """
    Returns formatted timestamp for logs/files.
    Example: 2026-02-19_21-45-03
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# =====================================================
# 💾 JSON UTILITIES
# =====================================================

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to JSON safely.

    - Creates folder automatically
    - Uses UTF-8 encoding
    - Prevents partial writes
    """
    try:
        file_path = normalize_path(file_path)
        folder = os.path.dirname(file_path)
        ensure_dir_exists(folder)

        temp_path = file_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # atomic replace (safer than direct write)
        os.replace(temp_path, file_path)

        print(f"💾 JSON saved → {file_path}")

    except Exception as e:
        print(f"❌ Failed to save JSON {file_path}: {e}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON safely.
    Returns empty dict if file missing.
    """
    try:
        file_path = normalize_path(file_path)

        if not os.path.exists(file_path):
            print(f"⚠️ JSON not found: {file_path}")
            return {}

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"❌ Failed to load JSON {file_path}: {e}")
        return {}


# =====================================================
# 📝 TEXT FILE UTILITIES
# =====================================================

def save_text(text: str, file_path: str) -> None:
    """
    Save plain text report safely.
    """
    try:
        file_path = normalize_path(file_path)
        ensure_dir_exists(os.path.dirname(file_path))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"📝 Text file saved → {file_path}")

    except Exception as e:
        print(f"❌ Failed saving text file {file_path}: {e}")


# =====================================================
# 🧠 SIMPLE LOGGER (LIGHTWEIGHT)
# =====================================================

def log(message: str, level: str = "INFO") -> None:
    """
    Lightweight console logger.

    Example:
        log("Model training started")
    """
    ts = get_timestamp()
    print(f"[{ts}] [{level}] {message}")