import os
import json

def ensure_dir_exists(path: str):
    """Create folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_json(data: dict, file_path: str):
    """Save dictionary as JSON file."""
    folder = os.path.dirname(file_path)
    ensure_dir_exists(folder)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
