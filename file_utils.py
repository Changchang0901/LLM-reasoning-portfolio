import simplejson as json

from pathlib import Path

def load_json_file(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path: Path, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def remove_all_solutions(file_path: Path):
    data = load_json_file(file_path)
    if "model_solutions" in data:
        data["model_solutions"] = []
    save_json_file(file_path, data)
    print(f"Removed all solutions for {file_path.name}")
