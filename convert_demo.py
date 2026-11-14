"""Exercise the generic LaTeX parser and export a local JSONL file."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from file_utils.parse_dataset_tex import parse_chapters, chapter_to_json

if __name__ == "__main__":
    target = ROOT / "output" / "converted"
    for title, content in parse_chapters(ROOT / "examples" / "demo_dataset.tex").items():
        chapter_to_json(target, title, content, update=False)
    with (target / "demo.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(target.glob("*.json")):
            handle.write(json.dumps(json.loads(path.read_text(encoding="utf-8"))) + "\n")
    print("Local conversion complete: output/converted/demo.jsonl")
