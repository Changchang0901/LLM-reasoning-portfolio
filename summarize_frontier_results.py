# Curated collaborative research code; see docs/PROVENANCE.md.
"""Summarize code-verification accuracy for selected TPBench models."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems_dir", default="data/demo")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--exclude_dir", action="append", default=[])
    parser.add_argument("--output", default="output/frontier_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.problems_dir)
    excluded = set(args.exclude_dir)
    stats = defaultdict(lambda: {"problems": set(), "attempts": 0, "graded": 0, "verified": 0})

    for path in root.rglob("*.json"):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts[:-1]):
            continue
        with path.open(encoding="utf-8") as handle:
            problem = json.load(handle)
        difficulty = str(problem.get("problem_metadata", {}).get("Difficulty level", "unknown")) or "unknown"
        problem_id = problem.get("problem_id", str(relative))
        by_model = {entry.get("model"): entry for entry in problem.get("model_solutions", [])}

        for model in args.models:
            entry = by_model.get(model)
            if not entry:
                continue
            for bucket in (difficulty, "overall"):
                current = stats[(model, bucket)]
                current["problems"].add(problem_id)
                for attempt in entry.get("attempts", []):
                    current["attempts"] += 1
                    result = attempt.get("verifier_result")
                    if isinstance(result, list):
                        result = result[0] if result else None
                    if isinstance(result, dict) and result:
                        current["graded"] += 1
                        current["verified"] += int(result.get("verified") == 1)

    rows = []
    for model in args.models:
        for difficulty in ["1", "2", "3", "4", "5", "unknown", "overall"]:
            current = stats.get((model, difficulty))
            if not current:
                continue
            accuracy = 100.0 * current["verified"] / current["graded"] if current["graded"] else 0.0
            rows.append({
                "model": model,
                "difficulty": difficulty,
                "problems": len(current["problems"]),
                "attempts": current["attempts"],
                "graded": current["graded"],
                "verified": current["verified"],
                "accuracy_percent": round(accuracy, 2),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [
            "model", "difficulty", "problems", "attempts", "graded", "verified", "accuracy_percent"
        ])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["difficulty"] == "overall":
            print(
                f"{row['model']}: {row['accuracy_percent']:.2f}% "
                f"({row['verified']}/{row['graded']} verified, {row['problems']} problems)"
            )
    print(f"CSV written to {output.resolve()}")


if __name__ == "__main__":
    main()
