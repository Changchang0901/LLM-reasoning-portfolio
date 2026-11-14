"""Offline verification demo using only the two trusted educational fixtures."""
import copy
import json
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "main"))
from solution_grader import CodeVerificationGrader
from scripts.rollout_stats import feature_extractors


def run_demo():
    # Keep the default demo offline, even if an optional tokenizer is installed.
    feature_extractors.TIKTOKEN_AVAILABLE = False
    target = ROOT / "output" / "demo" / "problems"
    target.mkdir(parents=True, exist_ok=True)
    grader = CodeVerificationGrader(timeout_seconds=5, quiet=True)
    rows = []
    for path in sorted((ROOT / "data" / "demo").glob("*.json")):
        problem = json.loads(path.read_text(encoding="utf-8"))
        for model in problem["model_solutions"]:
            for attempt in model["attempts"]:
                result = grader._grade_solution(attempt, problem, model["model"])
                attempt["verifier_result"] = result
                features = feature_extractors.extract_all_features(attempt["detailed_solution"])
                rows.append({"problem_id": problem["problem_id"], "candidate": model["model"],
                             "verified": result.get("verified"), "features": features})
        (target / path.name).write_text(json.dumps(problem, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"kind": "educational fixture verification; not research results", "rows": rows}
    (target.parent / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("EDUCATIONAL DEMO: hand-written candidates; no model/API calls.")
    for row in rows:
        print(f"{row['problem_id']} | {row['candidate']} | verified={row['verified']}")
    expected = [1 if row["candidate"] == "demo-correct" else 0 for row in rows]
    if [row["verified"] for row in rows] != expected:
        raise RuntimeError("Demo outcomes do not match the fixture's expected outcomes")
    print("All four expected outcomes confirmed. Dashboard data: output/demo/problems")
    return report


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_demo()
