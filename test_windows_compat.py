# Curated collaborative research code; see docs/PROVENANCE.md.
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "main"))

from problem_processor_base import load_problem_with_lock, save_problem_with_lock
from solution_grader import CodeVerificationGrader


class WindowsCompatibilityTests(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.json"
            save_problem_with_lock({"value": 42, "model_solutions": []}, path)
            self.assertEqual(load_problem_with_lock(path)["value"], 42)

    def test_code_grader_subprocess(self):
        problem = {
            "problem_details": {
                "Code": "```python\ndef solve(x):\n    return x + 1\n```",
                "test": {
                    "arguments": [{"name": "x", "type": "float"}],
                    "test_cases": [{"inputs": {"x": 2.0}}],
                },
            }
        }
        attempt = {
            "detailed_solution": "```python\ndef solve(x):\n    return x + 1\n```"
        }
        grader = CodeVerificationGrader(timeout_seconds=3, quiet=True)
        result = grader._grade_solution(attempt, problem)
        self.assertEqual(result.get("verified"), 1, result)

    def test_grading_worker_does_not_inherit_api_keys(self):
        problem = {
            "problem_details": {
                "Code": "```python\ndef solve():\n    return 'missing'\n```",
                "test": {"arguments": [], "test_cases": [{"inputs": {}}]},
            }
        }
        attempt = {
            "detailed_solution": (
                "```python\nimport os\ndef solve():\n"
                "    return os.environ.get('OPENAI_API_KEY', 'missing')\n```"
            )
        }
        os.environ["OPENAI_API_KEY"] = "test-secret"
        try:
            result = CodeVerificationGrader(timeout_seconds=3, quiet=True)._grade_solution(
                attempt, problem
            )
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
        self.assertEqual(result.get("verified"), 1, result)


if __name__ == "__main__":
    unittest.main()
