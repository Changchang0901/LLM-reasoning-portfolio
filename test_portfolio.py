import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main"))
sys.path.insert(0, str(ROOT / "rollout_analysis"))
from solution_generator import SolutionGenerator
from solution_grader import CodeVerificationGrader
from file_utils.parse_dataset_tex import parse_chapters, chapter_to_json
from stage4_error_classifier import ErrorClassifier
from scripts.rollout_stats import feature_extractors


class PortfolioTests(unittest.TestCase):
    def test_generic_latex_conversion_preserves_statement(self):
        chapters = parse_chapters(ROOT / "examples" / "demo_dataset.tex")
        self.assertEqual(list(chapters), ["demo_constant_acceleration"])
        with tempfile.TemporaryDirectory() as folder:
            for title, text in chapters.items():
                chapter_to_json(Path(folder), title, text, update=False)
            problem = json.loads((Path(folder) / "demo_constant_acceleration.json").read_text())
            self.assertIn("constant acceleration", problem["problem_details"]["Problem Statement"])
            self.assertEqual(problem["model_solutions"], [])

    def test_candidate_outcomes_and_fixture_identity(self):
        grader = CodeVerificationGrader(timeout_seconds=5, quiet=True)
        paths = sorted((ROOT / "data" / "demo").glob("*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            problem = json.loads(path.read_text())
            self.assertFalse(problem["provenance"]["benchmark_data"])
            for model in problem["model_solutions"]:
                result = grader._grade_solution(model["attempts"][0], problem)
                self.assertEqual(result["verified"], int(model["model"] == "demo-correct"), result)

    def test_generation_resume_and_prompt_without_network(self):
        # Exercise the original orchestration logic without constructing an API client.
        generator = SolutionGenerator.__new__(SolutionGenerator)
        generator.model = "demo-correct"
        generator.overwrite_attempts = False
        problem = json.loads((ROOT / "data/demo/demo_constant_acceleration.json").read_text())
        generator._load_problem = lambda _: problem
        result = generator._prepare_problem_generation(Path("fixture.json"), 3)
        self.assertEqual(result["attempt_to_generate"], 2)
        self.assertIn("displacement(v0, a, t)", result["user_prompt"])

    def test_error_category_normalization(self):
        self.assertEqual(ErrorClassifier._normalize_category("sign error"), "mathematical")
        self.assertEqual(ErrorClassifier._normalize_category("code bug"), "executional")

    def test_nonterminating_candidate_is_rejected(self):
        problem = json.loads((ROOT / "data/demo/demo_constant_acceleration.json").read_text())
        attempt = {"detailed_solution": "```python\ndef displacement(v0, a, t):\n    while True:\n        pass\n```"}
        result = CodeVerificationGrader(timeout_seconds=1, quiet=True)._grade_solution(attempt, problem)
        self.assertEqual(result["verified"], 0)
        self.assertIn("time", result.get("error", "").lower())

    def test_missing_tokenizer_is_not_a_zero_token_response(self):
        feature_extractors.TIKTOKEN_AVAILABLE = False
        features = feature_extractors.extract_all_features("A short response.")
        self.assertEqual(features["length_chars"], 17)
        self.assertIsNone(features["length_tokens"])


if __name__ == "__main__":
    unittest.main()
