# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Stage 4: Classify error types in incorrect model attempts.

Given distilled CoT steps (from Stage 2) and golden solution steps (from Stage 1),
identifies and classifies error types into four global categories:
factual, mathematical, logical, executional.

Single-pass approach: batch all incorrect attempts through LLM, then apply
deterministic normalization to map any non-standard categories to the four
global categories.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from rollout_analyzer_base import RolloutAnalyzerBase
from utils import format_golden_steps_for_prompt, sanitize_filename
from schemas import Stage4Output, ErrorEntry, get_timestamp


def _repair_json_latex(text: str) -> str:
    """Repair JSON strings that contain unescaped LaTeX backslashes.

    LLM responses often include LaTeX like \\varphi, \\mathcal in JSON
    string values, which are invalid JSON (\\v, \\m are not valid escapes).
    This function escapes lone backslashes that aren't valid JSON escapes.
    """
    # Valid JSON escape sequences after backslash
    valid_json_escapes = set('"\\bfnrtu/')

    result = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
            result.append(ch)
            i += 1
        elif in_string and ch == '\\':
            if i + 1 < len(text):
                next_ch = text[i + 1]
                if next_ch in valid_json_escapes:
                    # Valid JSON escape - keep as is
                    result.append(ch)
                    result.append(next_ch)
                    i += 2
                else:
                    # Not a valid JSON escape - double the backslash
                    result.append('\\\\')
                    i += 1
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


# Deterministic overrides: map common non-standard category names to one of the
# four global categories (factual, mathematical, logical, executional).
LOCAL_TO_GLOBAL_OVERRIDES = {
    "arithmetic error": "mathematical",
    "arithmetic/numerical error": "mathematical",
    "integration error": "mathematical",
    "algebraic simplification error": "mathematical",
    "algebraic error": "mathematical",
    "sign error": "mathematical",
    "sign mistake": "mathematical",
    "calculation error": "mathematical",
    "numerical error": "mathematical",
    "differentiation error": "mathematical",
    "error propagation": "mathematical",
    "code bug": "executional",
    "implementation bug": "executional",
    "code error": "executional",
    "translation error": "executional",
    "irrelevant": "logical",
}


class ErrorClassifier(RolloutAnalyzerBase):
    """Stage 3: Classify errors in incorrect model attempts."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.prompt_template = self._load_prompt_template("stage4_error_prompt.txt")

    @staticmethod
    def _normalize_category(category: str) -> str:
        """Map known local subcategories to their global parent."""
        if not category:
            return category or ""
        return LOCAL_TO_GLOBAL_OVERRIDES.get(category.lower(), category)

    def _extract_json_from_response(self, response: str) -> Dict:
        """Override to handle LaTeX backslashes in JSON responses."""
        # Try standard parsing first
        try:
            return super()._extract_json_from_response(response)
        except (ValueError, json.JSONDecodeError):
            pass

        # Try with LaTeX repair
        repaired = _repair_json_latex(response)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block with repair
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', repaired, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding balanced braces with repair
        brace_start = repaired.find('{')
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(repaired[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(repaired[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        # Last resort: try to find any JSON-like structure in the repaired text
        # and do a more aggressive repair (strip trailing incomplete content)
        brace_start = response.find('{')
        if brace_start != -1:
            # Try to repair and extract - even with truncated responses
            partial = _repair_json_latex(response[brace_start:])
            # Try to find the largest valid JSON substring
            for end_pos in range(len(partial) - 1, 0, -1):
                if partial[end_pos] == '}':
                    try:
                        return json.loads(partial[:end_pos + 1])
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Could not extract valid JSON from response: {response[:500]}...")

    def _format_cot_steps(self, step_texts: List[Dict]) -> str:
        """Format CoT step texts for the prompt."""
        parts = []
        for st in step_texts:
            parts.append(f"### CoT Step {st['step_number']}\n{st['verbatim_text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _separate_code_step(step_texts: List[Dict], boundaries: List[Dict] = None):
        """Separate the code step (last step) from mathematical steps.

        Returns (math_steps, code_text_or_None).
        """
        if not step_texts:
            return step_texts, None
        # Check via boundaries: synthetic code step has start_line == -1
        if boundaries and boundaries[-1].get("start_line") == -1:
            return step_texts[:-1], step_texts[-1].get("verbatim_text", "")
        # Fallback: check if last step text looks like golden code step
        last_text = step_texts[-1].get("verbatim_text", "")
        if last_text.lstrip().startswith("## Answer Requirements") or last_text.lstrip().startswith("## Code"):
            return step_texts[:-1], last_text
        return step_texts, None

    @staticmethod
    def _format_code_comparison(golden_code: str, cot_code: str) -> str:
        """Format a code comparison section for the prompt."""
        if not golden_code and not cot_code:
            return ""
        parts = ["## Code Comparison",
                  "Compare the golden code implementation against the model's code. "
                  "Pay special attention to bugs introduced only in the code.",
                  "",
                  "### Golden Code (correct implementation)"]
        parts.append(golden_code if golden_code else "(No code in golden solution)")
        parts.append("")
        parts.append("### Model's Code (from the INCORRECT attempt)")
        parts.append(cot_code if cot_code else "(No code provided by model)")
        return "\n".join(parts)

    def classify_all_errors(
        self,
        dataset_name: str,
        problem_ids: List[str],
        target_models: List[str],
        stage1_outputs_dir: Optional[Path] = None,
        stage2_outputs_dir: Optional[Path] = None,
        save_outputs: bool = True,
        skip_existing: bool = True,
        error_analysis_subdir: str = "error_analysis",
    ) -> Dict[str, List[Stage4Output]]:
        """
        Classify errors for all incorrect attempts across problems.

        Single-pass: batch all incorrect attempts through LLM, apply
        deterministic normalization, and save.

        Returns:
            Dict mapping problem_id -> list of Stage4Output
        """
        safe_analyzer = sanitize_filename(self.model_name)

        if stage1_outputs_dir is None:
            stage1_outputs_dir = self.output_dir / "stage1" / dataset_name
        if stage2_outputs_dir is None:
            stage2_outputs_dir = self.output_dir / "stage2" / safe_analyzer / dataset_name

        all_results: Dict[str, List[Stage4Output]] = {}

        for problem_id in problem_ids:
            if not self.quiet:
                print(f"\n{'='*60}")
                print(f"Stage 3: Error classification for {problem_id}")
                print(f"{'='*60}")

            # Load stage1 output (golden steps)
            stage1_file = stage1_outputs_dir / f"{problem_id}_{safe_analyzer}_stage1.json"
            if not stage1_file.exists():
                print(f"Stage 1 output not found: {stage1_file}")
                continue
            with open(stage1_file) as f:
                stage1_output = json.load(f)

            # Separate golden math steps from code step
            golden_step_texts = stage1_output["step_texts"]
            golden_math_steps, _ = self._separate_code_step(
                golden_step_texts, stage1_output.get("split_boundaries")
            )
            golden_steps_formatted = format_golden_steps_for_prompt(golden_math_steps)
            golden_code = stage1_output.get("code", "")

            # Load problem for problem statement
            problem_path = self.project_root / "data" / dataset_name / f"{problem_id}.json"
            problem = self._load_problem(problem_path)
            problem_statement = problem["problem_details"]["Problem Statement"]

            # Collect all incorrect stage2 outputs for this problem
            tasks = []
            output_subdir = f"{error_analysis_subdir}/{problem_id}"

            for target_model in target_models:
                safe_model = sanitize_filename(target_model)

                # Find all stage2 files for this problem+model
                for stage2_file in sorted(stage2_outputs_dir.glob(
                    f"{problem_id}_{safe_analyzer}_{safe_model}_attempt*_stage2.json"
                )):
                    with open(stage2_file) as f:
                        stage2_output = json.load(f)

                    # Only process incorrect attempts
                    if stage2_output.get("attempt_is_correct", True):
                        continue

                    attempt_idx = stage2_output["attempt_index"]

                    # Check skip
                    stage4_filename = (
                        f"{problem_id}_{safe_analyzer}_{safe_model}"
                        f"_attempt{attempt_idx}_stage4.json"
                    )
                    if skip_existing:
                        existing = self._load_output(stage4_filename, subdir=output_subdir)
                        if existing:
                            if not self.quiet:
                                print(f"  Skipping {target_model}/attempt{attempt_idx} (exists)")
                            all_results.setdefault(problem_id, []).append(
                                Stage4Output.from_dict(existing)
                            )
                            continue

                    # Separate CoT math steps from code step
                    cot_step_texts = stage2_output["step_texts"]
                    cot_boundaries = stage2_output.get("split_boundaries", [])
                    cot_math_steps, cot_code = self._separate_code_step(
                        cot_step_texts, cot_boundaries
                    )

                    # Build prompt with explicit code comparison
                    cot_steps_formatted = self._format_cot_steps(cot_math_steps)
                    code_comparison_section = self._format_code_comparison(
                        golden_code, cot_code
                    )
                    prompt = self.prompt_template.format(
                        problem_statement=problem_statement,
                        golden_steps_formatted=golden_steps_formatted,
                        cot_steps_formatted=cot_steps_formatted,
                        code_comparison_section=code_comparison_section,
                    )

                    tasks.append({
                        "prompt": prompt,
                        "problem_id": problem_id,
                        "problem_path": str(problem_path),
                        "target_model": target_model,
                        "attempt_idx": attempt_idx,
                        "stage2_file": str(stage2_file),
                        "stage1_file": str(stage1_file),
                        "stage4_filename": stage4_filename,
                    })

            if not tasks:
                if not self.quiet:
                    print(f"  No new incorrect attempts to process for {problem_id}")
                continue

            # Batch LLM calls
            if not self.quiet:
                print(f"  Classifying {len(tasks)} incorrect attempts...")

            prompts = [t["prompt"] for t in tasks]
            system_prompt = (
                "You are an expert theoretical physicist analyzing mathematical errors. "
                "Respond only with valid JSON matching the requested structure."
            )

            with tqdm(total=len(prompts), desc=f"Stage 3: {problem_id}",
                      disable=self.quiet) as pbar:
                results = self._call_llm_for_json_batch(
                    prompts, system_prompt, pbar=pbar
                )

            task_outputs = []

            for task, response in zip(tasks, results):
                if response is None:
                    if not self.quiet:
                        print(f"  Error: no response for {task['target_model']}/attempt{task['attempt_idx']}")
                    task_outputs.append(None)
                    continue

                try:
                    errors = []
                    for e in response.get("errors", []):
                        cat = self._normalize_category(
                            e.get("error_category", "unknown")
                        )
                        errors.append(ErrorEntry(
                            cot_step_number=e.get("cot_step_number", 0),
                            error_category=cat,
                            error_scope="global",
                            is_new_category=False,
                            explanation=e.get("explanation", ""),
                            severity=e.get("severity", "major"),
                        ))

                    primary = self._normalize_category(
                        response.get("primary_error_category", "unknown")
                    )
                    # Fallback: if primary is empty/invalid, infer from first major error
                    VALID_CATS = {"factual", "mathematical", "logical", "executional"}
                    if primary.lower() not in VALID_CATS:
                        major_errors = [e for e in errors if e.severity == "major"]
                        if major_errors:
                            primary = major_errors[0].error_category
                        elif errors:
                            primary = errors[0].error_category
                        else:
                            primary = "unknown"

                    output = Stage4Output(
                        problem_id=task["problem_id"],
                        problem_path=task["problem_path"],
                        timestamp=get_timestamp(),
                        analyzer_model=self.model,
                        target_model=task["target_model"],
                        attempt_index=task["attempt_idx"],
                        attempt_is_correct=False,
                        stage2_reference=task["stage2_file"],
                        stage1_reference=task["stage1_file"],
                        errors=errors,
                        primary_error_category=primary,
                        error_summary=response.get("error_summary", ""),
                    )
                    task_outputs.append(output)
                except Exception as e:
                    if not self.quiet:
                        print(f"  Error parsing response for attempt{task['attempt_idx']}: {e}")
                    task_outputs.append(None)

            # Save common_errors.json
            all_cats = set()
            for output in task_outputs:
                if output:
                    for err in output.errors:
                        all_cats.add(err.error_category)
            for output in all_results.get(problem_id, []):
                for err in output.errors:
                    all_cats.add(err.error_category)

            if all_cats:
                self._save_output(
                    {
                        "problem_id": problem_id,
                        "error_categories": sorted(all_cats),
                        "total_categories": len(all_cats),
                        "timestamp": get_timestamp(),
                    },
                    "common_errors.json",
                    subdir=output_subdir,
                )
                if not self.quiet:
                    print(f"  Saved common_errors.json with {len(all_cats)} categories")

            # Save individual stage4 outputs
            for task, output in zip(tasks, task_outputs):
                if output is None:
                    continue
                if save_outputs:
                    self._save_output(
                        output.to_dict(),
                        task["stage4_filename"],
                        subdir=output_subdir,
                    )
                all_results.setdefault(problem_id, []).append(output)

            if not self.quiet:
                valid_count = sum(1 for o in task_outputs if o is not None)
                print(f"  Completed: {valid_count}/{len(tasks)} attempts classified")

        return all_results
