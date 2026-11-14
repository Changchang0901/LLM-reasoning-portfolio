# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Stage 3: Match distilled CoT steps to golden solution steps.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm

from rollout_analyzer_base import RolloutAnalyzerBase
from utils import format_golden_steps_for_prompt, compute_coverage, sanitize_filename
from schemas import Stage3Output, MatchedStep, Stage3Summary, get_timestamp


class StepMatcher(RolloutAnalyzerBase):
    """
    Stage 3: Match distilled CoT steps to golden solution steps.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.prompt_template = self._load_prompt_template("stage3_match_prompt.txt")

    def match_steps(
        self,
        problem_path: Path,
        stage1_output: Dict,
        stage2_output: Dict
    ) -> Optional[Stage3Output]:
        """
        Match distilled CoT steps to golden steps.

        Args:
            problem_path: Path to problem JSON
            stage1_output: Stage 1 output dict with golden steps
            stage2_output: Stage 2 output dict with distilled CoT steps

        Returns:
            Stage3Output or None on error
        """
        problem = self._load_problem(problem_path)
        problem_id = problem.get("problem_id", problem_path.stem)

        # Format golden steps for prompt
        golden_steps_formatted = format_golden_steps_for_prompt(stage1_output.get("step_texts", []))

        # Format CoT steps for prompt
        cot_steps = stage2_output.get("step_texts", [])
        cot_steps_formatted = "\n\n".join([
            f"**CoT Step {step['step_number']}:**\n{step['verbatim_text']}"
            for step in cot_steps
        ])

        # Build prompt
        problem_statement = problem["problem_details"]["Problem Statement"]
        prompt = self.prompt_template.format(
            problem_statement=problem_statement,
            golden_steps_formatted=golden_steps_formatted,
            cot_steps_formatted=cot_steps_formatted
        )

        # Call LLM
        system_prompt = (
            "You are an expert theoretical physicist analyzing mathematical reasoning. "
            "Respond only with valid JSON matching the requested structure."
        )
        response = self._call_llm_for_json(prompt, system_prompt)

        # Parse matched steps
        matched_steps = []
        for s in response.get("matched_steps", []):
            # Get the corresponding CoT step text
            cot_step_num = s.get("cot_step_number", len(matched_steps) + 1)
            cot_step_text = ""
            for cot_step in cot_steps:
                if cot_step["step_number"] == cot_step_num:
                    cot_step_text = cot_step["verbatim_text"]
                    break

            matched_steps.append(MatchedStep(
                cot_step_number=cot_step_num,
                cot_step_text=cot_step_text,
                golden_step_mapping=s.get("golden_step_mapping"),
                mapping_confidence=s.get("mapping_confidence", "none"),
                mapping_explanation=s.get("mapping_explanation", ""),
                validity=s.get("validity", "unknown"),
                validity_explanation=s.get("validity_explanation", "")
            ))

        # Compute summary
        valid_count = sum(1 for s in matched_steps if s.validity == "valid")
        invalid_count = sum(1 for s in matched_steps if s.validity == "invalid")
        mapped_count = sum(1 for s in matched_steps if s.golden_step_mapping is not None)

        total_golden_steps = len(stage1_output.get("step_texts", []))
        coverage = compute_coverage(
            [{"golden_step_mapping": s.golden_step_mapping} for s in matched_steps],
            total_golden_steps
        )

        summary = Stage3Summary(
            total_cot_steps=len(matched_steps),
            mapped_steps=mapped_count,
            unmapped_steps=len(matched_steps) - mapped_count,
            valid_steps=valid_count,
            invalid_steps=invalid_count,
            coverage=coverage
        )

        # Build output
        output = Stage3Output(
            problem_id=problem_id,
            problem_path=str(problem_path),
            timestamp=get_timestamp(),
            analyzer_model=self.model,
            target_model=stage2_output["target_model"],
            attempt_index=stage2_output["attempt_index"],
            attempt_is_correct=stage2_output.get("attempt_is_correct"),
            cot_steps_reference=f"stage2/{problem_id}_stage2.json",
            golden_steps_reference=f"stage1/{problem_id}_stage1.json",
            matched_steps=matched_steps,
            summary=summary
        )

        return output

    def match_all_steps(
        self,
        dataset_name: str = "demo",
        stage1_outputs_dir: Optional[Path] = None,
        stage2_outputs_dir: Optional[Path] = None,
        save_outputs: bool = True,
        skip_existing: bool = True,
        problem_ids: Optional[List[str]] = None
    ) -> List[Stage3Output]:
        """
        Match all distilled CoT steps to golden steps using batch processing.

        Args:
            dataset_name: Dataset subdirectory name
            stage1_outputs_dir: Directory containing Stage 1 outputs
            stage2_outputs_dir: Directory containing Stage 2 outputs
            save_outputs: Whether to save individual JSON files
            skip_existing: Skip already processed rollouts
            problem_ids: List of specific problem IDs to process (None = all problems)

        Returns:
            List of Stage3Output objects
        """
        self.problems_dir = self.project_root / "data" / dataset_name

        safe_analyzer = sanitize_filename(self.model_name)

        if stage1_outputs_dir is None:
            stage1_outputs_dir = self.output_dir / "stage1" / dataset_name
        if stage2_outputs_dir is None:
            stage2_outputs_dir = self.output_dir / "stage2" / safe_analyzer / dataset_name

        if not stage2_outputs_dir.exists():
            if not self.quiet:
                print(f"Stage 2 output directory not found: {stage2_outputs_dir}")
            return []
        outputs = []
        tasks_to_process = []

        # Find all stage2 outputs
        all_stage2_files = list(stage2_outputs_dir.glob("*.json"))

        # Filter to only process files from this analyzer
        stage2_files = []
        for f in all_stage2_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get('analyzer_model') == self.model:
                    # Filter by problem IDs if specified
                    if problem_ids:
                        problem_id = data.get('problem_id', '')
                        if any(pid in problem_id for pid in problem_ids):
                            stage2_files.append((f, data))
                    else:
                        stage2_files.append((f, data))

        if not self.quiet:
            print(f"Found {len(all_stage2_files)} total Stage 2 outputs, {len(stage2_files)} from {self.model}")
            if problem_ids:
                print(f"Filtered to problems matching IDs: {problem_ids}")
            print("Stage 3: Preparing prompts...")

        for stage2_file, stage2_output in stage2_files:

            problem_id = stage2_output["problem_id"]
            target_model = stage2_output["target_model"]
            attempt_idx = stage2_output["attempt_index"]

            # Check if already processed
            safe_model = sanitize_filename(target_model)
            filename = f"{problem_id}_{safe_analyzer}_{safe_model}_attempt{attempt_idx}_stage3.json"

            if skip_existing:
                existing = self._load_output(filename, subdir=f"stage3/{safe_analyzer}/{dataset_name}")
                if existing:
                    if not self.quiet:
                        print(f"Skipping {problem_id}/{target_model}/attempt{attempt_idx} (already processed)")
                    outputs.append(Stage3Output.from_dict(existing))
                    continue

            # Load stage1 output
            stage1_path = stage1_outputs_dir / f"{problem_id}_{safe_analyzer}_stage1.json"
            if not stage1_path.exists():
                if not self.quiet:
                    print(f"Stage 1 output not found for {problem_id}, skipping")
                continue

            with open(stage1_path, 'r') as f:
                stage1_output = json.load(f)

            # Load problem
            problem_path = self.problems_dir / f"{problem_id}.json"
            if not problem_path.exists():
                if not self.quiet:
                    print(f"Problem file not found: {problem_path}, skipping")
                continue

            problem = self._load_problem(problem_path)

            # Format steps for prompt
            golden_steps_formatted = format_golden_steps_for_prompt(stage1_output.get("step_texts", []))

            cot_steps = stage2_output.get("step_texts", [])
            cot_steps_formatted = "\n\n".join([
                f"**CoT Step {step['step_number']}:**\n{step['verbatim_text']}"
                for step in cot_steps
            ])

            # Build prompt
            problem_statement = problem["problem_details"]["Problem Statement"]
            prompt = self.prompt_template.format(
                problem_statement=problem_statement,
                golden_steps_formatted=golden_steps_formatted,
                cot_steps_formatted=cot_steps_formatted
            )

            tasks_to_process.append({
                "problem_path": problem_path,
                "problem_id": problem_id,
                "problem": problem,
                "stage1_output": stage1_output,
                "stage2_output": stage2_output,
                "cot_steps": cot_steps,
                "filename": filename,
                "prompt": prompt
            })

        if not tasks_to_process:
            if not self.quiet:
                print("No steps to match.")
            return outputs

        # Batch call LLM
        prompts = [t["prompt"] for t in tasks_to_process]
        system_prompt = (
            "You are an expert theoretical physicist analyzing mathematical reasoning. "
            "Respond only with valid JSON matching the requested structure."
        )

        if not self.quiet:
            print(f"Stage 3: Processing {len(prompts)} rollouts...")

        with tqdm(total=len(prompts), desc="Stage 3: Matching steps", disable=self.quiet) as pbar:
            results = self._call_llm_for_json_batch(prompts, system_prompt, pbar=pbar)

        # Process results
        for task, response in zip(tasks_to_process, results):
            if response is None:
                if not self.quiet:
                    print(f"Error: No response for {task['problem_id']}")
                continue

            try:
                # Parse matched steps
                matched_steps = []
                for s in response.get("matched_steps", []):
                    cot_step_num = s.get("cot_step_number", len(matched_steps) + 1)
                    cot_step_text = ""
                    for cot_step in task["cot_steps"]:
                        if cot_step["step_number"] == cot_step_num:
                            cot_step_text = cot_step["verbatim_text"]
                            break

                    matched_steps.append(MatchedStep(
                        cot_step_number=cot_step_num,
                        cot_step_text=cot_step_text,
                        golden_step_mapping=s.get("golden_step_mapping"),
                        mapping_confidence=s.get("mapping_confidence", "none"),
                        mapping_explanation=s.get("mapping_explanation", ""),
                        validity=s.get("validity", "unknown"),
                        validity_explanation=s.get("validity_explanation", "")
                    ))

                # Compute summary
                valid_count = sum(1 for s in matched_steps if s.validity == "valid")
                invalid_count = sum(1 for s in matched_steps if s.validity == "invalid")
                mapped_count = sum(1 for s in matched_steps if s.golden_step_mapping is not None)

                total_golden_steps = len(task["stage1_output"].get("step_texts", []))
                coverage = compute_coverage(
                    [{"golden_step_mapping": s.golden_step_mapping} for s in matched_steps],
                    total_golden_steps
                )

                summary = Stage3Summary(
                    total_cot_steps=len(matched_steps),
                    mapped_steps=mapped_count,
                    unmapped_steps=len(matched_steps) - mapped_count,
                    valid_steps=valid_count,
                    invalid_steps=invalid_count,
                    coverage=coverage
                )

                # Build output
                output = Stage3Output(
                    problem_id=task["problem_id"],
                    problem_path=str(task["problem_path"]),
                    timestamp=get_timestamp(),
                    analyzer_model=self.model,
                    target_model=task["stage2_output"]["target_model"],
                    attempt_index=task["stage2_output"]["attempt_index"],
                    attempt_is_correct=task["stage2_output"].get("attempt_is_correct"),
                    cot_steps_reference=f"stage2/{task['problem_id']}_stage2.json",
                    golden_steps_reference=f"stage1/{task['problem_id']}_stage1.json",
                    matched_steps=matched_steps,
                    summary=summary
                )

                outputs.append(output)

                if save_outputs:
                    self._save_output(
                        output.to_dict(),
                        task["filename"],
                        subdir=f"stage3/{safe_analyzer}/{dataset_name}"
                    )

                if not self.quiet:
                    print(
                        f"Matched {task['problem_id']}: "
                        f"{output.summary.valid_steps}/{output.summary.total_cot_steps} valid, "
                        f"coverage={output.summary.coverage:.2f}"
                    )

            except Exception as e:
                if not self.quiet:
                    print(f"Error processing response for {task['problem_id']}: {e}")
                continue

        return outputs


def main():
    """CLI entry point for Stage 3."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Match distilled CoT steps to golden steps")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset subdirectory under data (for this portfolio: demo)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for matching")
    parser.add_argument("--stage1-dir", type=str, default=None,
                        help="Directory containing Stage 1 outputs")
    parser.add_argument("--stage2-dir", type=str, default=None,
                        help="Directory containing Stage 2 outputs")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="Comma-separated list of problem IDs to process (e.g., 'demo_constant_acceleration,demo_spring_frequency')")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip already processed rollouts")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    stage1_dir = None
    if args.stage1_dir:
        stage1_dir = Path(args.stage1_dir)

    stage2_dir = None
    if args.stage2_dir:
        stage2_dir = Path(args.stage2_dir)

    # Parse problem IDs
    problem_ids = None
    if args.problem_ids:
        problem_ids = [p.strip() for p in args.problem_ids.split(",")]

    matcher = StepMatcher(
        model_name=args.model,
        quiet=args.quiet
    )

    outputs = matcher.match_all_steps(
        dataset_name=args.dataset,
        stage1_outputs_dir=stage1_dir,
        stage2_outputs_dir=stage2_dir,
        skip_existing=not args.no_skip,
        problem_ids=problem_ids
    )

    print(f"\nStage 3 complete: Matched {len(outputs)} rollouts")

    # Print summary statistics
    if outputs:
        total_steps = sum(o.summary.total_cot_steps for o in outputs)
        total_valid = sum(o.summary.valid_steps for o in outputs)
        avg_coverage = sum(o.summary.coverage for o in outputs) / len(outputs)

        print(f"\nSummary:")
        print(f"  Total CoT steps: {total_steps}")
        if total_steps > 0:
            print(f"  Valid steps: {total_valid} ({100*total_valid/total_steps:.1f}%)")
        print(f"  Average coverage: {100*avg_coverage:.1f}%")


if __name__ == "__main__":
    main()
