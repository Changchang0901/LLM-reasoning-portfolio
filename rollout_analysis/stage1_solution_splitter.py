# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Stage 1: Split golden solutions into logical steps.
"""

from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from rollout_analyzer_base import RolloutAnalyzerBase
from utils import add_line_numbers, extract_lines, remove_code_blocks, sanitize_filename
from schemas import Stage1Output, StepBoundary, StepText, get_timestamp


class SolutionSplitter(RolloutAnalyzerBase):
    """
    Stage 1: Split golden solutions into logical steps.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.prompt_template = self._load_prompt_template("stage1_split_prompt.txt")

    @staticmethod
    def _append_code_step(
        boundaries: List[StepBoundary],
        step_texts: List[StepText],
        answer_requirements: str,
        code: str,
    ) -> tuple:
        """Append a code-translation step as the final step if code content exists."""
        if not answer_requirements and not code:
            return boundaries, step_texts
        last_num = boundaries[-1].step_number if boundaries else 0
        code_step_num = last_num + 1
        verbatim_parts = []
        if answer_requirements:
            verbatim_parts.append(f"## Answer Requirements\n{answer_requirements}")
        if code:
            verbatim_parts.append(f"## Code\n{code}")
        boundaries.append(StepBoundary(
            step_number=code_step_num,
            start_line=-1,
            end_line=-1,
            step_description="Code translation: answer requirements and implementation",
        ))
        step_texts.append(StepText(
            step_number=code_step_num,
            verbatim_text="\n\n".join(verbatim_parts),
        ))
        return boundaries, step_texts

    def split_solution(self, problem_path: Path) -> Stage1Output:
        """
        Split a single problem's golden solution into steps.

        Args:
            problem_path: Path to problem JSON file

        Returns:
            Stage1Output with split boundaries and step texts
        """
        problem = self._load_problem(problem_path)
        problem_id = problem.get("problem_id", problem_path.stem)

        # Extract golden solution (mathematical part only)
        golden_solution_raw = problem["problem_details"]["Solution"]
        golden_solution_math = remove_code_blocks(golden_solution_raw)

        # Add line numbers
        golden_with_lines, total_lines = add_line_numbers(golden_solution_math)

        # Build prompt
        problem_statement = problem["problem_details"]["Problem Statement"]
        prompt = self.prompt_template.format(
            problem_statement=problem_statement,
            golden_solution_with_lines=golden_with_lines
        )

        # Call LLM
        system_prompt = (
            "You are an expert theoretical physicist. "
            "Respond only with valid JSON matching the requested structure."
        )
        response = self._call_llm_for_json(prompt, system_prompt)

        # Parse boundaries
        boundaries = []
        for b in response.get("split_boundaries", []):
            boundaries.append(StepBoundary(
                step_number=b["step_number"],
                start_line=b["start_line"],
                end_line=b["end_line"],
                step_description=b.get("step_description", "")
            ))

        # Extract verbatim text for each step
        step_texts = []
        for boundary in boundaries:
            verbatim = extract_lines(
                golden_solution_math,
                boundary.start_line,
                boundary.end_line
            )
            step_texts.append(StepText(
                step_number=boundary.step_number,
                verbatim_text=verbatim
            ))

        # Append code translation as final step
        answer_requirements = problem["problem_details"].get("Answer Requirements", "")
        code = problem["problem_details"].get("Code", "")
        boundaries, step_texts = self._append_code_step(
            boundaries, step_texts, answer_requirements, code
        )

        # Build output
        output = Stage1Output(
            problem_id=problem_id,
            problem_path=str(problem_path),
            timestamp=get_timestamp(),
            analyzer_model=self.model,
            golden_solution_raw=golden_solution_raw,
            golden_solution_with_lines=golden_with_lines,
            total_lines=total_lines,
            split_boundaries=boundaries,
            step_texts=step_texts,
            answer_requirements=answer_requirements,
            code=code,
        )

        return output

    def split_all_solutions(
        self,
        dataset_name: str = "demo",
        save_outputs: bool = True,
        skip_existing: bool = True
    ) -> List[Stage1Output]:
        """
        Process all problems in a dataset using batch processing for efficiency.

        Args:
            dataset_name: Name of dataset subdirectory
            save_outputs: Whether to save individual JSON files
            skip_existing: Skip problems that already have Stage 1 output

        Returns:
            List of Stage1Output objects
        """
        self.problems_dir = self.project_root / "data" / dataset_name
        problem_files = self.get_problem_files()
        safe_analyzer = sanitize_filename(self.model_name)

        outputs = []
        tasks_to_process = []  # List of dicts with problem metadata and prompts

        # First pass: collect prompts and check for existing outputs
        if not self.quiet:
            print("Stage 1: Preparing prompts...")
        
        for problem_file in problem_files:
            problem_path = self.problems_dir / problem_file
            problem_id = problem_path.stem

            # Check if already processed
            if skip_existing:
                existing = self._load_output(f"{problem_id}_{safe_analyzer}_stage1.json", subdir=f"stage1/{dataset_name}")
                if existing:
                    if not self.quiet:
                        print(f"Skipping {problem_id} (already processed)")
                    outputs.append(Stage1Output.from_dict(existing))
                    continue

            try:
                problem = self._load_problem(problem_path)
                
                # Extract golden solution (mathematical part only)
                golden_solution_raw = problem["problem_details"]["Solution"]
                golden_solution_math = remove_code_blocks(golden_solution_raw)
                golden_with_lines, total_lines = add_line_numbers(golden_solution_math)

                # Build prompt
                problem_statement = problem["problem_details"]["Problem Statement"]
                prompt = self.prompt_template.format(
                    problem_statement=problem_statement,
                    golden_solution_with_lines=golden_with_lines
                )

                tasks_to_process.append({
                    "problem_path": problem_path,
                    "problem_id": problem_id,
                    "problem": problem,
                    "golden_solution_raw": golden_solution_raw,
                    "golden_solution_math": golden_solution_math,
                    "golden_with_lines": golden_with_lines,
                    "total_lines": total_lines,
                    "prompt": prompt
                })
            except Exception as e:
                if not self.quiet:
                    print(f"Error preparing {problem_path}: {e}")
                continue

        if not tasks_to_process:
            if not self.quiet:
                print("No problems to process.")
            return outputs

        # Batch call LLM
        prompts = [t["prompt"] for t in tasks_to_process]
        system_prompt = (
            "You are an expert theoretical physicist. "
            "Respond only with valid JSON matching the requested structure."
        )

        if not self.quiet:
            print(f"Stage 1: Processing {len(prompts)} problems...")

        with tqdm(total=len(prompts), desc="Stage 1: Splitting solutions", disable=self.quiet) as pbar:
            results = self._call_llm_for_json_batch(prompts, system_prompt, pbar=pbar)

        # Process results
        for task, response in zip(tasks_to_process, results):
            if response is None:
                if not self.quiet:
                    print(f"Error: No response for {task['problem_id']}")
                continue

            try:
                # Parse boundaries
                boundaries = []
                for b in response.get("split_boundaries", []):
                    boundaries.append(StepBoundary(
                        step_number=b["step_number"],
                        start_line=b["start_line"],
                        end_line=b["end_line"],
                        step_description=b.get("step_description", "")
                    ))

                # Extract verbatim text for each step
                step_texts = []
                for boundary in boundaries:
                    verbatim = extract_lines(
                        task["golden_solution_math"],
                        boundary.start_line,
                        boundary.end_line
                    )
                    step_texts.append(StepText(
                        step_number=boundary.step_number,
                        verbatim_text=verbatim
                    ))

                # Append code translation as final step
                answer_requirements = task["problem"]["problem_details"].get("Answer Requirements", "")
                code = task["problem"]["problem_details"].get("Code", "")
                boundaries, step_texts = self._append_code_step(
                    boundaries, step_texts, answer_requirements, code
                )

                # Build output
                output = Stage1Output(
                    problem_id=task["problem_id"],
                    problem_path=str(task["problem_path"]),
                    timestamp=get_timestamp(),
                    analyzer_model=self.model,
                    golden_solution_raw=task["golden_solution_raw"],
                    golden_solution_with_lines=task["golden_with_lines"],
                    total_lines=task["total_lines"],
                    split_boundaries=boundaries,
                    step_texts=step_texts,
                    answer_requirements=answer_requirements,
                    code=code,
                )

                outputs.append(output)

                if save_outputs:
                    filename = f"{output.problem_id}_{safe_analyzer}_stage1.json"
                    self._save_output(
                        output.to_dict(),
                        filename,
                        subdir=f"stage1/{dataset_name}"
                    )

                if not self.quiet:
                    print(f"Processed {output.problem_id}: {len(output.split_boundaries)} steps")

            except Exception as e:
                if not self.quiet:
                    print(f"Error processing response for {task['problem_id']}: {e}")
                continue

        return outputs


def main():
    """CLI entry point for Stage 1."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Split golden solutions into steps")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset subdirectory under data (for this portfolio: demo)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for analysis")
    parser.add_argument("--problem", type=str, default=None,
                        help="Single problem ID to process (optional)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip already processed problems")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    splitter = SolutionSplitter(
        model_name=args.model,
        quiet=args.quiet
    )

    if args.problem:
        # Process single problem
        problem_path = splitter.project_root / "data" / args.dataset / f"{args.problem}.json"
        if not problem_path.exists():
            # Try searching for it
            splitter.problems_dir = splitter.project_root / "data" / args.dataset
            for pf in splitter.get_problem_files():
                if args.problem in str(pf):
                    problem_path = splitter.problems_dir / pf
                    break

        output = splitter.split_solution(problem_path)
        safe_analyzer = sanitize_filename(args.model)
        filename = f"{output.problem_id}_{safe_analyzer}_stage1.json"
        splitter._save_output(output.to_dict(), filename, subdir=f"stage1/{args.dataset}")
        print(f"Processed {output.problem_id}: {len(output.split_boundaries)} steps")
    else:
        # Process all
        outputs = splitter.split_all_solutions(
            dataset_name=args.dataset,
            skip_existing=not args.no_skip
        )
        print(f"\nStage 1 complete: Processed {len(outputs)} problems")


if __name__ == "__main__":
    main()
