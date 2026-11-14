# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Stage 2: Distill model CoT rollouts into logical steps.

Includes optional deduplication pre-processing to handle long rollouts:
  Raw CoT → remove code blocks → chunk by token limit → async dedup each chunk
  → concat deduplicated text → run distillation on shorter text
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import tiktoken

from rollout_analyzer_base import RolloutAnalyzerBase
from utils import add_line_numbers, extract_lines, extract_code_blocks, remove_code_blocks, sanitize_filename
from schemas import Stage2Output, CoTStepBoundary, CoTStepText, get_timestamp

# Shared tokenizer instance (fast, thread-safe)
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


class CoTDistiller(RolloutAnalyzerBase):
    """
    Stage 2: Distill model CoT rollouts into logical steps.
    """

    def __init__(self, model_name: str, dedup_chunk_tokens: int = 4000,
                 enable_dedup: bool = True, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.prompt_template = self._load_prompt_template("stage2_distill_prompt.txt")
        self.dedup_prompt_template = self._load_prompt_template("stage2_dedup_prompt.txt")
        self.dedup_chunk_tokens = dedup_chunk_tokens
        self.enable_dedup = enable_dedup

    # ------------------------------------------------------------------
    # Token counting & chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(_TIKTOKEN_ENC.encode(text))

    @staticmethod
    def _chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
        """
        Split text into chunks of at most `max_tokens` tokens.
        Splits on paragraph boundaries (\\n\\n) to avoid cutting mid-sentence.
        """
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = CoTDistiller._count_tokens(para)

            # If a single paragraph exceeds max_tokens, split it by lines
            if para_tokens > max_tokens:
                # Flush current accumulator first
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                lines = para.split("\n")
                line_parts: List[str] = []
                line_tokens = 0
                for line in lines:
                    lt = CoTDistiller._count_tokens(line)
                    if line_tokens + lt > max_tokens and line_parts:
                        chunks.append("\n".join(line_parts))
                        line_parts = []
                        line_tokens = 0
                    line_parts.append(line)
                    line_tokens += lt
                if line_parts:
                    chunks.append("\n".join(line_parts))
                continue

            if current_tokens + para_tokens > max_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedup_text(self, cot_math: str, problem_statement: str) -> Tuple[str, int, int, float]:
        """
        Deduplicate a CoT rollout by chunking and cleaning each chunk async.

        Returns:
            (deduped_text, original_tokens, result_tokens, reduction_pct)
        """
        original_tokens = self._count_tokens(cot_math)

        # Skip dedup if already short enough
        if original_tokens <= self.dedup_chunk_tokens:
            return cot_math, original_tokens, original_tokens, 0.0

        chunks = self._chunk_by_tokens(cot_math, self.dedup_chunk_tokens)

        if not self.quiet:
            print(f"  Dedup: {original_tokens} tokens -> {len(chunks)} chunks "
                  f"(max {self.dedup_chunk_tokens} tok/chunk)")

        # Build dedup prompts for each chunk
        dedup_prompts = []
        for chunk in chunks:
            prompt = self.dedup_prompt_template.format(
                problem_statement=problem_statement,
                cot_chunk=chunk
            )
            dedup_prompts.append(prompt)

        system_prompt = (
            "You are an expert theoretical physicist. "
            "Clean up the provided reasoning chunk as instructed. "
            "Return ONLY the cleaned text."
        )

        # Async batch call
        dedup_results = self._call_llm_for_text_batch_api(
            dedup_prompts, system_prompt
        )

        # Concat results (use original chunk as fallback on failure)
        cleaned_parts = []
        for i, result in enumerate(dedup_results):
            if result is not None and result.strip():
                cleaned_parts.append(result.strip())
            else:
                if not self.quiet:
                    print(f"  Dedup chunk {i} failed, keeping original")
                cleaned_parts.append(chunks[i])

        deduped_text = "\n\n".join(cleaned_parts)
        result_tokens = self._count_tokens(deduped_text)
        reduction_pct = (1.0 - result_tokens / original_tokens) * 100.0 if original_tokens > 0 else 0.0

        if not self.quiet:
            print(f"  Dedup result: {original_tokens} -> {result_tokens} tokens "
                  f"({reduction_pct:.1f}% reduction)")

        return deduped_text, original_tokens, result_tokens, reduction_pct

    # ------------------------------------------------------------------
    # Single rollout distillation
    # ------------------------------------------------------------------

    def distill_cot(
        self,
        problem_path: Path,
        target_model: str,
        attempt_index: int
    ) -> Optional[Stage2Output]:
        """
        Distill a single model CoT rollout into steps.

        Args:
            problem_path: Path to problem JSON
            target_model: Model name whose CoT to distill
            attempt_index: Which attempt (0-indexed) to distill

        Returns:
            Stage2Output or None if attempt not found
        """
        problem = self._load_problem(problem_path)
        problem_id = problem.get("problem_id", problem_path.stem)

        # Find the target model's attempt
        model_solution = None
        for ms in problem.get("model_solutions", []):
            if ms.get("model") == target_model:
                model_solution = ms
                break

        if not model_solution:
            if not self.quiet:
                print(f"Model {target_model} not found in {problem_id}")
            return None

        attempts = model_solution.get("attempts", [])
        if attempt_index >= len(attempts):
            if not self.quiet:
                print(f"Attempt {attempt_index} not found for {target_model} in {problem_id}")
            return None

        attempt = attempts[attempt_index]
        cot_rollout = attempt.get("detailed_solution", "")
        attempt_timestamp = attempt.get("timestamp", "")

        # Extract correctness from verifier result
        verifier_result = attempt.get("verifier_result", {})
        verified = verifier_result.get("verified")
        attempt_is_correct = bool(verified) if verified is not None else None

        if not cot_rollout:
            if not self.quiet:
                print(f"Empty CoT for {target_model} attempt {attempt_index} in {problem_id}")
            return None

        # Extract code blocks before removing them
        extracted_code = extract_code_blocks(cot_rollout)

        # Remove code blocks (focus on mathematical reasoning)
        cot_math = remove_code_blocks(cot_rollout)

        # Dedup if enabled
        dedup_original_tokens = None
        dedup_result_tokens = None
        dedup_reduction_pct = None

        if self.enable_dedup:
            problem_statement = problem["problem_details"]["Problem Statement"]
            cot_math, dedup_original_tokens, dedup_result_tokens, dedup_reduction_pct = \
                self._dedup_text(cot_math, problem_statement)

        # Add line numbers
        cot_with_lines, total_lines = add_line_numbers(cot_math)

        # Build prompt
        problem_statement = problem["problem_details"]["Problem Statement"]
        prompt = self.prompt_template.format(
            problem_statement=problem_statement,
            cot_rollout_with_lines=cot_with_lines
        )

        # Call LLM
        system_prompt = (
            "You are an expert theoretical physicist analyzing mathematical reasoning. "
            "Respond only with valid JSON matching the requested structure."
        )
        response = self._call_llm_for_json(prompt, system_prompt)

        # Parse boundaries
        boundaries = []
        for b in response.get("split_boundaries", []):
            boundaries.append(CoTStepBoundary(
                step_number=b["step_number"],
                start_line=b["start_line"],
                end_line=b["end_line"],
                step_description=b.get("step_description", "")
            ))

        # Extract verbatim text for each step
        step_texts = []
        for boundary in boundaries:
            verbatim = extract_lines(
                cot_math,
                boundary.start_line,
                boundary.end_line
            )
            step_texts.append(CoTStepText(
                step_number=boundary.step_number,
                verbatim_text=verbatim
            ))

        # Append code translation as final step if code was extracted
        if extracted_code:
            code_step_num = (boundaries[-1].step_number + 1) if boundaries else 1
            boundaries.append(CoTStepBoundary(
                step_number=code_step_num,
                start_line=-1,
                end_line=-1,
                step_description="Code translation: converting derivation to implementation",
            ))
            step_texts.append(CoTStepText(
                step_number=code_step_num,
                verbatim_text=extracted_code,
            ))

        # Build output
        output = Stage2Output(
            problem_id=problem_id,
            problem_path=str(problem_path),
            timestamp=get_timestamp(),
            analyzer_model=self.model,
            target_model=target_model,
            attempt_index=attempt_index,
            attempt_timestamp=attempt_timestamp,
            attempt_is_correct=attempt_is_correct,
            cot_rollout_raw=cot_rollout,
            cot_rollout_with_lines=cot_with_lines,
            total_lines=total_lines,
            split_boundaries=boundaries,
            step_texts=step_texts,
            dedup_original_tokens=dedup_original_tokens,
            dedup_result_tokens=dedup_result_tokens,
            dedup_reduction_pct=dedup_reduction_pct,
        )

        return output

    # ------------------------------------------------------------------
    # Batch distillation
    # ------------------------------------------------------------------

    def distill_all_cots(
        self,
        dataset_name: str = "demo",
        target_models: Optional[List[str]] = None,
        max_attempts_per_model: int = 1,
        save_outputs: bool = True,
        skip_existing: bool = True,
        problem_ids: Optional[List[str]] = None
    ) -> List[Stage2Output]:
        """
        Distill all CoT rollouts for all problems in a dataset using batch processing.

        Args:
            dataset_name: Dataset subdirectory name
            target_models: List of models to distill (None = all found)
            max_attempts_per_model: Max attempts per model to distill
            save_outputs: Whether to save individual JSON files
            skip_existing: Skip already processed rollouts
            problem_ids: List of specific problem IDs to process (None = all problems)

        Returns:
            List of Stage2Output objects
        """
        self.problems_dir = self.project_root / "data" / dataset_name
        all_problem_files = self.get_problem_files()

        # Filter by problem IDs if specified
        if problem_ids:
            problem_files = []
            for pf in all_problem_files:
                problem_id = Path(pf).stem
                if any(pid in problem_id for pid in problem_ids):
                    problem_files.append(pf)
            if not self.quiet:
                print(f"Filtered to {len(problem_files)} problems matching IDs: {problem_ids}")
        else:
            problem_files = all_problem_files

        safe_analyzer = sanitize_filename(self.model_name)
        outputs = []
        tasks_to_process = []

        # First pass: collect tasks and check for existing outputs
        if not self.quiet:
            print("Stage 2: Preparing prompts...")

        for problem_file in problem_files:
            problem_path = self.problems_dir / problem_file
            problem = self._load_problem(problem_path)
            problem_id = problem.get("problem_id", problem_path.stem)

            # Get models to analyze
            available_models = [
                ms["model"] for ms in problem.get("model_solutions", [])
            ]

            if target_models:
                models_to_analyze = [m for m in target_models if m in available_models]
            else:
                models_to_analyze = available_models

            # Collect tasks for each model's attempts
            for target_model in models_to_analyze:
                model_solution = next(
                    (ms for ms in problem["model_solutions"] if ms["model"] == target_model),
                    None
                )

                if not model_solution:
                    continue

                num_attempts = min(
                    len(model_solution.get("attempts", [])),
                    max_attempts_per_model
                )

                for attempt_idx in range(num_attempts):
                    # Check if already processed
                    safe_model = sanitize_filename(target_model)
                    filename = f"{problem_id}_{safe_analyzer}_{safe_model}_attempt{attempt_idx}_stage2.json"

                    if skip_existing:
                        existing = self._load_output(filename, subdir=f"stage2/{safe_analyzer}/{dataset_name}")
                        if existing:
                            if not self.quiet:
                                print(f"Skipping {problem_id}/{target_model}/attempt{attempt_idx} (already processed)")
                            outputs.append(Stage2Output.from_dict(existing))
                            continue

                    # Get attempt data
                    attempts = model_solution.get("attempts", [])
                    if attempt_idx >= len(attempts):
                        continue

                    attempt = attempts[attempt_idx]
                    cot_rollout = attempt.get("detailed_solution", "")
                    attempt_timestamp = attempt.get("timestamp", "")

                    # Extract correctness from verifier result
                    verifier_result = attempt.get("verifier_result", {})
                    verified = verifier_result.get("verified")
                    attempt_is_correct = bool(verified) if verified is not None else None

                    if not cot_rollout:
                        if not self.quiet:
                            print(f"Empty CoT for {target_model} attempt {attempt_idx} in {problem_id}")
                        continue

                    # Extract code blocks before removing them
                    extracted_code = extract_code_blocks(cot_rollout)

                    # Remove code blocks
                    cot_math = remove_code_blocks(cot_rollout)

                    tasks_to_process.append({
                        "problem_path": problem_path,
                        "problem_id": problem_id,
                        "problem": problem,
                        "target_model": target_model,
                        "attempt_idx": attempt_idx,
                        "attempt_timestamp": attempt_timestamp,
                        "attempt_is_correct": attempt_is_correct,
                        "cot_rollout": cot_rollout,
                        "cot_math": cot_math,
                        "extracted_code": extracted_code,
                        "filename": filename,
                    })

        if not tasks_to_process:
            if not self.quiet:
                print("No CoT rollouts to process.")
            return outputs

        # ------------------------------------------------------------------
        # Group tasks by (problem_id, target_model) so we can process and
        # save incrementally — completed work is saved immediately.
        # ------------------------------------------------------------------
        from collections import OrderedDict
        task_groups: OrderedDict[str, List[dict]] = OrderedDict()
        for task in tasks_to_process:
            key = f"{task['problem_id']}/{task['target_model']}"
            task_groups.setdefault(key, []).append(task)

        total_tasks = len(tasks_to_process)
        completed_tasks = 0

        if not self.quiet:
            print(f"Stage 2: Processing {total_tasks} rollouts in {len(task_groups)} groups")

        dedup_system_prompt = (
            "You are an expert theoretical physicist. "
            "Clean up the provided reasoning chunk as instructed. "
            "Return ONLY the cleaned text."
        )
        distill_system_prompt = (
            "You are an expert theoretical physicist analyzing mathematical reasoning. "
            "Respond only with valid JSON matching the requested structure."
        )

        # ==================================================================
        # Phase 1: Dedup ALL groups in one mega-batch
        # ==================================================================
        # Build a flat list of all tasks with a global index for tracking
        all_tasks_flat = []
        task_global_idx = {}  # (group_key, local_idx) -> global_idx
        for group_key, group_tasks in task_groups.items():
            for local_idx, task in enumerate(group_tasks):
                gidx = len(all_tasks_flat)
                task_global_idx[(group_key, local_idx)] = gidx
                all_tasks_flat.append(task)

        dedup_meta = {}  # global_idx -> (orig, result, pct)

        if self.enable_dedup:
            dedup_prompts = []
            dedup_job_map = []  # (global_idx, chunk_idx)
            task_chunk_info = {}  # global_idx -> (n_chunks, chunks_list, original_tokens)

            for gidx, task in enumerate(all_tasks_flat):
                original_tokens = self._count_tokens(task["cot_math"])

                if original_tokens <= self.dedup_chunk_tokens:
                    task_chunk_info[gidx] = (0, [], original_tokens)
                    dedup_meta[gidx] = (original_tokens, original_tokens, 0.0)
                    continue

                chunks = self._chunk_by_tokens(task["cot_math"], self.dedup_chunk_tokens)
                task_chunk_info[gidx] = (len(chunks), chunks, original_tokens)

                problem_statement = task["problem"]["problem_details"]["Problem Statement"]
                for chunk_idx, chunk in enumerate(chunks):
                    prompt = self.dedup_prompt_template.format(
                        problem_statement=problem_statement,
                        cot_chunk=chunk
                    )
                    dedup_prompts.append(prompt)
                    dedup_job_map.append((gidx, chunk_idx))

            if dedup_prompts:
                if not self.quiet:
                    print(f"  Dedup: {len(dedup_prompts)} chunks across all groups")

                with tqdm(total=len(dedup_prompts),
                          desc=f"  Dedup all",
                          disable=self.quiet) as pbar:
                    dedup_results = self._call_llm_for_text_batch_api(
                        dedup_prompts, dedup_system_prompt, pbar=pbar
                    )

                # Reassemble per task
                task_chunk_results: dict = {}  # gidx -> {chunk_idx: result}
                for i, (gidx, chunk_idx) in enumerate(dedup_job_map):
                    task_chunk_results.setdefault(gidx, {})[chunk_idx] = dedup_results[i]

                for gidx, task in enumerate(all_tasks_flat):
                    info = task_chunk_info.get(gidx)
                    if info is None:
                        continue
                    n_chunks, chunks, original_tokens = info
                    if n_chunks == 0:
                        continue

                    chunk_res = task_chunk_results.get(gidx, {})
                    cleaned_parts = []
                    for chunk_idx in range(n_chunks):
                        result = chunk_res.get(chunk_idx)
                        if result is not None and result.strip():
                            cleaned_parts.append(result.strip())
                        else:
                            if not self.quiet:
                                print(f"  Dedup failed for {task['problem_id']} attempt {task['attempt_idx']} chunk {chunk_idx}, keeping original")
                            cleaned_parts.append(chunks[chunk_idx])

                    deduped_text = "\n\n".join(cleaned_parts)
                    task["cot_math"] = deduped_text

                    result_tokens = self._count_tokens(deduped_text)
                    reduction_pct = (1.0 - result_tokens / original_tokens) * 100.0
                    dedup_meta[gidx] = (original_tokens, result_tokens, reduction_pct)

                    if not self.quiet:
                        print(f"  {task['problem_id']} attempt {task['attempt_idx']}: "
                              f"{original_tokens} -> {result_tokens} tokens "
                              f"({reduction_pct:.1f}% reduction)")

        # ==================================================================
        # Phase 2: Distill ALL groups in one mega-batch
        # ==================================================================
        for task in all_tasks_flat:
            cot_with_lines, total_lines = add_line_numbers(task["cot_math"])
            task["cot_with_lines"] = cot_with_lines
            task["total_lines"] = total_lines

            problem_statement = task["problem"]["problem_details"]["Problem Statement"]
            task["prompt"] = self.prompt_template.format(
                problem_statement=problem_statement,
                cot_rollout_with_lines=cot_with_lines
            )

        all_prompts = [t["prompt"] for t in all_tasks_flat]

        if not self.quiet:
            print(f"\n  Distilling {len(all_prompts)} rollouts in one batch...")

        with tqdm(total=len(all_prompts),
                  desc=f"  Distill all",
                  disable=self.quiet) as pbar:
            distill_results = self._call_llm_for_json_batch(
                all_prompts, distill_system_prompt, pbar=pbar
            )

        # ==================================================================
        # Phase 3: Parse and save all results
        # ==================================================================
        for gidx, (task, response) in enumerate(zip(all_tasks_flat, distill_results)):
            if response is None:
                if not self.quiet:
                    print(f"  Error: No response for {task['problem_id']} attempt {task['attempt_idx']}")
                continue

            try:
                boundaries = []
                for b in response.get("split_boundaries", []):
                    boundaries.append(CoTStepBoundary(
                        step_number=b["step_number"],
                        start_line=b["start_line"],
                        end_line=b["end_line"],
                        step_description=b.get("step_description", "")
                    ))

                step_texts = []
                for boundary in boundaries:
                    verbatim = extract_lines(
                        task["cot_math"],
                        boundary.start_line,
                        boundary.end_line
                    )
                    step_texts.append(CoTStepText(
                        step_number=boundary.step_number,
                        verbatim_text=verbatim
                    ))

                extracted_code = task.get("extracted_code", "")
                if extracted_code:
                    code_step_num = (boundaries[-1].step_number + 1) if boundaries else 1
                    boundaries.append(CoTStepBoundary(
                        step_number=code_step_num,
                        start_line=-1,
                        end_line=-1,
                        step_description="Code translation: converting derivation to implementation",
                    ))
                    step_texts.append(CoTStepText(
                        step_number=code_step_num,
                        verbatim_text=extracted_code,
                    ))

                dm = dedup_meta.get(gidx)
                d_orig = dm[0] if dm else None
                d_result = dm[1] if dm else None
                d_pct = dm[2] if dm else None

                output = Stage2Output(
                    problem_id=task["problem_id"],
                    problem_path=str(task["problem_path"]),
                    timestamp=get_timestamp(),
                    analyzer_model=self.model,
                    target_model=task["target_model"],
                    attempt_index=task["attempt_idx"],
                    attempt_timestamp=task["attempt_timestamp"],
                    attempt_is_correct=task["attempt_is_correct"],
                    cot_rollout_raw=task["cot_rollout"],
                    cot_rollout_with_lines=task["cot_with_lines"],
                    total_lines=task["total_lines"],
                    split_boundaries=boundaries,
                    step_texts=step_texts,
                    dedup_original_tokens=d_orig,
                    dedup_result_tokens=d_result,
                    dedup_reduction_pct=d_pct,
                )

                outputs.append(output)

                if save_outputs:
                    self._save_output(
                        output.to_dict(),
                        task["filename"],
                        subdir=f"stage2/{safe_analyzer}/{dataset_name}"
                    )

                completed_tasks += 1
                if not self.quiet and completed_tasks % 100 == 0:
                    print(f"  [{completed_tasks}/{total_tasks}] saved")

            except Exception as e:
                if not self.quiet:
                    print(f"  Error processing {task['problem_id']} attempt {task['attempt_idx']}: {e}")
                continue

        return outputs


def main():
    """CLI entry point for Stage 2."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Distill model CoT rollouts")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset subdirectory under data (for this portfolio: demo)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for distillation")
    parser.add_argument("--target-models", type=str, default=None,
                        help="Comma-separated list of models to distill (default: all)")
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="Max attempts per model to distill")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="Comma-separated list of problem IDs to process (e.g., 'demo_constant_acceleration,demo_spring_frequency')")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip already processed rollouts")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--dedup-chunk-tokens", type=int, default=4000,
                        help="Max tokens per dedup chunk (default: 4000)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication pre-processing")

    args = parser.parse_args()

    # Parse target models
    target_models = None
    if args.target_models:
        target_models = [m.strip() for m in args.target_models.split(",")]

    # Parse problem IDs
    problem_ids = None
    if args.problem_ids:
        problem_ids = [p.strip() for p in args.problem_ids.split(",")]

    distiller = CoTDistiller(
        model_name=args.model,
        dedup_chunk_tokens=args.dedup_chunk_tokens,
        enable_dedup=not args.no_dedup,
        quiet=args.quiet
    )

    outputs = distiller.distill_all_cots(
        dataset_name=args.dataset,
        target_models=target_models,
        max_attempts_per_model=args.max_attempts,
        skip_existing=not args.no_skip,
        problem_ids=problem_ids
    )

    print(f"\nStage 2 complete: Distilled {len(outputs)} CoT rollouts")

    # Print summary statistics
    if outputs:
        total_steps = sum(len(o.split_boundaries) for o in outputs)
        avg_steps = total_steps / len(outputs) if outputs else 0

        # Dedup stats
        dedup_outputs = [o for o in outputs if o.dedup_reduction_pct is not None and o.dedup_reduction_pct > 0]
        if dedup_outputs:
            avg_reduction = sum(o.dedup_reduction_pct for o in dedup_outputs) / len(dedup_outputs)
            print(f"\nDedup summary:")
            print(f"  Rollouts deduped: {len(dedup_outputs)}/{len(outputs)}")
            print(f"  Average reduction: {avg_reduction:.1f}%")

        print(f"\nDistillation summary:")
        print(f"  Total distilled steps: {total_steps}")
        print(f"  Average steps per rollout: {avg_steps:.1f}")


if __name__ == "__main__":
    main()
