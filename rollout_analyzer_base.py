# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Base class for rollout analysis, extending BaseProblemProcessor.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'main'))
from problem_processor_base import BaseProblemProcessor, load_problem_with_lock


class RolloutAnalyzerBase(BaseProblemProcessor):
    """
    Base class for rollout analysis stages.
    Extends BaseProblemProcessor to reuse LLM API infrastructure.
    
    Supports batched processing for both API-based and vLLM models.
    """

    def __init__(
        self,
        model_name: str,
        config_path: str = "config/inference_config.json",
        problems_dir: str = "data/demo",
        output_dir: str = "rollout_analysis/outputs",
        multi_gpu: int = 1,
        quiet: bool = False,
        temperature: Optional[float] = 0.0,  # Low temp for deterministic analysis
        requests_per_minute: int = 30,
        max_concurrent_requests: int = 2,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            config_path=config_path,
            problems_dir=problems_dir,
            multi_gpu=multi_gpu,
            quiet=quiet,
            temperature=temperature,
            **kwargs
        )
        self.output_dir = Path(__file__).parent / "outputs"
        if output_dir != "rollout_analysis/outputs":
            self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir = Path(__file__).parent / "prompts"
        
        # Batch processing parameters
        self.requests_per_minute = requests_per_minute
        self.max_concurrent_requests = max_concurrent_requests

        # Retry parameters
        self.max_retries = 5
        self.retry_base_delay = 30.0   # seconds (start at 30s)
        self.retry_max_delay = 120.0   # seconds cap
        self.batch_max_retry_rounds = 3  # how many full rounds to retry failed items

        # Auto-configure rate limits for specific providers
        if self.model_name.startswith("gemini"):
            if requests_per_minute > 20:
                self.requests_per_minute = 20
                if not self.quiet:
                    print(f"Gemini model detected. Auto-setting requests_per_minute to {self.requests_per_minute}")

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        template_path = self.prompts_dir / template_name
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _call_llm_for_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call LLM and return raw text response (no JSON parsing).
        Handles both API and vLLM backends.
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        if self._should_use_claude_api():
            return self._generate_claude_response(
                user_content=prompt, system_prompt=system_prompt, max_tokens=8192
            )
        elif self._should_use_openai_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            return self._generate_openai_response(messages, max_tokens=8192)
        elif self._should_use_grok_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            return self._generate_grok_response(messages)
        elif self.model_name.startswith("gemini"):
            full_prompt = f"{system_prompt}\n\n{prompt}"
            return self._generate_gemini_response(full_prompt)
        elif self.local_llm is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            prompt_text = self.local_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=8192,
                temperature=self.temperature if self.temperature is not None else 0.0
            )
            output = self.local_llm.generate([prompt_text], sampling_params)
            return output[0].outputs[0].text
        else:
            raise RuntimeError(f"No valid LLM backend configured for model: {self.model_name}")

    def _call_llm_for_text_single(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Single LLM call for text response with exponential backoff retry."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self._call_llm_for_text(prompt, system_prompt)
            except Exception as e:
                last_err = e
                err_str = str(e)
                retryable = any(k in err_str for k in ("502", "503", "429", "overloaded", "rate", "timeout", "timed out", "Bad gateway", "bad gateway", "Could not extract valid JSON"))
                if not retryable:
                    if not self.quiet:
                        print(f"Error in LLM text call (non-retryable): {e}")
                    return None
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.retry_max_delay,
                )
                if not self.quiet:
                    print(f"Retryable error (attempt {attempt + 1}/{self.max_retries}), "
                          f"waiting {delay:.1f}s: {e}")
                time.sleep(delay)
        if not self.quiet:
            print(f"LLM text call failed after {self.max_retries} retries: {last_err}")
        return None

    def _call_llm_for_text_batch_api(
        self,
        prompts: List[str],
        system_prompt: str,
        pbar: Optional[Any] = None
    ) -> List[Optional[str]]:
        """
        Batch LLM calls returning raw text (no JSON parsing).
        Includes automatic retry rounds for failed items.
        """
        if not prompts:
            return []

        async def async_worker(index, prompt, semaphore, limiter_delay, executor, loop):
            async with semaphore:
                await asyncio.sleep(limiter_delay)
                try:
                    result = await loop.run_in_executor(
                        executor, self._call_llm_for_text_single, prompt, system_prompt
                    )
                except Exception as e:
                    if not self.quiet:
                        print(f"Error in text batch call {index}: {e}")
                    result = None
                if pbar is not None:
                    pbar.update(1)
                return index, result

        async def run_batch(indices_and_prompts):
            sem = asyncio.Semaphore(self.max_concurrent_requests)
            delay = 60.0 / max(1, self.requests_per_minute)
            with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                loop = asyncio.get_running_loop()
                tasks = []
                for idx, prompt in indices_and_prompts:
                    task = asyncio.create_task(async_worker(idx, prompt, sem, delay, executor, loop))
                    tasks.append(task)
                    await asyncio.sleep(delay)
                return await asyncio.gather(*tasks)

        def _run_async(indices_and_prompts):
            try:
                return asyncio.run(run_batch(indices_and_prompts))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(run_batch(indices_and_prompts))

        if not self.quiet:
            print(f"Starting async text batch for {len(prompts)} prompts...")

        # Initial run
        all_pairs = list(enumerate(prompts))
        raw_results = _run_async(all_pairs)

        results: List[Optional[str]] = [None] * len(prompts)
        for idx, result in raw_results:
            results[idx] = result

        # Retry rounds for failed items
        for retry_round in range(self.batch_max_retry_rounds):
            failed = [(i, prompts[i]) for i in range(len(prompts)) if results[i] is None]
            if not failed:
                break
            if not self.quiet:
                print(f"  Retry round {retry_round + 1}/{self.batch_max_retry_rounds}: "
                      f"{len(failed)} failed items")
            # Cooldown before retry round
            cooldown = min(30.0 * (retry_round + 1), 90.0)
            if not self.quiet:
                print(f"  Waiting {cooldown:.0f}s before retry round...")
            time.sleep(cooldown)
            retry_results = _run_async(failed)
            for idx, result in retry_results:
                if result is not None:
                    results[idx] = result

        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0 and not self.quiet:
            print(f"  {n_failed}/{len(prompts)} items still failed after all retries")

        return results

    def _call_llm_for_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Call LLM and parse JSON response.
        Handles both API and vLLM backends.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON dictionary from the response
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that responds only with valid JSON."

        # Use existing infrastructure from BaseProblemProcessor
        if self._should_use_claude_api():
            response = self._generate_claude_response(
                user_content=prompt,
                system_prompt=system_prompt,
                max_tokens=8192
            )
        elif self._should_use_openai_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self._generate_openai_response(messages, max_tokens=8192)
        elif self._should_use_grok_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self._generate_grok_response(messages)
        elif self.model_name.startswith("gemini"):
            # Gemini doesn't have a separate system prompt in the same way
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self._generate_gemini_response(full_prompt)
        elif self.local_llm is not None:
            # vLLM local model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            prompt_text = self.local_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=8192,
                temperature=self.temperature if self.temperature is not None else 0.0
            )
            output = self.local_llm.generate([prompt_text], sampling_params)
            response = output[0].outputs[0].text
        else:
            raise RuntimeError(f"No valid LLM backend configured for model: {self.model_name}")

        # Parse JSON from response
        return self._extract_json_from_response(response)

    def _extract_json_from_response(self, response: str) -> Dict:
        """
        Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dictionary
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object directly (greedy match for outermost braces)
        # Use a more careful approach to find balanced braces
        brace_start = response.find('{')
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(response[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[brace_start:i+1])
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"Could not extract valid JSON from response: {response[:500]}...")

    def _save_output(self, output: Dict, filename: str, subdir: str = "") -> Path:
        """
        Save analysis output to JSON file.

        Args:
            output: Dictionary to save
            filename: Output filename
            subdir: Optional subdirectory within output_dir

        Returns:
            Path to saved file
        """
        if subdir:
            output_path = self.output_dir / subdir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output_path

    def _load_output(self, filename: str, subdir: str = "") -> Optional[Dict]:
        """
        Load an existing output file.

        Args:
            filename: Output filename
            subdir: Optional subdirectory within output_dir

        Returns:
            Loaded dictionary or None if not found
        """
        if subdir:
            output_path = self.output_dir / subdir / filename
        else:
            output_path = self.output_dir / filename

        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _call_llm_for_json_single(self, prompt: str, system_prompt: str) -> Optional[Dict]:
        """Single LLM call for JSON response with exponential backoff retry."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self._call_llm_for_json(prompt, system_prompt)
            except Exception as e:
                last_err = e
                err_str = str(e)
                retryable = any(k in err_str for k in ("502", "503", "429", "overloaded", "rate", "timeout", "timed out", "Bad gateway", "bad gateway", "Could not extract valid JSON"))
                if not retryable:
                    if not self.quiet:
                        print(f"Error in LLM JSON call (non-retryable): {e}")
                    return None
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.retry_max_delay,
                )
                if not self.quiet:
                    print(f"Retryable error (attempt {attempt + 1}/{self.max_retries}), "
                          f"waiting {delay:.1f}s: {e}")
                time.sleep(delay)
        if not self.quiet:
            print(f"LLM JSON call failed after {self.max_retries} retries: {last_err}")
        return None

    def _call_llm_for_json_batch_api(
        self,
        prompts: List[str],
        system_prompt: str,
        pbar: Optional[Any] = None,
        on_progress: Optional[Callable[[int, Optional[Dict]], None]] = None
    ) -> List[Optional[Dict]]:
        """
        Batch LLM calls for API-based models using async processing.
        Includes automatic retry rounds for failed items.
        """
        if not prompts:
            return []

        async def async_worker(index, prompt, semaphore, limiter_delay, executor, loop):
            async with semaphore:
                await asyncio.sleep(limiter_delay)

                try:
                    result = await loop.run_in_executor(
                        executor,
                        self._call_llm_for_json_single,
                        prompt,
                        system_prompt
                    )
                except Exception as e:
                    if not self.quiet:
                        print(f"Error in batch call {index}: {e}")
                    result = None

                if pbar is not None:
                    pbar.update(1)

                if on_progress:
                    try:
                        on_progress(index, result)
                    except Exception as e:
                        if not self.quiet:
                            print(f"Error in on_progress callback: {e}")

                return index, result

        async def run_batch(indices_and_prompts):
            sem = asyncio.Semaphore(self.max_concurrent_requests)
            delay = 60.0 / max(1, self.requests_per_minute)

            with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                loop = asyncio.get_running_loop()
                tasks = []
                for idx, prompt in indices_and_prompts:
                    task = asyncio.create_task(async_worker(idx, prompt, sem, delay, executor, loop))
                    tasks.append(task)
                    await asyncio.sleep(delay)

                return await asyncio.gather(*tasks)

        def _run_async(indices_and_prompts):
            try:
                return asyncio.run(run_batch(indices_and_prompts))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(run_batch(indices_and_prompts))

        if not self.quiet:
            print(f"Starting async batch for {len(prompts)} prompts...")

        # Initial run
        all_pairs = list(enumerate(prompts))
        raw_results = _run_async(all_pairs)

        results: List[Optional[Dict]] = [None] * len(prompts)
        for idx, result in raw_results:
            results[idx] = result

        # Retry rounds for failed items
        for retry_round in range(self.batch_max_retry_rounds):
            failed = [(i, prompts[i]) for i in range(len(prompts)) if results[i] is None]
            if not failed:
                break
            if not self.quiet:
                print(f"  Retry round {retry_round + 1}/{self.batch_max_retry_rounds}: "
                      f"{len(failed)} failed items")
            cooldown = min(30.0 * (retry_round + 1), 90.0)
            if not self.quiet:
                print(f"  Waiting {cooldown:.0f}s before retry round...")
            time.sleep(cooldown)
            retry_results = _run_async(failed)
            for idx, result in retry_results:
                if result is not None:
                    results[idx] = result

        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0 and not self.quiet:
            print(f"  {n_failed}/{len(prompts)} items still failed after all retries")

        return results

    def _call_llm_for_json_batch_vllm(
        self,
        prompts: List[str],
        system_prompt: str,
        pbar: Optional[Any] = None
    ) -> List[Optional[Dict]]:
        """
        Batch LLM calls for vLLM using native batch processing.
        
        Args:
            prompts: List of prompts to process
            system_prompt: System prompt for all calls
            pbar: Optional tqdm progress bar
            
        Returns:
            List of parsed JSON dicts (or None for failures)
        """
        if not prompts or self.local_llm is None:
            return []

        from vllm import SamplingParams
        
        # Build all prompt texts
        prompt_texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            try:
                prompt_text = self.local_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except (TypeError, Exception) as e:
                if "add_generation_prompt" in str(e):
                    prompt_text = self.local_tokenizer.apply_chat_template(
                        messages, tokenize=False
                    )
                else:
                    raise
            prompt_texts.append(prompt_text)

        # Single batch call to vLLM
        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=self.temperature if self.temperature is not None else 0.0
        )
        
        if not self.quiet:
            print(f"Running vLLM batch for {len(prompts)} prompts...")
        
        outputs = self.local_llm.generate(prompt_texts, sampling_params)

        # Parse results
        results: List[Optional[Dict]] = []
        for i, out in enumerate(outputs):
            try:
                response_text = out.outputs[0].text if out.outputs else ""
                parsed = self._extract_json_from_response(response_text)
                results.append(parsed)
            except Exception as e:
                if not self.quiet:
                    print(f"Error parsing vLLM response {i}: {e}")
                results.append(None)
            
            if pbar is not None:
                pbar.update(1)
        
        return results

    def _call_llm_for_json_batch(
        self,
        prompts: List[str],
        system_prompt: str,
        pbar: Optional[Any] = None,
        on_progress: Optional[Callable[[int, Optional[Dict]], None]] = None
    ) -> List[Optional[Dict]]:
        """
        Unified batch processing for both API and vLLM backends.
        
        Args:
            prompts: List of prompts to process
            system_prompt: System prompt for all calls
            pbar: Optional tqdm progress bar
            on_progress: Optional callback(index, result) for API batch
            
        Returns:
            List of parsed JSON dicts (or None for failures)
        """
        if not prompts:
            return []
        
        if self.local_llm is not None:
            # vLLM batch processing
            return self._call_llm_for_json_batch_vllm(prompts, system_prompt, pbar)
        else:
            # API batch processing
            return self._call_llm_for_json_batch_api(prompts, system_prompt, pbar, on_progress)
