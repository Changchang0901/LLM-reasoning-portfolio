# Curated collaborative research code; see docs/PROVENANCE.md.
import argparse
import logging
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

from solution_generator import (
    SolutionGenerator,
)
from solution_grader import (
    CodeVerificationGrader,
)

# Add parent directory to path to import file_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from file_utils.file_utils import remove_all_solutions


def parse_arguments():
    parser = argparse.ArgumentParser(description="main script for the project, which can be used to generate solutions and grade them.")
    
    # Operation mode
    parser.add_argument("--grading", action="store_true", default=False, help="Set to grade solutions. If not set, generates solutions and then grades them.")
    parser.add_argument("--unit_test", action="store_true", default=False, help="Set to run unit tests.")
    
    # Input/output paths
    parser.add_argument("--problems_dir", type=str, default="data/demo", help="Target directory for problems.")
    parser.add_argument("--problem_path", type=str, default=None, help="Path to a single problem file. When set, problems_dir is ignored.")
    parser.add_argument("--exclude_dir", action="append", default=[], help="Exclude a subdirectory while scanning problems. Repeat as needed (for example: --exclude_dir local_only).")
    
    # Generation parameters
    parser.add_argument("--model_name", type=str, default="demo-no-api", help="Model name to use.")
    parser.add_argument("--num_attempts", type=int, default=1, help="Number of attempts to generate solutions.")
    parser.add_argument("--prompt_type", type=str, default="CoT", choices=['standard', 'CoT'], help="Type of prompt to use.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation. If not provided, uses each model's default temperature.")
    parser.add_argument("--multi_gpu", type=int, default=1, help="Number of GPUs to use for local models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization fraction (0.0-1.0) for vLLM. Default: 0.95")
    parser.add_argument("--model_alias", type=str, default=None, help="Custom model name for results (e.g., 'oss-20b-100-step'). If not set, uses model_name.")
    parser.add_argument("--batch_size", type=int, default=None, help="Global batch size for solution generation (shared with vLLM batching). Defaults to requests_per_minute for API models, 16 for local.")
    parser.add_argument("--requests_per_minute", type=int, default=60, help="Rate limit for API requests per minute. Conservative Windows/API default: 60.")
    parser.add_argument("--max_concurrent_requests", type=int, default=8, help="Maximum concurrent API requests. Conservative default: 8.")
    parser.add_argument("--reasoning_level", type=str, default=None, choices=["low", "medium", "high"], help="Reasoning level for oss models (low/medium/high). Prepends 'Reasoning: <level>' to system prompt.")
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high", "xhigh", "max"], help="Reasoning effort for models that support output_config.effort (Claude Opus/Fable). Defaults to max.")
    
    # Behavior flags
    parser.add_argument("--overwrite_attempts", action="store_true", default=False, help="Set to overwrite existing attempts instead of appending.")
    parser.add_argument("--regrade_all", action="store_true", default=False, help="Regrade all attempts, even if already graded (default: skip already graded).")
    
    # Grading parameters
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress debug output during grading.")
    parser.add_argument("--latex_report", action="store_true", default=False, help="Generate and compile the optional LaTeX report after grading (requires pdflatex).")
    
    # Configuration
    parser.add_argument("--config_path", type=str, default=None, help="Path to custom config file.")
    
    return parser.parse_args()


def create_grader(args):
    # Code verification doesn't use model_name, multi_gpu, config_path, or temperature
    # These are only kept for BaseProblemProcessor compatibility
    grader_kwargs = {
        "problems_dir": args.problems_dir,
        "quiet": args.quiet,
        "exclude_dirs": args.exclude_dir,
    }
    
    # Optional parameters (not used by code verification but required by base class)
    if args.config_path:
        grader_kwargs["config_path"] = args.config_path
    
    return CodeVerificationGrader(**grader_kwargs)


def create_generator(args):
    generator_kwargs = {
        "model_name": args.model_name,
        "prompt_type": args.prompt_type,
        "multi_gpu": args.multi_gpu,
        "problems_dir": args.problems_dir,
        "overwrite_attempts": args.overwrite_attempts,
        "batch_size": args.batch_size,
        "requests_per_minute": args.requests_per_minute,
        "max_concurrent_requests": args.max_concurrent_requests,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "exclude_dirs": args.exclude_dir,
    }
    if args.config_path:
        generator_kwargs["config_path"] = args.config_path
    if args.temperature is not None:
        generator_kwargs["temperature"] = args.temperature
    if args.model_alias:
        generator_kwargs["model_alias"] = args.model_alias
    if args.reasoning_effort:
        generator_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_level:
        generator_kwargs["reasoning_level"] = args.reasoning_level
        # Append reasoning level to model alias for oss models
        if "oss" in args.model_name:
            base_alias = args.model_alias if args.model_alias else args.model_name
            generator_kwargs["model_alias"] = f"{base_alias}-{args.reasoning_level}"
    return SolutionGenerator(**generator_kwargs)


def run_latex_report(args):
    """Run latex generation for analysis report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating analysis report...")
    
    # Construct command
    cmd = [
        sys.executable,
        "main/generate_latex.py", 
        "--analysis_only", 
        "--pdf",
        "--problems_dir", args.problems_dir
    ]
    
    # Add optional args if they affect the report
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
        
    try:
        subprocess.run(cmd, check=True)
        logger.info("Analysis report generated successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate analysis report: {e}")


def run_grading(args):
    logger = logging.getLogger(__name__)
    grader = create_grader(args)
    logger.info("%s", grader)

    if args.problem_path is not None:
        grader._grade_problem(
            problem_path=args.problem_path,
            regrade_all=args.regrade_all,
        )
    else:
        logger.info("Grading all problems")
        grader.grade_all_problems(
            regrade_all=args.regrade_all,
        )
        
        if args.latex_report:
            run_latex_report(args)


def run_generation(args):
    if args.model_name == "demo-no-api":
        raise ValueError("Generation requires an explicit --model_name; use --grading for offline verification.")
    generator = create_generator(args)
    try:
        if args.problem_path is not None:
            generator.generate_solutions_for_problem(problem_path=args.problem_path, num_attempts=args.num_attempts)
        else:
            generator.generate_solutions_for_all_problems(num_attempts=args.num_attempts)
    finally:
        generator._terminate_subprocesses()


def main():
    # Basic logging setup for CLI entrypoint
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    
    if args.unit_test:
        logger.info("Unit Testing Mode")
        args.problems_dir = "data/demo"
        args.overwrite_attempts = True
        args.regrade_all = True
        
        # Remove all solutions from all files in unit test directory
        unit_test_dir = Path(args.problems_dir)
        if unit_test_dir.exists() and not args.grading:
            logger.info(f"Removing all solutions from files in {unit_test_dir}")
            for json_file in unit_test_dir.rglob("*.json"):
                try:
                    remove_all_solutions(json_file)
                    logger.info(f"Removed all solutions from {json_file.name}")
                except Exception as e:
                    logger.warning(f"Error removing solutions from {json_file.name}: {e}")
        else:
            logger.warning(f"Unit test directory {unit_test_dir} does not exist")


    args.problem_path = Path(args.problem_path) if args.problem_path else None
    if args.grading:
        run_grading(args)
    else:
        run_generation(args)
        # Automatically run code grading after generation
        run_grading(args)
   

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
