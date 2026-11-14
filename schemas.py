# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Data schemas for rollout analysis pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class StepBoundary:
    """A boundary defining a single step in the golden solution."""
    step_number: int
    start_line: int
    end_line: int
    step_description: str


@dataclass
class StepText:
    """Verbatim text for a single step."""
    step_number: int
    verbatim_text: str


@dataclass
class Stage1Output:
    """Output schema for Stage 1: Solution Splitting."""
    problem_id: str
    problem_path: str
    timestamp: str
    analyzer_model: str
    golden_solution_raw: str
    golden_solution_with_lines: str
    total_lines: int
    split_boundaries: List[StepBoundary]
    step_texts: List[StepText]
    answer_requirements: Optional[str] = None
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "problem_id": self.problem_id,
            "problem_path": self.problem_path,
            "timestamp": self.timestamp,
            "analyzer_model": self.analyzer_model,
            "golden_solution_raw": self.golden_solution_raw,
            "golden_solution_with_lines": self.golden_solution_with_lines,
            "total_lines": self.total_lines,
            "split_boundaries": [asdict(sb) for sb in self.split_boundaries],
            "step_texts": [asdict(st) for st in self.step_texts]
        }
        if self.answer_requirements is not None:
            d["answer_requirements"] = self.answer_requirements
        if self.code is not None:
            d["code"] = self.code
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stage1Output':
        """Reconstruct from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            problem_path=data["problem_path"],
            timestamp=data["timestamp"],
            analyzer_model=data["analyzer_model"],
            golden_solution_raw=data["golden_solution_raw"],
            golden_solution_with_lines=data["golden_solution_with_lines"],
            total_lines=data["total_lines"],
            split_boundaries=[
                StepBoundary(**sb) for sb in data["split_boundaries"]
            ],
            step_texts=[
                StepText(**st) for st in data["step_texts"]
            ],
            answer_requirements=data.get("answer_requirements"),
            code=data.get("code"),
        )


@dataclass
class CoTStepBoundary:
    """A boundary defining a single step in the model's CoT."""
    step_number: int
    start_line: int
    end_line: int
    step_description: str


@dataclass
class CoTStepText:
    """Verbatim text for a single CoT step."""
    step_number: int
    verbatim_text: str


@dataclass
class Stage2Output:
    """Output schema for Stage 2: CoT Distillation."""
    problem_id: str
    problem_path: str
    timestamp: str
    analyzer_model: str
    target_model: str
    attempt_index: int
    attempt_timestamp: str
    attempt_is_correct: Optional[bool]
    cot_rollout_raw: str
    cot_rollout_with_lines: str
    total_lines: int
    split_boundaries: List[CoTStepBoundary]
    step_texts: List[CoTStepText]
    dedup_original_tokens: Optional[int] = None
    dedup_result_tokens: Optional[int] = None
    dedup_reduction_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "problem_id": self.problem_id,
            "problem_path": self.problem_path,
            "timestamp": self.timestamp,
            "analyzer_model": self.analyzer_model,
            "target_model": self.target_model,
            "attempt_index": self.attempt_index,
            "attempt_timestamp": self.attempt_timestamp,
            "attempt_is_correct": self.attempt_is_correct,
            "cot_rollout_raw": self.cot_rollout_raw,
            "cot_rollout_with_lines": self.cot_rollout_with_lines,
            "total_lines": self.total_lines,
            "split_boundaries": [asdict(sb) for sb in self.split_boundaries],
            "step_texts": [asdict(st) for st in self.step_texts]
        }
        if self.dedup_original_tokens is not None:
            d["dedup_original_tokens"] = self.dedup_original_tokens
            d["dedup_result_tokens"] = self.dedup_result_tokens
            d["dedup_reduction_pct"] = self.dedup_reduction_pct
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stage2Output':
        """Reconstruct from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            problem_path=data["problem_path"],
            timestamp=data["timestamp"],
            analyzer_model=data["analyzer_model"],
            target_model=data["target_model"],
            attempt_index=data["attempt_index"],
            attempt_timestamp=data["attempt_timestamp"],
            attempt_is_correct=data.get("attempt_is_correct"),
            cot_rollout_raw=data["cot_rollout_raw"],
            cot_rollout_with_lines=data["cot_rollout_with_lines"],
            total_lines=data["total_lines"],
            split_boundaries=[
                CoTStepBoundary(**sb) for sb in data["split_boundaries"]
            ],
            step_texts=[
                CoTStepText(**st) for st in data["step_texts"]
            ],
            dedup_original_tokens=data.get("dedup_original_tokens"),
            dedup_result_tokens=data.get("dedup_result_tokens"),
            dedup_reduction_pct=data.get("dedup_reduction_pct"),
        )


@dataclass
class MatchedStep:
    """A matched step from distilled CoT to golden step."""
    cot_step_number: int
    cot_step_text: str
    golden_step_mapping: Optional[int]
    mapping_confidence: str  # "high", "medium", "low", "none"
    mapping_explanation: str
    validity: str  # "valid" or "invalid"
    validity_explanation: str


@dataclass
class Stage3Summary:
    """Summary statistics for Stage 3 matching."""
    total_cot_steps: int
    mapped_steps: int
    unmapped_steps: int
    valid_steps: int
    invalid_steps: int
    coverage: float  # What fraction of golden steps are covered


@dataclass
class Stage3Output:
    """Output schema for Stage 3: Step Matching."""
    problem_id: str
    problem_path: str
    timestamp: str
    analyzer_model: str
    target_model: str
    attempt_index: int
    attempt_is_correct: Optional[bool]  # Whether the attempt was verified as correct
    cot_steps_reference: str  # Path to Stage 2 output
    golden_steps_reference: str  # Path to Stage 1 output
    matched_steps: List[MatchedStep]
    summary: Stage3Summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_id": self.problem_id,
            "problem_path": self.problem_path,
            "timestamp": self.timestamp,
            "analyzer_model": self.analyzer_model,
            "target_model": self.target_model,
            "attempt_index": self.attempt_index,
            "attempt_is_correct": self.attempt_is_correct,
            "cot_steps_reference": self.cot_steps_reference,
            "golden_steps_reference": self.golden_steps_reference,
            "matched_steps": [asdict(ms) for ms in self.matched_steps],
            "summary": asdict(self.summary)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stage3Output':
        """Reconstruct from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            problem_path=data["problem_path"],
            timestamp=data["timestamp"],
            analyzer_model=data["analyzer_model"],
            target_model=data["target_model"],
            attempt_index=data["attempt_index"],
            attempt_is_correct=data.get("attempt_is_correct"),
            cot_steps_reference=data["cot_steps_reference"],
            golden_steps_reference=data["golden_steps_reference"],
            matched_steps=[
                MatchedStep(**ms) for ms in data["matched_steps"]
            ],
            summary=Stage3Summary(**data["summary"])
        )


def validate_stage1_output(data: Dict[str, Any]) -> bool:
    """
    Validate Stage 1 JSON output structure.

    Args:
        data: Dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["problem_id", "split_boundaries"]
    for field in required_fields:
        if field not in data:
            return False

    if not isinstance(data.get("split_boundaries"), list):
        return False

    for boundary in data["split_boundaries"]:
        if not all(k in boundary for k in ["step_number", "start_line", "end_line"]):
            return False

    return True


@dataclass
class ErrorEntry:
    """A single classified error in a model attempt."""
    cot_step_number: int
    error_category: str
    error_scope: str  # "global" or "local"
    is_new_category: bool
    explanation: str
    severity: str  # "major", "minor"


@dataclass
class Stage4Output:
    """Output schema for Stage 4: Error Classification."""
    problem_id: str
    problem_path: str
    timestamp: str
    analyzer_model: str
    target_model: str
    attempt_index: int
    attempt_is_correct: bool
    stage2_reference: str
    stage1_reference: str
    errors: List[ErrorEntry]
    primary_error_category: str
    error_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_path": self.problem_path,
            "timestamp": self.timestamp,
            "analyzer_model": self.analyzer_model,
            "target_model": self.target_model,
            "attempt_index": self.attempt_index,
            "attempt_is_correct": self.attempt_is_correct,
            "stage2_reference": self.stage2_reference,
            "stage1_reference": self.stage1_reference,
            "errors": [asdict(e) for e in self.errors],
            "primary_error_category": self.primary_error_category,
            "error_summary": self.error_summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stage4Output':
        return cls(
            problem_id=data["problem_id"],
            problem_path=data["problem_path"],
            timestamp=data["timestamp"],
            analyzer_model=data["analyzer_model"],
            target_model=data["target_model"],
            attempt_index=data["attempt_index"],
            attempt_is_correct=data["attempt_is_correct"],
            stage2_reference=data.get("stage2_reference", data.get("stage2a_reference", "")),
            stage1_reference=data["stage1_reference"],
            errors=[ErrorEntry(**e) for e in data["errors"]],
            primary_error_category=data["primary_error_category"],
            error_summary=data["error_summary"]
        )


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
