# Curated collaborative research code; see docs/PROVENANCE.md.
"""
Utility functions for rollout analysis pipeline.
"""

from typing import List, Tuple, Optional, Dict, Any
import re


def add_line_numbers(text: str) -> Tuple[str, int]:
    """
    Add line numbers to text for Stage 1 prompt.

    Args:
        text: The text to number

    Returns:
        Tuple of (numbered_text, total_lines)
    """
    lines = text.split('\n')
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:4d} | {line}")
    return '\n'.join(numbered_lines), len(lines)


def extract_lines(text: str, start_line: int, end_line: int) -> str:
    """
    Extract specific lines from text (1-indexed, inclusive).

    Args:
        text: Source text
        start_line: First line to extract (1-indexed)
        end_line: Last line to extract (1-indexed, inclusive)

    Returns:
        Extracted lines joined by newlines
    """
    lines = text.split('\n')
    # Convert to 0-indexed
    return '\n'.join(lines[start_line - 1:end_line])


def extract_code_blocks(text: str) -> str:
    """
    Extract Python code blocks from text before they are removed.

    Finds all ```python...``` and \\begin{python}...\\end{python} blocks
    and returns their contents concatenated.

    Args:
        text: Text potentially containing code blocks

    Returns:
        Concatenated code block contents, or empty string if none found
    """
    blocks = []
    # Match ```python ... ``` blocks (case-insensitive)
    for m in re.finditer(r'```[pP]ython\s*\n(.*?)\n```', text, flags=re.DOTALL):
        blocks.append(m.group(1).strip())
    # Match \begin{python} ... \end{python} blocks
    for m in re.finditer(r'\\begin\{python\}(.*?)\\end\{python\}', text, flags=re.DOTALL):
        blocks.append(m.group(1).strip())
    # Match generic ``` blocks that look like code
    for m in re.finditer(r'```\s*\n((?:import|def |class ).*?)\n```', text, flags=re.DOTALL):
        content = m.group(1).strip()
        if content not in blocks:
            blocks.append(content)
    return "\n\n".join(blocks)


def remove_code_blocks(text: str) -> str:
    """
    Remove Python code blocks from solution text for analysis.
    Keeps only mathematical derivation.

    Args:
        text: Solution text potentially containing code blocks

    Returns:
        Text with code blocks removed
    """
    # Remove ```python ... ``` blocks
    text = re.sub(r'```python\s*\n.*?\n```', '', text, flags=re.DOTALL)
    # Remove ```Python ... ``` blocks (case variation)
    text = re.sub(r'```Python\s*\n.*?\n```', '', text, flags=re.DOTALL)
    # Remove generic ``` ... ``` blocks that look like code
    text = re.sub(r'```\s*\n(?:import|def |class ).*?\n```', '', text, flags=re.DOTALL)
    # Remove \begin{python} ... \end{python} blocks
    text = re.sub(r'\\begin\{python\}.*?\\end\{python\}', '', text, flags=re.DOTALL)
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_golden_steps_for_prompt(step_texts: List[Dict[str, Any]]) -> str:
    """
    Format golden steps for Stage 2 prompt.

    Args:
        step_texts: List of step dictionaries with 'step_number' and 'verbatim_text'

    Returns:
        Formatted string for prompt inclusion
    """
    formatted = []
    for step in step_texts:
        step_num = step.get('step_number', '?')
        text = step.get('verbatim_text', '')
        formatted.append(f"### Step {step_num}\n{text}")
    return '\n\n'.join(formatted)


def find_text_position(full_text: str, excerpt: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Find start and end character positions of excerpt in full_text.
    Uses fuzzy matching if exact match fails.

    Args:
        full_text: The complete text to search in
        excerpt: The excerpt to find

    Returns:
        Tuple of (start_position, end_position) or (None, None) if not found
    """
    if not excerpt:
        return None, None

    # Try exact match first
    pos = full_text.find(excerpt)
    if pos != -1:
        return pos, pos + len(excerpt)

    # Try normalized matching (collapse whitespace)
    normalized_full = ' '.join(full_text.split())
    normalized_excerpt = ' '.join(excerpt.split())
    pos = normalized_full.find(normalized_excerpt)
    if pos != -1:
        # Return approximate position (won't be exact due to whitespace changes)
        return pos, pos + len(normalized_excerpt)

    return None, None


def compute_coverage(extracted_steps: List[Dict[str, Any]], total_golden_steps: int) -> float:
    """
    Compute what fraction of golden steps have at least one mapping.

    Args:
        extracted_steps: List of extracted step dictionaries
        total_golden_steps: Total number of golden steps

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if total_golden_steps == 0:
        return 0.0

    mapped_golden = set()
    for step in extracted_steps:
        mapping = step.get('golden_step_mapping')
        if mapping is not None:
            mapped_golden.add(mapping)

    return len(mapped_golden) / total_golden_steps


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: The string to sanitize

    Returns:
        Sanitized filename-safe string
    """
    # Replace path separators and spaces
    safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    # Remove other problematic characters
    safe = re.sub(r'[<>:"|?*]', '', safe)
    return safe
