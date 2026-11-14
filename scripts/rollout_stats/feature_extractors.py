# Curated collaborative research code; see docs/PROVENANCE.md.
#!/usr/bin/env python3
"""
Deterministic feature extractors for model rollout responses.

All features are computed deterministically using standard libraries only.
No ML models or probabilistic methods are used.
"""

import re
import math
from collections import Counter
from typing import Dict, Optional, List

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Token-based features will be disabled.")


# Mathematical symbols to count
MATH_SYMBOLS = {
    '∀', '∃', '∈', '∉', '⊂', '⊆', '∪', '∩', '∅', '∑', '∏', '∫', '∮',
    '∇', '∂', '∞', '±', '∓', '×', '÷', '≤', '≥', '≠', '≈', '≡', '∝',
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν',
    'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Γ', 'Δ', 'Θ',
    'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω'
}


def calculate_entropy(counts: Dict[str, int]) -> float:
    """Calculate Shannon entropy from a counter dictionary."""
    if not counts:
        return 0.0
    
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def extract_length_features(text: str) -> Dict[str, float]:
    """Extract length-related features."""
    if not text:
        return {
            'length_chars': 0,
            'length_words': 0,
            'length_tokens': 0,
            'length_lines': 0
        }
    
    # Character count
    char_count = len(text)
    
    # Word count (split on whitespace)
    words = text.split()
    word_count = len(words)
    
    # Token count using tiktoken
    token_count = None
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (GPT-4 tokenizer)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            token_count = len(tokens)
        except Exception as e:
            # If tokenization fails, return 0
            token_count = None
    
    # Line count
    line_count = len(text.splitlines())
    
    return {
        'length_chars': float(char_count),
        'length_words': float(word_count),
        'length_tokens': float(token_count) if token_count is not None else None,
        'length_lines': float(line_count)
    }


def extract_entropy_features(text: str) -> Dict[str, Optional[float]]:
    """Extract entropy-related features."""
    if not text:
        return {
            'token_entropy': None,
            'char_entropy': None,
            'word_entropy': None
        }
    
    results = {
        'token_entropy': None,
        'char_entropy': None,
        'word_entropy': None
    }
    
    # Character-level entropy
    char_counts = Counter(text)
    results['char_entropy'] = calculate_entropy(char_counts)
    
    # Word-level entropy
    words = text.split()
    if words:
        word_counts = Counter(words)
        results['word_entropy'] = calculate_entropy(word_counts)
    
    # Token-level entropy using tiktoken
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if tokens:
                # Convert tokens back to strings for counting
                token_strings = [str(t) for t in tokens]
                token_counts = Counter(token_strings)
                results['token_entropy'] = calculate_entropy(token_counts)
        except Exception:
            pass
    
    return results


def extract_math_features(text: str) -> Dict[str, float]:
    """Extract mathematical content features."""
    if not text:
        return {
            'math_char_density': 0.0,
            'math_symbol_ratio': 0.0,
            'equation_block_count': 0,
            'math_mode_fraction': 0.0,
            'latex_command_count': 0
        }
    
    total_chars = len(text)
    
    # Count LaTeX math mode content
    # Match $...$ (inline math) and $$...$$ (display math)
    inline_math_pattern = r'\$([^$]+)\$'
    display_math_pattern = r'\$\$([^$]+)\$\$'
    
    inline_matches = re.findall(inline_math_pattern, text)
    display_matches = re.findall(display_math_pattern, text)
    
    # Count characters in math mode
    math_chars = sum(len(match) for match in inline_matches)
    math_chars += sum(len(match) for match in display_matches)
    
    # Also count \begin{equation}, \begin{align}, etc.
    equation_blocks = len(re.findall(r'\\begin\{(equation|align|eqnarray|multline|gather|split)\}', text))
    
    # Count LaTeX commands (backslash followed by letters)
    latex_commands = len(re.findall(r'\\[a-zA-Z]+\b', text))
    
    # Count mathematical symbols
    math_symbol_count = sum(1 for char in text if char in MATH_SYMBOLS)
    
    # Calculate densities and ratios
    math_char_density = math_chars / total_chars if total_chars > 0 else 0.0
    math_symbol_ratio = math_symbol_count / total_chars if total_chars > 0 else 0.0
    math_mode_fraction = math_chars / total_chars if total_chars > 0 else 0.0
    
    return {
        'math_char_density': math_char_density,
        'math_symbol_ratio': math_symbol_ratio,
        'equation_block_count': float(equation_blocks),
        'math_mode_fraction': math_mode_fraction,
        'latex_command_count': float(latex_commands)
    }


def extract_code_features(text: str) -> Dict[str, float]:
    """Extract code-related features."""
    if not text:
        return {
            'code_block_count': 0,
            'code_line_count': 0,
            'code_char_density': 0.0
        }
    
    total_chars = len(text)
    
    # Match Python code blocks: ```python ... ```
    python_block_pattern = r'```python\s*\n(.*?)\n```'
    python_blocks = re.findall(python_block_pattern, text, re.DOTALL)
    
    # Also match generic code blocks that might contain Python
    generic_block_pattern = r'```\s*\n(.*?)\n```'
    generic_blocks = re.findall(generic_block_pattern, text, re.DOTALL)
    
    # Count code blocks (prioritize Python blocks)
    code_block_count = len(python_blocks) if python_blocks else len(generic_blocks)
    
    # Count code lines
    code_lines = 0
    code_chars = 0
    
    if python_blocks:
        for block in python_blocks:
            lines = block.splitlines()
            code_lines += len([l for l in lines if l.strip()])
            code_chars += len(block)
    elif generic_blocks:
        for block in generic_blocks:
            lines = block.splitlines()
            code_lines += len([l for l in lines if l.strip()])
            code_chars += len(block)
    
    code_char_density = code_chars / total_chars if total_chars > 0 else 0.0
    
    return {
        'code_block_count': float(code_block_count),
        'code_line_count': float(code_lines),
        'code_char_density': code_char_density
    }


def extract_structural_features(text: str) -> Dict[str, float]:
    """Extract structural features."""
    if not text:
        return {
            'paragraph_count': 0,
            'avg_sentence_length': 0.0,
            'heading_count': 0
        }
    
    # Paragraph count (split on double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Sentence count (split on sentence-ending punctuation)
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Average sentence length (in words)
    words = text.split()
    word_count = len(words)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
    
    # Heading count (markdown headers: #, ##, ###, etc.)
    heading_count = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
    
    return {
        'paragraph_count': float(paragraph_count),
        'avg_sentence_length': avg_sentence_length,
        'heading_count': float(heading_count)
    }


def extract_all_features(text: str) -> Dict[str, Optional[float]]:
    """
    Extract all features from response text.
    
    Returns a dictionary with all feature values.
    Features that cannot be computed return None.
    """
    if not text:
        text = ""
    
    features = {}
    
    # Length features
    features.update(extract_length_features(text))
    
    # Entropy features
    features.update(extract_entropy_features(text))
    
    # Math features
    features.update(extract_math_features(text))
    
    # Code features
    features.update(extract_code_features(text))
    
    # Structural features
    features.update(extract_structural_features(text))
    
    return features


def get_feature_names() -> List[str]:
    """Return list of all feature names."""
    return [
        'length_chars',
        'length_words',
        'length_tokens',
        'length_lines',
        'token_entropy',
        'char_entropy',
        'word_entropy',
        'math_char_density',
        'math_symbol_ratio',
        'equation_block_count',
        'math_mode_fraction',
        'latex_command_count',
        'code_block_count',
        'code_line_count',
        'code_char_density',
        'paragraph_count',
        'avg_sentence_length',
        'heading_count'
    ]
