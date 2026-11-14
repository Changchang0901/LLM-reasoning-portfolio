# Curated collaborative research code; see docs/PROVENANCE.md.
import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Optional, Tuple
import simplejson as json
import re
from config import COMMON_PATHS, DEFAULT_SETTINGS, CHART_SETTINGS


def _verification_results(attempt):
    """Normalize current dict results and legacy list results for all views."""
    value = attempt.get("verifier_result")
    if isinstance(value, dict):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


class ProblemAnalyzer:
    def __init__(self):
        self.problems = []
        self.analysis_data = {}
        
    def load_problems_from_folder(self, folder_path: str) -> List[dict]:
        """Load all JSON files from a folder"""
        problems = []
        folder = Path(folder_path)
        
        if not folder.exists():
            st.error(f"Folder {folder_path} does not exist!")
            return problems
            
        json_files = list(folder.rglob("*.json"))
        
        if not json_files:
            st.warning(f"No JSON files found in {folder_path}")
            return problems
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, json_file in enumerate(json_files):
            status_text.text(f"Loading {json_file.name}...")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    problem = json.load(f)
                    problems.append(problem)
            except Exception as e:
                st.warning(f"Error loading {json_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(json_files))
        
        status_text.text("Loading complete!")
        progress_bar.empty()
        status_text.empty()
        
        return problems
    
    def calculate_verification_metrics(self, problems: List[dict]) -> Dict[str, Dict]:
        """Calculate verification success rates for each model"""
        verification_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'unknown': 0})
        
        for problem in problems:
            for solution in problem.get('model_solutions', []):
                model = solution.get('model')
                for attempt in solution.get('attempts', []):
                    if 'verifier_result' not in attempt:
                        continue
                    
                    for result in _verification_results(attempt):
                        status = result.get('verified')
                        if status == 1:
                            verification_stats[model]['correct'] += 1
                        elif status == 0:
                            verification_stats[model]['incorrect'] += 1
                        else:
                            verification_stats[model]['unknown'] += 1
        
        # Calculate success rates
        for model, stats in verification_stats.items():
            total_verified = stats['correct'] + stats['incorrect']
            stats['success_rate'] = stats['correct'] / total_verified if total_verified > 0 else 0
            stats['total_attempts'] = stats['correct'] + stats['incorrect'] + stats['unknown']
        
        return dict(verification_stats)
    
    def calculate_best_of_n_accuracy(self, problems: List[dict], n: int = 5) -> Dict[str, Dict]:
        """Calculate best of N accuracy for each model"""
        best_of_n_stats = defaultdict(lambda: {'solved_problems': 0, 'total_problems': 0})
        
        for problem in problems:
            problem_id = problem.get('problem_id', 'Unknown')
            
            for solution in problem.get('model_solutions', []):
                model = solution.get('model')
                best_of_n_stats[model]['total_problems'] += 1
                
                # Check if model solved this problem in any of the first N attempts
                attempts = solution.get('attempts', [])[:n]  # Limit to first N attempts
                solved = False
                
                for attempt in attempts:
                    if 'verifier_result' not in attempt:
                        continue
                    
                    for result in _verification_results(attempt):
                        if result.get('verified') == 1:
                            solved = True
                            break
                    
                    if solved:
                        break
                
                if solved:
                    best_of_n_stats[model]['solved_problems'] += 1
        
        # Calculate best of N accuracy
        for model, stats in best_of_n_stats.items():
            stats['best_of_n_accuracy'] = stats['solved_problems'] / stats['total_problems'] if stats['total_problems'] > 0 else 0
        
        return dict(best_of_n_stats)
    
    def calculate_grade_metrics(self, problems: List[dict]) -> Dict[str, Dict]:
        """Calculate grade distribution for each model"""
        grade_stats = defaultdict(lambda: {'A': 0, 'B': 0, 'C': 0, 'D': 0})
        
        for problem in problems:
            for solution in problem.get('model_solutions', []):
                model = solution.get('model')
                for attempt in solution.get('attempts', []):
                    for evaluation in attempt.get('evaluations', []):
                        grade = evaluation.get('grade')
                        if grade in ['A', 'B', 'C', 'D']:
                            grade_stats[model][grade] += 1
        
        # Calculate average grades
        grade_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        for model, grades in grade_stats.items():
            total = sum(grades.values())
            if total > 0:
                weighted_sum = sum(grade_mapping[grade] * count for grade, count in grades.items())
                grades['average'] = weighted_sum / total
                grades['total'] = total
            else:
                grades['average'] = 0
                grades['total'] = 0
        
        return dict(grade_stats)
    
    def create_problem_model_matrix(self, problems: List[dict], metric_type: str = 'verification') -> pd.DataFrame:
        """Create a matrix showing performance for each problem-model combination"""
        matrix_data = []
        
        for problem in problems:
            problem_id = problem.get('problem_id', 'Unknown')
            difficulty = problem.get('problem_metadata', {}).get('Difficulty level', 'Unknown')
            
            row = {
                'Problem ID': problem_id,
                'Difficulty': difficulty,
                'Domain': problem.get('problem_metadata', {}).get('Domain of theoretical physics', 'Unknown')
            }
            
            for solution in problem.get('model_solutions', []):
                model = solution.get('model')
                
                if metric_type == 'verification':
                    # Calculate verification success rate for this problem
                    correct = 0
                    total = 0
                    for attempt in solution.get('attempts', []):
                        for result in _verification_results(attempt):
                            status = result.get('verified')
                            if status is not None:
                                total += 1
                                if status == 1:
                                    correct += 1
                    
                    success_rate = correct / total if total > 0 else 0
                    row[model] = f"{success_rate:.2f}" if total > 0 else "N/A"
                    
                elif metric_type == 'grades':
                    # Calculate average grade for this problem
                    grades = []
                    for attempt in solution.get('attempts', []):
                        for evaluation in attempt.get('evaluations', []):
                            grade = evaluation.get('grade')
                            if grade in ['A', 'B', 'C', 'D']:
                                grade_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
                                grades.append(grade_mapping[grade])
                    
                    avg_grade = sum(grades) / len(grades) if grades else 0
                    row[model] = f"{avg_grade:.2f}" if grades else "N/A"
            
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def create_summary_statistics(self, problems: List[dict]) -> Dict:
        """Create summary statistics for the dataset"""
        total_problems = len(problems)
        models = set()
        difficulties = Counter()
        domains = Counter()
        
        for problem in problems:
            difficulties[problem.get('problem_metadata', {}).get('Difficulty level', 'Unknown')] += 1
            domains[problem.get('problem_metadata', {}).get('Domain of theoretical physics', 'Unknown')] += 1
            
            for solution in problem.get('model_solutions', []):
                models.add(solution.get('model'))
        
        return {
            'total_problems': total_problems,
            'total_models': len(models),
            'models': list(models),
            'difficulty_distribution': dict(difficulties),
            'domain_distribution': dict(domains)
        }


def _attempt_status(attempt: dict) -> Tuple[str, str]:
    """Return (status_text, emoji) based on verifier_result (dict or list for backward compatibility)."""
    verifier_result = attempt.get('verifier_result')
    if not verifier_result:
        return ("Unknown", "❔")
    
    # Handle both dict (new format) and list (old format)
    if isinstance(verifier_result, dict):
        verified_values = [verifier_result.get('verified')]
    elif isinstance(verifier_result, list):
        verified_values = [vr.get('verified') for vr in verifier_result if vr.get('verified') is not None]
    else:
        return ("Unknown", "❔")
    
    if any(v == 1 for v in verified_values):
        return ("Correct", "✅")
    if any(v == 0 for v in verified_values):
        return ("Incorrect", "❌")
    return ("Unknown", "❔")


def _sanitize_latex_snippets(s: str) -> str:
    """Best-effort cleanup for common malformed LaTeX from LLM outputs.

    - Normalizes whitespace after \left/\right
    - Inserts default parentheses when delimiter is missing (e.g., "\\left - ...")
    - Removes stray trailing backslashes inside math content
    """
    if not s:
        return s

    # Remove non-renderable layout tokens
    s = re.sub(r"\\noindent\s*", "", s)
    s = re.sub(r"\\nonumber\b", "", s)

    # Collapse whitespace/newlines after \left and \right
    s = re.sub(r"\\left\s+", r"\\left ", s)
    s = re.sub(r"\\right\s+", r"\\right ", s)

    # If \left/\right is followed by a control sequence or an invalid delimiter, default to parentheses
    s = re.sub(r"\\left\s+(?=\\[A-Za-z])", r"\\left(", s)
    s = re.sub(r"\\right\s+(?=\\[A-Za-z])", r"\\right)", s)
    # Also handle non-delimiter next characters
    s = re.sub(r"\\left\s+(?=[^\\(\[\{\|<])", r"\\left(", s)
    s = re.sub(r"\\right\s+(?=[^\\)\]\}\|>])", r"\\right)", s)

    # Trim stray trailing backslashes at end of math-ish lines
    s = re.sub(r"\$\$(.*?)\\\s*\$\$", lambda m: "$$" + re.sub(r"\\\s*$", "", m.group(1)) + "$$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", lambda m: "\\[" + re.sub(r"\\\s*$", "", m.group(1)) + "\\]", s, flags=re.DOTALL)

    return s


def _segment_latex_for_streamlit(text: str, show_cot_r1: bool = False) -> List[Tuple[str, str]]:
    """Convert LaTeX-ish solution into segments for Streamlit rendering.

    Returns list of (segment_type, content) where segment_type in {'text','code','display_math'}.
    This loosely mirrors transformations in main/latex_generator.py:_process_latex_text.
    """
    if text is None:
        return []

    # Handle DeepSeek-R1 <think> tags
    if "<think>" in text and "</think>" in text:
        if not show_cot_r1:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        else:
            # Keep but wrap as verbatim-like block
            def break_lines(match):
                content = match.group(1)
                lines = []
                for line in content.splitlines():
                    lines.append(line)
                return "\n" + "\n".join(lines) + "\n"
            text = re.sub(r'<think>(.*?)</think>', break_lines, text, flags=re.DOTALL)

    # Remove document-level LaTeX commands
    text = re.sub(r'\\documentclass.*?\n', '', text)
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)
    text = re.sub(r'\\usepackage.*?\n', '', text)

    # Best-effort sanitize malformed LaTeX fragments
    text = _sanitize_latex_snippets(text)

    segments: List[Tuple[str, str]] = []

    # We'll iteratively extract code blocks and math blocks (\begin{python}, ```python, LaTeX math envs, \[...\], $$...$$, $...$, and raw [ ... ] heuristics)
    pattern_python_env = re.compile(r"\\begin\{python\}(.*?)\\end\{python\}", re.DOTALL)
    pattern_fence = re.compile(r"```python\s*(.*?)```", re.DOTALL)
    # Common LaTeX display math environments. KaTeX (used by st.latex) supports aligned but not align; convert align-like to aligned.
    pattern_latex_env = re.compile(r"\\begin\\{(align\*?|equation\*?|gather\*?|eqnarray\*?)\\\}(.*?)\\end\\{\\1\\}", re.DOTALL)
    pattern_display_math_dollars = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    pattern_display_math_bracket = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
    # TikZ-Feynman diagrams are unsupported by KaTeX; show as code
    pattern_feyn = re.compile(r"\\feynmandiagram\s*(\[[^\]]*\])?\s*\{(.*?)\}", re.DOTALL)
    # Inline $ ... $ math (avoid matching $$ ... $$)
    pattern_inline_dollar = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
    # Heuristic for raw bracketed math blocks like: [ ... ]
    pattern_raw_bracket = re.compile(r"\[\s*(.*?)\s*\]", re.DOTALL)

    # Unified scan: find earliest next special block among three patterns
    pos = 0
    while pos < len(text):
        # Find next matches from current pos
        m_env = pattern_python_env.search(text, pos)
        m_fence = pattern_fence.search(text, pos)
        m_latex_env = pattern_latex_env.search(text, pos)
        m_math_dollars = pattern_display_math_dollars.search(text, pos)
        m_math_bracket = pattern_display_math_bracket.search(text, pos)
        m_inline_dollar = pattern_inline_dollar.search(text, pos)
        m_raw = pattern_raw_bracket.search(text, pos)
        m_feyn = pattern_feyn.search(text, pos)

        # Filter raw bracket matches to only those likely containing LaTeX (\alpha, \frac, ^, _ etc.)
        if m_raw:
            raw_content = m_raw.group(1)
            if not re.search(r"\\[a-zA-Z]|\^|_", raw_content):
                m_raw = None

        # Determine earliest
        candidates = [m for m in [m_env, m_fence, m_latex_env, m_math_dollars, m_math_bracket, m_inline_dollar, m_raw, m_feyn] if m]
        if not candidates:
            remaining = text[pos:]
            if remaining:
                # Remove backticks used for inline code in LLM outputs
                remaining = re.sub(r'`', '', remaining)
                # Remove \noindent tokens
                remaining = re.sub(r"\\noindent\s*", "", remaining)
                # Remove stray \nonumber tokens
                remaining = re.sub(r"\\nonumber\b", "", remaining)
                # Normalize LaTeX line breaks: single backslash at EOL and double backslash
                remaining = re.sub(r"\\\s*\n\s*", " ", remaining)
                remaining = remaining.replace("\\\\", "\n")
                # Convert \textbf{...} to markdown bold in text segments
                remaining = re.sub(r"\\textbf\\{([^{}]+)\\}", r"**\\1**", remaining)
                # Further split remaining by inline math in parentheses that include LaTeX markers
                segments.extend(_split_inline_math_segments(remaining))
            break

        next_match = min(candidates, key=lambda m: m.start())

        # Preceding text
        if next_match.start() > pos:
            chunk = text[pos:next_match.start()]
            chunk = re.sub(r'`', '', chunk)
            # Remove \noindent tokens
            chunk = re.sub(r"\\noindent\s*", "", chunk)
            # Remove stray \nonumber tokens
            chunk = re.sub(r"\\nonumber\b", "", chunk)
            # Normalize LaTeX line breaks in plain text
            chunk = re.sub(r"\\\s*\n\s*", " ", chunk)
            chunk = chunk.replace("\\\\", "\n")
            # Convert \textbf{...} to markdown bold in text segments
            chunk = re.sub(r"\\textbf\\{([^{}]+)\\}", r"**\\1**", chunk)
            segments.extend(_split_inline_math_segments(chunk))

        content = next_match.group(1).strip()
        if next_match == m_env or next_match == m_fence:
            segments.append(("code", content))
        elif next_match == m_feyn:
            # Render unsupported diagram as code for readability
            diag_body = next_match.group(0)
            segments.append(("code", diag_body))
        elif next_match == m_latex_env:
            env_name = m_latex_env.group(1)
            env_body = m_latex_env.group(2).strip()
            # Remove \noindent inside math envs
            env_body = re.sub(r"\\noindent\s*", "", env_body)
            # Remove stray \nonumber inside math envs
            env_body = re.sub(r"\\nonumber\b", "", env_body)
            # Map align-like environments to aligned for KaTeX/Streamlit
            if env_name.startswith("align") or env_name.startswith("gather") or env_name.startswith("eqnarray"):
                # Only wrap in aligned if & alignment markers exist
                if '&' in env_body:
                    env_body_wrapped = "\\begin{aligned}\n" + env_body + "\n\\end{aligned}"
                    segments.append(("display_math", env_body_wrapped))
                else:
                    segments.append(("display_math", env_body))
            else:
                # equation or equation* -> just treat inner as display math
                segments.append(("display_math", env_body))
        else:
            # Handle raw bracket that might be a bold heading like \textbf{...}
            if next_match == m_raw:
                m_bold = re.match(r"\\textbf\{(.*?)\}$", content)
                if m_bold:
                    segments.append(("text", f"**{m_bold.group(1)}**"))
                else:
                    # Trim trailing stray backslash
                    content = re.sub(r"\\\s*$", "", content)
                    segments.append(("display_math", content))
            elif next_match == m_inline_dollar:
                # Inline math $ ... $
                content = re.sub(r"\\\s*$", "", content)
                segments.append(("inline_math", content))
            else:
                # display math (either $$...$$ or \[...\])
                content = re.sub(r"\\\s*$", "", content)
                segments.append(("display_math", content))

        pos = next_match.end()

    return segments


def _render_detailed_solution_streamlit(text: str, show_cot_r1: bool = False):
    """Render a detailed solution string by splitting into text/code/math segments."""
    segments = _segment_latex_for_streamlit(text or "", show_cot_r1=show_cot_r1)
    for seg_type, content in segments:
        if seg_type == "code":
            st.code(content, language="python")
        elif seg_type == "display_math":
            # Render with st.latex; ensure stripped
            st.latex(content)
        elif seg_type == "inline_math":
            st.latex(content)
        else:
            # Basic markdown; keep bold **text** by converting to markdown style if present
            st.markdown(content)


def _split_inline_math_segments(text_chunk: str) -> List[Tuple[str, str]]:
    """Split a text chunk into text and inline_math segments for parentheses that include LaTeX markers.

    Heuristic: parentheses content that contains a backslash command or ^/_ is treated as inline math.
    """
    segments: List[Tuple[str, str]] = []
    pattern_inline = re.compile(r"\(([^()]*?(?:\\[a-zA-Z]+|\^|_)[^()]*)\)")
    pos = 0
    while True:
        m = pattern_inline.search(text_chunk, pos)
        if not m:
            tail = text_chunk[pos:]
            if tail:
                segments.append(("text", tail))
            break
        if m.start() > pos:
            segments.append(("text", text_chunk[pos:m.start()]))
        content = m.group(1).strip()
        # Remove trailing stray backslash in inline math content
        content = re.sub(r"\\\s*$", "", content)
        segments.append(("inline_math", content))
        pos = m.end()
    return segments


def main():
    st.set_page_config(
        page_title="Problem Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Physics Evaluation Portfolio")
    st.caption("Educational fixtures and hand-written candidate responses. All displayed scores are demonstration outcomes.")
    st.markdown("Analyze model performance across physics problems")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ProblemAnalyzer()
    
    # Add a button to clear session state (useful for development)
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Sidebar for folder selection
    st.sidebar.header("📁 Data Selection")
    
    # Quick path selection
    st.sidebar.subheader("Quick Select")
    selected_path = st.sidebar.selectbox(
        "Choose a Problem Set path:",
        list(COMMON_PATHS.keys()),
        help="Select from problem set folder paths"
    )
    
    folder_path = COMMON_PATHS[selected_path]
    st.sidebar.info(f"Selected: {folder_path}")
    
    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading problems..."):
            st.session_state.problems = st.session_state.analyzer.load_problems_from_folder(folder_path)
            st.session_state.data_loaded = True
    
    # Check if data is loaded
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.info("👈 Please select a folder and click 'Load Data' to begin analysis")
        return
    
    problems = st.session_state.problems
    
    if not problems:
        st.error("No problems loaded. Please check your folder path.")
        return
    
    # Analysis options
    st.sidebar.header("⚙️ Analysis Options")
    metric_type = "verification"  # Always use verification metrics
    
    
    # Best of N settings
    st.sidebar.subheader("Best of N Settings")
    best_of_n = st.sidebar.slider(
        "Best of N attempts",
        min_value=1,
        max_value=50,
        value=5,
        help="Number of attempts to consider for 'Best of N' accuracy calculation"
    )
    
    # Additional options
    st.sidebar.subheader("Display Options")
    show_pie_charts = st.sidebar.checkbox(
        "Show Distribution Pie Charts",
        value=False,
        help="Display difficulty and domain distribution pie charts"
    )
    
    max_problems = st.sidebar.slider(
        "Max Problems to Display",
        min_value=10,
        max_value=500,
        value=DEFAULT_SETTINGS["max_problems_display"],
        help="Limit the number of problems shown in tables"
    )
    
    # Main content
    st.header("📈 Dataset Overview")
    
    # Summary statistics
    summary = st.session_state.analyzer.create_summary_statistics(problems)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Problems", summary['total_problems'])
    
    with col2:
        st.metric("Total Models", summary['total_models'])
    
    with col3:
        st.metric("Unique Domains", len(summary['domain_distribution']))
    
    with col4:
        st.metric("Difficulty Levels", len(summary['difficulty_distribution']))
    
    # Difficulty and domain distribution (optional)
    if show_pie_charts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Difficulty Distribution")
            if summary['difficulty_distribution']:
                fig = px.pie(
                    values=list(summary['difficulty_distribution'].values()),
                    names=list(summary['difficulty_distribution'].keys()),
                    title="Problems by Difficulty Level"
                )
                st.plotly_chart(fig, config={'displayModeBar': False, 'responsive': True})
        
        with col2:
            st.subheader("🔬 Domain Distribution")
            if summary['domain_distribution']:
                fig = px.pie(
                    values=list(summary['domain_distribution'].values()),
                    names=list(summary['domain_distribution'].keys()),
                    title="Problems by Domain"
                )
                st.plotly_chart(fig, config={'displayModeBar': False, 'responsive': True})
    
    # Model performance analysis
    st.header("🎯 Model Performance Analysis")
    
    verification_metrics = st.session_state.analyzer.calculate_verification_metrics(problems)
    best_of_n_metrics = st.session_state.analyzer.calculate_best_of_n_accuracy(problems, best_of_n)
    if not verification_metrics:
        st.info("These educational fixtures are ungraded. Run python demo.py and select Demo results.")
        return
    
    # Create verification metrics table
    metrics_data = []
    for model, stats in verification_metrics.items():
        best_of_n_stats = best_of_n_metrics.get(model, {'best_of_n_accuracy': 0, 'solved_problems': 0, 'total_problems': 0})
        metrics_data.append({
            'Model': model,
            'Correct': stats['correct'],
            'Incorrect': stats['incorrect'],
            'Unknown': stats['unknown'],
            'Success Rate %': stats['success_rate'],
            'Best of N Accuracy': best_of_n_stats['best_of_n_accuracy'],
            'Problems Solved': f"{best_of_n_stats['solved_problems']}/{best_of_n_stats['total_problems']}",
            'Total Attempts': stats['total_attempts']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    # Scale rates to percentage points for display/sorting in tables
    metrics_df['Success Rate %'] = metrics_df['Success Rate %'] * 100.0
    metrics_df['Best of N Accuracy'] = metrics_df['Best of N Accuracy'] * 100.0
    # Sort by problems solved (extract numeric value for sorting)
    metrics_df['_sort_solved'] = metrics_df['Problems Solved'].str.split('/').str[0].astype(int)
    metrics_df = metrics_df.sort_values('_sort_solved', ascending=False)
    metrics_df = metrics_df.drop('_sort_solved', axis=1)
    
    st.subheader("📋 Verification Results Summary")
    st.dataframe(
        metrics_df,
        width='stretch',
        column_config={
            'Success Rate %': st.column_config.NumberColumn(format="%.2f%%"),
            'Best of N Accuracy': st.column_config.NumberColumn(format="%.2f%%"),
        }
    )
    
    # Verification success rate chart
    # Create a copy with numeric values for plotting
    chart_df = metrics_df.copy()
    chart_df['Success Rate Numeric'] = chart_df['Success Rate %'] / 100.0
    chart_df['Success Rate Text'] = chart_df['Success Rate Numeric'].map(lambda v: f"{v:.2%}")
    
    fig = px.bar(
        chart_df,
        x='Model',
        y='Success Rate Numeric',
        title="Verification Success Rate by Model",
        text='Success Rate Text',
        color='Success Rate Numeric',
        color_continuous_scale='RdYlGn'
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Success Rate")
    st.plotly_chart(fig, config={'displayModeBar': False, 'responsive': True})
    
    # Model ranking
    st.subheader("🏆 Model Rankings")
    ranking_df = metrics_df[['Model', 'Success Rate %', 'Best of N Accuracy', 'Problems Solved', 'Total Attempts']].copy()
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[['Rank', 'Model', 'Success Rate %', 'Best of N Accuracy', 'Problems Solved', 'Total Attempts']]
    st.dataframe(
        ranking_df,
        width='stretch',
        column_config={
            'Success Rate %': st.column_config.NumberColumn(format="%.2f%%"),
            'Best of N Accuracy': st.column_config.NumberColumn(format="%.2f%%"),
        }
    )
    
    # Problem-level analysis (merged from Detailed Analysis)
    st.subheader("📋 Problem-Level Analysis")
    
    # Show problems with their metadata
    problem_details = []
    for problem in problems:
        # Check if problem is solved by at least one model
        solved_by_any = False
        for solution in problem.get('model_solutions', []):
            for attempt in solution.get('attempts', []):
                if 'verifier_result' not in attempt:
                    continue
                for result in _verification_results(attempt):
                    if result.get('verified') == 1:
                        solved_by_any = True
                        break
                if solved_by_any:
                    break
            if solved_by_any:
                break
        
        problem_details.append({
            'Problem ID': problem.get('problem_id', 'Unknown'),
            'Difficulty': problem.get('problem_metadata', {}).get('Difficulty level', 'Unknown'),
            'Domain': problem.get('problem_metadata', {}).get('Domain of theoretical physics', 'Unknown'),
            'Solved By At Least One Model': '✅ SOLVED' if solved_by_any else '❌ UNSOLVED',
            'Models': len(problem.get('model_solutions', [])),
            'Total Attempts': sum(len(sol.get('attempts', [])) for sol in problem.get('model_solutions', []))
        })
    
    problem_df = pd.DataFrame(problem_details)
    
    # Add filtering options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        filter_option = st.selectbox(
            "Filter by Solution Status:",
            ["All", "Solved Only", "Unsolved Only"],
            help="Filter problems based on whether they are solved by at least one model"
        )
        
        # Add status legend
        st.markdown("**Legend:**")
        st.markdown("✅ **SOLVED** - Solved by at least one model")
        st.markdown("❌ **UNSOLVED** - Not solved by any model")
    
    with col2:
        st.metric("Total Problems", len(problem_df))
    
    with col3:
        solved_count = len(problem_df[problem_df['Solved By At Least One Model'] == '✅ SOLVED'])
        unsolved_count = len(problem_df[problem_df['Solved By At Least One Model'] == '❌ UNSOLVED'])
        st.metric("Solved/Unsolved", f"{solved_count}/{unsolved_count}")
    
    # Apply filtering
    if filter_option == "Solved Only":
        filtered_df = problem_df[problem_df['Solved By At Least One Model'] == '✅ SOLVED']
    elif filter_option == "Unsolved Only":
        filtered_df = problem_df[problem_df['Solved By At Least One Model'] == '❌ UNSOLVED']
    else:
        filtered_df = problem_df
    
    st.dataframe(filtered_df, width='stretch')
    
    # Problem-Model Performance Matrix
    st.header("🔍 Problem-Model Performance Matrix")
    
    matrix_df = st.session_state.analyzer.create_problem_model_matrix(problems, metric_type)
    
    # Sort by difficulty and problem ID
    matrix_df = matrix_df.sort_values(['Difficulty', 'Problem ID'])
    
    # Limit display if too many problems
    if len(matrix_df) > max_problems:
        st.warning(f"Showing first {max_problems} problems out of {len(matrix_df)} total. Use the slider to adjust.")
        matrix_df_display = matrix_df.head(max_problems)
    else:
        matrix_df_display = matrix_df
    
    st.subheader(f"📊 {metric_type.title()} Performance by Problem and Model")
    
    # Display the matrix
    st.dataframe(
        matrix_df_display,
        width='stretch',
        height=400
    )
    
    # Download button for the matrix
    csv = matrix_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Matrix as CSV",
        data=csv,
        file_name=f"problem_model_matrix_{metric_type}.csv",
        mime="text/csv"
    )
    

    # Model Comparison Analysis
    st.header("⚖️ Head-to-Head Model Comparison Analysis")
    
    # Get available models from the matrix
    available_models = [col for col in matrix_df.columns if col not in ['Problem ID', 'Difficulty', 'Domain']]
    
    if len(available_models) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_model = st.selectbox(
                "Select Baseline Model:",
                available_models,
                help="Choose the first model for comparison"
            )
        
        with col2:
            comparison_models = [model for model in available_models if model != baseline_model]
            comparison_model = st.selectbox(
                "Select Comparison Model:",
                comparison_models,
                help="Choose the second model for comparison"
            )
        
        # Create comparison matrix
        comparison_df = matrix_df[['Problem ID', 'Difficulty', 'Domain', baseline_model, comparison_model]].copy()
        
        # Convert success rates to numeric for analysis
        def parse_success_rate(rate_str):
            if rate_str == "N/A":
                return 0.0
            try:
                return float(rate_str)
            except:
                return 0.0
        
        comparison_df[f'{baseline_model}_numeric'] = comparison_df[baseline_model].apply(parse_success_rate)
        comparison_df[f'{comparison_model}_numeric'] = comparison_df[comparison_model].apply(parse_success_rate)
        
        # Determine solve status (success rate > 0 means solved)
        comparison_df[f'{baseline_model}_solved'] = comparison_df[f'{baseline_model}_numeric'] > 0
        comparison_df[f'{comparison_model}_solved'] = comparison_df[f'{comparison_model}_numeric'] > 0
        
         # Detailed Analysis
        st.subheader("📊 Detailed Comparison Analysis")
        
        # Calculate metrics for both models
        baseline_stats = verification_metrics.get(baseline_model, {})
        comparison_stats = verification_metrics.get(comparison_model, {})
        baseline_best_of_n = best_of_n_metrics.get(baseline_model, {})
        comparison_best_of_n = best_of_n_metrics.get(comparison_model, {})
        
        # Calculate problem-level comparisons
        both_solved = len(comparison_df[
            (comparison_df[f'{baseline_model}_solved'] == True) & 
            (comparison_df[f'{comparison_model}_solved'] == True)
        ])
        
        baseline_only = len(comparison_df[
            (comparison_df[f'{baseline_model}_solved'] == True) & 
            (comparison_df[f'{comparison_model}_solved'] == False)
        ])
        
        comparison_only = len(comparison_df[
            (comparison_df[f'{baseline_model}_solved'] == False) & 
            (comparison_df[f'{comparison_model}_solved'] == True)
        ])
        
        neither_solved = len(comparison_df[
            (comparison_df[f'{baseline_model}_solved'] == False) & 
            (comparison_df[f'{comparison_model}_solved'] == False)
        ])
        
        # Key metrics summary
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                f"{baseline_model} Success Rate",
                f"{baseline_stats.get('success_rate', 0):.2%}",
                delta=f"{baseline_stats.get('success_rate', 0) - comparison_stats.get('success_rate', 0):.2%}" if baseline_stats.get('success_rate', 0) > comparison_stats.get('success_rate', 0) else None
            )
        
        with col2:
            st.metric(
                f"{comparison_model} Success Rate",
                f"{comparison_stats.get('success_rate', 0):.2%}",
                delta=f"{comparison_stats.get('success_rate', 0) - baseline_stats.get('success_rate', 0):.2%}" if comparison_stats.get('success_rate', 0) > baseline_stats.get('success_rate', 0) else None
            )
        
        with col3:
            st.metric(
                f"{baseline_model} Best of {best_of_n}",
                f"{baseline_best_of_n.get('best_of_n_accuracy', 0):.2%}",
                delta=f"{baseline_best_of_n.get('best_of_n_accuracy', 0) - comparison_best_of_n.get('best_of_n_accuracy', 0):.2%}" if baseline_best_of_n.get('best_of_n_accuracy', 0) > comparison_best_of_n.get('best_of_n_accuracy', 0) else None
            )
        
        with col4:
            st.metric(
                f"{comparison_model} Best of {best_of_n}",
                f"{comparison_best_of_n.get('best_of_n_accuracy', 0):.2%}",
                delta=f"{comparison_best_of_n.get('best_of_n_accuracy', 0) - baseline_best_of_n.get('best_of_n_accuracy', 0):.2%}" if comparison_best_of_n.get('best_of_n_accuracy', 0) > baseline_best_of_n.get('best_of_n_accuracy', 0) else None
            )
        
        with col5:
            st.metric(
                "Problems Both Solved",
                both_solved
            )
        
        with col6:
            total_solved = both_solved + baseline_only + comparison_only
            st.metric(
                "Total Problems Solved",
                total_solved
            )
        
        
        # Detailed breakdown table with filtering options
        st.subheader("📋 Problem-by-Problem Comparison")
        
        # Add filtering options
        st.subheader("🔍 Filtering Options")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            baseline_filter = st.selectbox(
                f"Filter by {baseline_model} status:",
                ["All", "Solved", "Unsolved"],
                help=f"Filter problems based on whether {baseline_model} solved them"
            )
        
        with filter_col2:
            comparison_filter = st.selectbox(
                f"Filter by {comparison_model} status:",
                ["All", "Solved", "Unsolved"],
                help=f"Filter problems based on whether {comparison_model} solved them"
            )
        
        # Apply filters to comparison_df
        filtered_comparison = comparison_df.copy()
        
        if baseline_filter == "Solved":
            filtered_comparison = filtered_comparison[filtered_comparison[f'{baseline_model}_solved'] == True]
        elif baseline_filter == "Unsolved":
            filtered_comparison = filtered_comparison[filtered_comparison[f'{baseline_model}_solved'] == False]
        
        if comparison_filter == "Solved":
            filtered_comparison = filtered_comparison[filtered_comparison[f'{comparison_model}_solved'] == True]
        elif comparison_filter == "Unsolved":
            filtered_comparison = filtered_comparison[filtered_comparison[f'{comparison_model}_solved'] == False]
        
        comparison_breakdown = []
        for _, row in filtered_comparison.iterrows():
            baseline_solved = row[f'{baseline_model}_solved']
            comparison_solved = row[f'{comparison_model}_solved']
            
            if baseline_solved and comparison_solved:
                status = "✅ Both Solved"
                status_color = "🟢"
            elif baseline_solved and not comparison_solved:
                status = f"🔵 Only {baseline_model}"
                status_color = "🔵"
            elif not baseline_solved and comparison_solved:
                status = f"🟠 Only {comparison_model}"
                status_color = "🟠"
            else:
                status = "❌ Neither Solved"
                status_color = "🔴"
            
            comparison_breakdown.append({
                'Problem ID': row['Problem ID'],
                'Difficulty': row['Difficulty'],
                'Domain': row['Domain'],
                f'{baseline_model}': row[baseline_model],
                f'{comparison_model}': row[comparison_model],
                'Status': status,
                'Status Icon': status_color
            })
        
        breakdown_df = pd.DataFrame(comparison_breakdown)
        
        # Check if we have any data to display
        if len(breakdown_df) == 0:
            st.info("🔍 No problems match the selected filters. Please adjust your filter criteria.")
        else:
            # Add summary statistics above the table
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                both_count = len(breakdown_df[breakdown_df['Status'] == "✅ Both Solved"])
                st.metric("Both Solved", both_count)
            
            with col2:
                baseline_only_count = len(breakdown_df[breakdown_df['Status'].str.contains(f"Only {baseline_model}")])
                st.metric(f"{baseline_model} Only", baseline_only_count, delta=f"{baseline_only_count/len(breakdown_df):.1%}")
            
            with col3:
                comparison_only_count = len(breakdown_df[breakdown_df['Status'].str.contains(f"Only {comparison_model}")])
                st.metric(f"{comparison_model} Only", comparison_only_count, delta=f"{comparison_only_count/len(breakdown_df):.1%}")
            
            with col4:
                neither_count = len(breakdown_df[breakdown_df['Status'] == "❌ Neither Solved"])
                st.metric("Neither Solved", neither_count)
            
            # Display the table with better formatting
            st.dataframe(
                breakdown_df[['Problem ID', 'Difficulty', 'Domain', baseline_model, comparison_model, 'Status']], 
                width='stretch',
                height=400
            )
            
            # Add download functionality
            csv = breakdown_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Results as CSV",
                data=csv,
                file_name=f"model_comparison_{baseline_model}_vs_{comparison_model}.csv",
                mime="text/csv"
            )

            # Attempt-level viewer
            st.subheader("🧪 Attempt-level Viewer")

            # Choose a problem to inspect
            # Build problem labels with status appended
            problem_options = []  # list of (label, problem_id)
            for _, prow in filtered_comparison.iterrows():
                pid = str(prow['Problem ID'])
                b_ok = bool(prow.get(f'{baseline_model}_solved', False))
                c_ok = bool(prow.get(f'{comparison_model}_solved', False))
                if b_ok and c_ok:
                    status_txt = "✅ Both Solved"
                elif b_ok and not c_ok:
                    status_txt = f"🔵 Only {baseline_model}"
                elif (not b_ok) and c_ok:
                    status_txt = f"🟠 Only {comparison_model}"
                else:
                    status_txt = "❌ Neither Solved"
                problem_options.append((f"{pid} — {status_txt}", pid))

            if len(problem_options) == 0:
                st.info("No problems available for attempt-level viewing with current filters.")
            else:
                selected_problem_label = st.selectbox(
                    "Select a problem to inspect attempts:",
                    [lbl for (lbl, _) in problem_options],
                    key=f"problem_select_{baseline_model}_{comparison_model}"
                )
                selected_problem_id_str = next((pid for (lbl, pid) in problem_options if lbl == selected_problem_label), None)

                # Locate the full problem object by id (handle id as str/int robustly)
                selected_problem = None
                for p in problems:
                    pid = str(p.get('problem_id', ''))
                    if pid == selected_problem_id_str:
                        selected_problem = p
                        break

                if selected_problem is None:
                    st.warning("Selected problem not found in loaded data.")
                else:
                    # Show problem details
                    st.subheader("📝 Problem Details")

                    details = selected_problem.get('problem_details', {}) or {}
                    problem_statement = details.get('Problem Statement')
                    answer_requirements = details.get('Answer Requirements')
                    solution_text = details.get('Solution')
                    code_text = selected_problem.get('Code') or details.get('Code')

                    if problem_statement:
                        st.markdown("**Problem Statement**")
                        _render_detailed_solution_streamlit(problem_statement)

                    if answer_requirements:
                        st.markdown("**Answer Requirements**")
                        _render_detailed_solution_streamlit(answer_requirements)

                    if solution_text:
                        st.markdown("**Solution**")
                        _render_detailed_solution_streamlit(solution_text)

                    if code_text:
                        st.markdown("**Code**")
                        # Extract python blocks from \begin{python}...\end{python}; fallback to renderer
                        code_blocks = re.findall(r"\\begin\{python\}(.*?)\\end\{python\}", code_text or "", flags=re.DOTALL)
                        if code_blocks:
                            for block in code_blocks:
                                st.code(block.strip(), language="python")
                        else:
                            _render_detailed_solution_streamlit(code_text)

                    # Fetch attempts for baseline and comparison models
                    def get_attempts_for_model(problem_obj: dict, model_name: str) -> List[dict]:
                        for sol in problem_obj.get('model_solutions', []) or []:
                            if sol.get('model') == model_name:
                                return sol.get('attempts', []) or []
                        return []

                    baseline_attempts = get_attempts_for_model(selected_problem, baseline_model)
                    comparison_attempts = get_attempts_for_model(selected_problem, comparison_model)

                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown(f"### {baseline_model}")
                        if not baseline_attempts:
                            st.info("No attempts found for this model.")
                            baseline_selected_idx = None
                        else:
                            options_left = []
                            for idx, att in enumerate(baseline_attempts):
                                status_text, emoji = _attempt_status(att)
                                label = f"Attempt {idx+1} | {emoji} {status_text} | {att.get('timestamp','')}"
                                options_left.append((label, idx))
                            baseline_selected_label = st.selectbox(
                                "Choose an attempt:",
                                [lbl for (lbl, _) in options_left],
                                key=f"baseline_attempt_select_{selected_problem_id_str}_{baseline_model}"
                            )
                            baseline_selected_idx = next(
                                (i for (lbl, i) in options_left if lbl == baseline_selected_label),
                                0
                            )

                    with c2:
                        st.markdown(f"### {comparison_model}")
                        if not comparison_attempts:
                            st.info("No attempts found for this model.")
                            comparison_selected_idx = None
                        else:
                            options_right = []
                            for idx, att in enumerate(comparison_attempts):
                                status_text, emoji = _attempt_status(att)
                                label = f"Attempt {idx+1} | {emoji} {status_text} | {att.get('timestamp','')}"
                                options_right.append((label, idx))
                            comparison_selected_label = st.selectbox(
                                "Choose an attempt:",
                                [lbl for (lbl, _) in options_right],
                                key=f"comparison_attempt_select_{selected_problem_id_str}_{comparison_model}"
                            )
                            comparison_selected_idx = next(
                                (i for (lbl, i) in options_right if lbl == comparison_selected_label),
                                0
                            )

                    # Render side-by-side solutions
                    r1, r2 = st.columns(2)

                    with r1:
                        if baseline_selected_idx is not None and 0 <= baseline_selected_idx < len(baseline_attempts):
                            att = baseline_attempts[baseline_selected_idx]
                            status_text, emoji = _attempt_status(att)
                            st.markdown(f"**Status:** {emoji} {status_text}")
                            _render_detailed_solution_streamlit(att.get('detailed_solution', ''))
                        else:
                            st.write("")

                    with r2:
                        if comparison_selected_idx is not None and 0 <= comparison_selected_idx < len(comparison_attempts):
                            att = comparison_attempts[comparison_selected_idx]
                            status_text, emoji = _attempt_status(att)
                            st.markdown(f"**Status:** {emoji} {status_text}")
                            _render_detailed_solution_streamlit(att.get('detailed_solution', ''))
                        else:
                            st.write("")
        
    else:
        st.warning("Need at least 2 models to perform comparison analysis.")


if __name__ == "__main__":
    main()
