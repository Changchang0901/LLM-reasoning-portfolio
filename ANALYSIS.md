# Collaborative reasoning-analysis source

The source in `rollout_analysis/` implements four stages:

1. Split a reference solution into steps.
2. Distill a candidate response into steps and remove redundant material.
3. Match candidate steps to reference steps and summarize coverage/validity.
4. Classify reasoning errors into factual, mathematical, logical, or executional
   categories, with severity and a primary category.

Xu's emphasized participation and learning were in error classification, response
feature analysis, and the Streamlit dashboard. Other stages provide collaborative
context and are not presented as individually authored modules.

Generic analysis prompt templates are included; filled prompts, model annotations,
and research outputs are excluded. API-based analysis requires
`requirements-analysis.txt`, provider credentials, and the appropriate stage inputs.
Stage CLI help is available with `python rollout_analysis/stage1_solution_splitter.py --help`
(and likewise for stages 2 and 3). The classifier is used programmatically by the
research pipeline; a private experiment-specific batch runner is excluded.

LLM-based annotations are hypotheses to inspect with subject-matter review. A passed
code test establishes agreement on the selected test inputs, not a proof that every
reasoning step is valid. This local package does not report new research findings.
