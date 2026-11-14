# Demo walkthrough

## What is demonstrated

1. `scripts/convert_demo.py` uses the research parser on a newly written LaTeX fixture
   and exports local JSON and JSONL. This conversion example has no test cases yet;
   the ready-to-grade fixtures are supplied separately in `data/demo/`.
2. `demo.py` runs the actual code-verification grader on two elementary problems.
3. Each problem has one hand-written correct candidate and one intentionally wrong
   candidate. A missing factor of one-half and a missing square root are the two
   controlled mistakes. These are illustrative, not a statistical model evaluation.
4. The grader executes trusted reference code to obtain expected outputs, executes
   each candidate in a timeout-controlled subprocess, and compares the outputs.
5. Deterministic text features and graded problem copies are saved locally.
6. The Streamlit dashboard can inspect the resulting problem/candidate matrix.

Expected result: four checks, two accepted and two rejected, matching the four
hand-written candidates. Token counts are `null` when unavailable; the offline
demo disables optional tokenizer downloads.

## Optional LaTeX report

After `python demo.py`, generate report source with:

```bash
python main/generate_latex.py --problems_dir output/demo/problems --output_name educational_demo
```

Add `--pdf` if a working `pdflatex` installation is available. Generated reports
stay in the ignored `output/` directory. This preserves selected original reporting
utilities; the primary review route is the offline demo and interactive dashboard.
