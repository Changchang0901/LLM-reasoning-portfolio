"""Optional UI regression check. Install dashboard extras and run demo.py first."""
import copy
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "streamlit_visualization"))
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(ROOT / "streamlit_visualization/app.py"), default_timeout=30)
app.run()
assert not app.exception, [e.message for e in app.exception]
app.sidebar.button[1].click().run()
assert not app.exception, [e.message for e in app.exception]
assert any(m.label == "Total Problems" and m.value == "2" for m in app.metric)
table = app.dataframe[0].value.set_index("Model")
assert table.loc["demo-correct", "Success Rate %"] == 100.0
assert table.loc["demo-incorrect", "Success Rate %"] == 0.0
assert len(app.dataframe) >= 4

# Verify the same analyzer accepts current and legacy serialized results.
analyzer = app.session_state.analyzer
problems = app.session_state.problems
legacy = copy.deepcopy(problems)
for problem in legacy:
    for model in problem["model_solutions"]:
        for attempt in model["attempts"]:
            attempt["verifier_result"] = [attempt["verifier_result"]]
assert analyzer.calculate_verification_metrics(legacy) == analyzer.calculate_verification_metrics(problems)
app.sidebar.selectbox[0].select("Ungraded educational fixtures").run()
app.sidebar.button[1].click().run()
assert not app.exception, [e.message for e in app.exception]
assert any("ungraded" in info.value for info in app.info)
print("PASS: fixture results, percentages, comparison tables, legacy format, and ungraded guidance.")
