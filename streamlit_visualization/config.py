"""Public demonstration sources only."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
COMMON_PATHS = {
    "Demo results (run python demo.py first)": str(ROOT / "output" / "demo" / "problems"),
    "Ungraded educational fixtures": str(ROOT / "data" / "demo"),
}
DEFAULT_SETTINGS = {"max_problems_display": 100}
CHART_SETTINGS = {"color_scheme": "plotly", "height": 400, "width": "100%"}
