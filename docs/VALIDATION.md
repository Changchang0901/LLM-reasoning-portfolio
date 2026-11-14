# Local validation

Prepared on November 9, 2026. Validated on Windows with CPython 3.12.14.
The direct dependency versions are recorded in `requirements-tested.txt`.

| Check | Outcome |
| --- | --- |
| Offline demo | Two fixtures, four candidate answers; all expected pass/fail outcomes matched |
| LaTeX data conversion | New fixture converted into JSON and local JSONL |
| Core regression suite | 9 tests passed: candidate checks, timeout rejection, resume logic, JSON I/O, error normalization, missing token counts, and credential-variable removal |
| Streamlit app | Fixture results and percentages checked; comparison tables, legacy-result compatibility, and ungraded-data guidance passed |
| LaTeX report utility | Educational report source generated successfully |
| Source and package audit | Syntax checks, relative documentation links, strict file allowlist, secret/path checks, and ZIP/manifest integrity verified |
| Independent archive check | ZIP extracted into a separate folder; demo, conversion, 9 core tests, and dashboard regression passed there |

Optional dashboard regression command, after installing dashboard extras and running the demo:

```bash
python tests/dashboard_smoke.py
```

Live provider requests, optional LLM-driven analysis, GPU inference, and Linux/macOS
execution were not exercised in this local validation. Report source generation was
checked; compiling its optional PDF depends on a separate LaTeX installation.
The demo outcomes are fixture checks, not research benchmark measurements.
