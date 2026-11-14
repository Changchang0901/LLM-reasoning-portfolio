# Source provenance and preparation changes

## Collaborative source

The `main/`, generic `file_utils/`, `rollout_analysis/`, response-feature extractor,
Streamlit app, Windows setup helper, and compatibility tests were selected from
the collaborative research repository. Relative paths are preserved for review.
File inclusion does not imply sole authorship by the portfolio owner.

## Portfolio-only changes

- Fresh elementary fixtures and hand-written candidates in `data/demo/` and `examples/`.
- `demo.py`, `scripts/convert_demo.py`, and `tests/test_portfolio.py` make selected
  workflows demonstrable without research data or provider calls.
- The parser exports generic functions only; its private-dataset CLI was removed.
- JSON utilities are limited to load, save, and clear-solution helpers.
- Default input paths use `data/demo`; trailing private scratch commands are removed.
- Credential loading uses environment variables only; no credential file is shipped.
- Generation requires an explicit model identifier; no default paid model is selected.
- Analysis concurrency defaults are reduced for a small demonstration setting.
- Missing token counts are represented as unavailable, rather than zero.
- Dashboard selections point only to demo fixtures or locally generated demo results;
  its caption identifies the data as educational fixtures. Result handling accepts
  both current dict results and legacy list results, and ungraded fixtures show
  guidance instead of an empty-table error.
- Requirements are split into core, provider, analysis, and dashboard extras.
- Reporting attribution identifies reports as educational demonstrations.

`RELEASE_MANIFEST.json` lists the source origin and SHA-256 digest of each packaged
file. It does not include itself. No Git history, original datasets, model responses,
research plots, unpublished results, personal application documents, or model weights
are part of this release.

## Dashboard screenshot gallery (added November 13, 2026)

The four JPEG files in `assets/screenshots/` are actual browser captures from the
original project running on the public-data subset. They show existing aggregate
verification records, not fresh model inference. The overview and comparison were
captured during the earlier local run; distributions and the matrix were captured
on November 13. Model IDs that contain internal filesystem paths were display-aliased.
The gallery includes no question statements, solutions, or reasoning traces.
Screenshots complement the runnable educational demo; the research data itself
is not shipped. The README captions describe both sources separately.
