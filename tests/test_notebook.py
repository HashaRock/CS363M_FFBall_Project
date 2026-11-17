"""Structural checks on the deliverable notebook. Run after verify.sh executes it."""
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "fantasy_football_rankings.ipynb"

REQUIRED_SECTIONS = [
    "## 1. Introduction",
    "## 3. Load and clean",
    "## 4. Explore",
    "## 5. Features",
    "## 6. Points regression",
    "## 7. Tiers",
    "## 8. Usage clustering",
    "## 9. Final test vs. experts",
    "## 10. Analysis and wrap-up",
]


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_required_sections_present_in_order():
    headings = [c.source for c in load_notebook().cells if c.cell_type == "markdown"]
    text = "\n".join(headings)
    positions = [text.find(s) for s in REQUIRED_SECTIONS]
    assert all(p >= 0 for p in positions), f"missing: {[s for s, p in zip(REQUIRED_SECTIONS, positions) if p < 0]}"
    assert positions == sorted(positions), "sections out of order"


def test_every_code_cell_ran_without_error():
    for cell in load_notebook().cells:
        if cell.cell_type != "code":
            continue
        assert cell.get("execution_count") is not None, "notebook was not executed"
        assert not any(o.get("output_type") == "error" for o in cell.get("outputs", []))


def test_starter_uses_the_same_evaluation_code():
    """The scratch-notebook starter must carry an exact copy of main notebook section 5.6."""
    def shared_cell(path):
        nb = nbformat.read(path, as_version=4)
        return next(c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("def cv_folds"))

    assert shared_cell(NOTEBOOK) == shared_cell(NOTEBOOK.parent / "notebooks" / "starter.ipynb")
