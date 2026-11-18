"""Phase J: no 2025 game is used to choose any setting. Run after the notebook has been executed.

Every choice (half-life, model, settings, tier classifier, clusters) is made in sections 4-8. Those sections reach
data in two ways: through the shared cv_folds(), or by fitting on a table filtered to training seasons. So we check
that cv_folds() cannot hand out a 2025 row, that sections 4-8 only use those two routes, that the files they save
hold no 2025 rows, and that section 9 reads the frozen choice instead of tuning anything.
"""
import ast
import json
import re
import tokenize
from io import StringIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fantasy_football_rankings.ipynb"
PROCESSED = ROOT / "data" / "processed"
TRAIN_SEASONS = set(range(2020, 2025))
TEST_SEASON = 2025

# Tables a direct .fit() in sections 4-8 may use. Each is built from training seasons only (checked below).
TRAINING_TABLES = ("data.loc[train", "rows[", "fit_rows[", "train_features", "g_initial[", "X[")


@pytest.fixture(scope="module")
def cells():
    return nbformat.read(NOTEBOOK, as_version=4).cells


def code_between(cells, start, end):
    "Source of the code cells after the heading `start` and before the heading `end`."
    starts = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith(start)]
    ends = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith(end)]
    assert starts and ends, f"headings not found: {start!r}, {end!r}"
    return [c.source for c in cells[starts[0]:ends[0]] if c.cell_type == "code"]


def code_only(source):
    "Source with strings and comments removed, so prose like 'no 2025 labels' is not mistaken for code."
    skip = {tokenize.STRING, tokenize.COMMENT, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    return " ".join(t.string for t in tokenize.generate_tokens(StringIO(source).readline) if t.type not in skip)


def config_values(cells, names):
    "Read settings from the notebook's config cell without running it."
    config = next(c.source for c in cells if c.cell_type == "code" and c.source.startswith("# Config"))
    values = {}
    for node in ast.parse(config).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in names:
            values[node.targets[0].id] = eval(ast.unparse(node.value), {"list": list, "range": range}, values)
    return values


def test_cv_folds_never_hand_out_a_test_season_row(cells):
    shared = next(c.source for c in cells if c.cell_type == "code" and c.source.startswith("def cv_folds"))
    namespace = {"np": np, "pd": pd, **config_values(cells, {"TRAIN_SEASONS", "CV_VALID_SEASONS", "TARGET", "TOP_N"})}
    exec(shared, namespace)
    data = pd.DataFrame({"season": np.repeat(range(2019, TEST_SEASON + 1), 10)})
    for valid_season, train, valid in namespace["cv_folds"](data):
        assert set(data.loc[train, "season"]) <= TRAIN_SEASONS and TEST_SEASON not in set(data.loc[valid, "season"])
        assert data.loc[train, "season"].max() < valid_season, "a fold trains on its own season or later"


def test_choice_sections_only_reach_data_through_training_seasons(cells):
    for source in code_between(cells, "## 4. Explore", "## 9. Final test"):
        code = code_only(source)
        assert "preds_test" not in source and "2025" not in code
        assert "TEST_SEASON" not in code.replace("< TEST_SEASON", ""), "TEST_SEASON used for something other than excluding it"
        for argument in re.findall(r"\.fit\(([^,)]*)", source):
            assert argument.strip().startswith(TRAINING_TABLES), f"fit on a table not known to be training-only: {argument}"


def test_training_tables_are_built_from_training_seasons(cells):
    source = "\n".join(c.source for c in cells if c.cell_type == "code")
    assert 'return table[table["season"].isin(TRAIN_SEASONS)]' in source  # train_rows()
    assert "train_features = train_rows(features)" in source
    assert "tier_rows = train_features." in source
    assert "g_rows = features[features.season.isin(TRAIN_SEASONS)]" in source
    assert "g_initial = g_rows[" in source


@pytest.mark.parametrize("name", ["preds_regression", "tier_results", "usage_clusters", "usage_cluster_cv_predictions"])
def test_saved_choice_outputs_hold_no_test_season_rows(name):
    seasons = set(pd.read_parquet(PROCESSED / f"{name}.parquet", columns=["season"])["season"])
    assert seasons <= TRAIN_SEASONS


def test_cluster_choice_used_training_seasons_only():
    choice = json.loads((PROCESSED / "usage_cluster_choice.json").read_text())
    for key in ["selection_seasons", "evaluation_seasons", "export_seasons"]:
        assert set(choice[key]) <= TRAIN_SEASONS


def test_final_test_reads_the_frozen_choice_and_tunes_nothing(cells):
    after = "\n".join(code_between(cells, "## 9. Final test", "### 10.4"))
    assert 'json.loads((PROCESSED_DIR / "regression_choice.json").read_text())' in after
    assert 'frozen_choice["status"] == "final"' in after
    for tuning in ["cross_validate(", "cross_validate_recency(", "MODEL_GRIDS", "tune_all(", "GridSearchCV"]:
        assert tuning not in after, f"section 9 or 10 tunes something: {tuning}"
