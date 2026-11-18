"""Phase J: the numbers written in the notebook match the saved predictions. Run after the notebook has been executed.

Each test recomputes a headline number from the files in data/processed/ (with the notebook's own section 5.6 metric
code) and checks that the same number, rounded the way the text rounds it, appears in the notebook's markdown.
"""
import ast
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fantasy_football_rankings.ipynb"
PROCESSED = ROOT / "data" / "processed"
KEY = ["gsis_id", "season", "week"]
TARGET = "fantasy_points_ppr"
POSITIONS = ["QB", "RB", "WR", "TE"]


@pytest.fixture(scope="module")
def cells():
    return nbformat.read(NOTEBOOK, as_version=4).cells


@pytest.fixture(scope="module")
def evaluate(cells):
    "The notebook's own evaluate(), so the test checks the numbers and not a second copy of the metric code."
    config = next(c.source for c in cells if c.cell_type == "code" and c.source.startswith("# Config"))
    namespace = {"np": np, "pd": pd}
    for node in ast.parse(config).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {"TRAIN_SEASONS", "CV_VALID_SEASONS", "TARGET", "TOP_N"}:
            namespace[node.targets[0].id] = eval(ast.unparse(node.value), {"list": list, "range": range})
    exec(next(c.source for c in cells if c.cell_type == "code" and c.source.startswith("def cv_folds")), namespace)
    return namespace["evaluate"]


def section_text(cells, heading):
    "All markdown from the heading `heading` up to the next heading at the same or a higher level."
    level = heading.split(" ")[0]
    start = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith(heading))
    text = [cells[start].source]
    for cell in cells[start + 1:]:
        if cell.cell_type == "markdown" and cell.source.startswith("#") and len(cell.source.split(" ")[0]) <= len(level):
            break
        if cell.cell_type == "markdown":
            text.append(cell.source)
    return "\n".join(text)


def assert_stated(text, *phrases):
    for phrase in phrases:
        assert phrase in text, f"notebook text does not say {phrase!r}"


@pytest.fixture(scope="module")
def targets():
    return pd.read_parquet(PROCESSED / "features.parquet", columns=KEY + [TARGET])


@pytest.fixture(scope="module")
def head_to_head(targets):
    "2025 player-weeks with an expert rank made before kickoff, as in section 9.3."
    test = pd.read_parquet(PROCESSED / "preds_test.parquet").merge(targets, on=KEY)
    return test.dropna(subset=["ecr_pos_rank"]).assign(ecr_score=lambda d: -d["ecr_pos_rank"])


def test_cv_scores_of_the_final_model(cells, evaluate, targets):
    rows = pd.read_parquet(PROCESSED / "preds_regression.parquet").merge(targets, on=KEY)
    s = evaluate(rows, rows["pred"])
    assert_stated(section_text(cells, "### 6.7"),
                  f"RMSE {s['rmse']:.3f}, MAE {s['mae']:.3f}, Spearman {s['spearman']:.3f}, top-12 hit rate {s['top12_hit_rate']:.3f}")


def test_2025_point_accuracy(cells, evaluate, targets):
    rows = pd.read_parquet(PROCESSED / "preds_test.parquet").merge(targets, on=KEY)
    s = evaluate(rows, rows["pred"])
    assert_stated(section_text(cells, "### 9.2"),
                  f"RMSE {s['rmse']:.2f}, MAE {s['mae']:.2f}, Spearman {s['spearman']:.3f}, top-12 hit rate {s['top12_hit_rate']:.3f}")


def test_2025_ranking_vs_experts(cells, evaluate, head_to_head):
    ours, experts = evaluate(head_to_head, head_to_head["pred"]), evaluate(head_to_head, head_to_head["ecr_score"])
    text = section_text(cells, "### 9.3")
    assert_stated(text, f"our Spearman is {ours['spearman']:.3f} and the experts' is {experts['spearman']:.3f} "
                        f"on the same {len(head_to_head):,} player-weeks",
                  f"({ours['top12_hit_rate']:.3f} vs. {experts['top12_hit_rate']:.3f})")
    for position in POSITIONS:
        rows = head_to_head[head_to_head["position"] == position]
        assert_stated(text, f"{evaluate(rows, rows['pred'])['spearman']:.3f}", f"{evaluate(rows, rows['ecr_score'])['spearman']:.3f}")


def test_weekly_gap_behind_the_verdict(cells, evaluate, head_to_head):
    gap = pd.Series({week: evaluate(rows, rows["pred"])["spearman"] - evaluate(rows, rows["ecr_score"])["spearman"]
                     for week, rows in head_to_head.groupby("week")})
    assert_stated(section_text(cells, "### 9.3"),
                  f"averages {gap.mean():.3f}", f"ahead in {(gap > 0).sum()} of {len(gap)} weeks")
    assert_stated(section_text(cells, "### 9.6"), "**We lose to the experts")
    assert gap.mean() < 0, "the text says we lose, so the mean weekly gap must be below zero"


def test_disagreement_numbers(cells, head_to_head):
    by_week = head_to_head.groupby(["season", "week", "position"])
    actual = by_week[TARGET].rank(ascending=False)
    rows = head_to_head.assign(our_rank=by_week["pred"].rank(ascending=False), ecr_rank=by_week["ecr_pos_rank"].rank())
    our_error, ecr_error = (rows["our_rank"] - actual).abs(), (rows["ecr_rank"] - actual).abs()
    agreement = by_week.apply(lambda w: w["pred"].corr(-w["ecr_pos_rank"], method="spearman")).mean()
    text = section_text(cells, "### 10.2")
    assert_stated(text, f"correlate at {agreement:.3f}")
    for gap in [6, 12]:
        apart = ((rows["our_rank"] - rows["ecr_rank"]).abs() >= gap) & (rows[["our_rank", "ecr_rank"]].min(axis=1) <= 24)
        assert_stated(text, f"({apart.sum()}", f"{(our_error[apart] < ecr_error[apart]).mean():.1%}",
                      f"{(ecr_error[apart] < our_error[apart]).mean():.1%}")
