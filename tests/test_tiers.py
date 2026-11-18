"""Phase F checks: tier labels follow the ranks, ECR tiers only where allowed, and saved scores match the file."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
KEY = ["gsis_id", "season", "week"]
GROUP = ["season", "week", "position"]
TIER_COLUMNS = ["tier_actual", "tier_from_regression", "tier_classifier", "tier_ecr"]


@pytest.fixture(scope="module")
def tiers():
    return pd.read_parquet(PROCESSED / "tier_results.parquet")


@pytest.fixture(scope="module")
def choice():
    return json.loads((PROCESSED / "tier_choice.json").read_text())


def assert_tiers_follow_score(rows, tier, score):
    "Tiers 1 and 2 hold 12 players each (fewer only if the position-week is small), and a higher tier never scores lower."
    for _, week in rows.groupby(GROUP):
        counts = week[tier].value_counts()
        n = len(week)
        assert counts.get(1, 0) == min(12, n) and counts.get(2, 0) == min(12, max(n - 12, 0))
        lowest, highest = week.groupby(tier)[score].min(), week.groupby(tier)[score].max()
        for better, worse in [(1, 2), (2, 3)]:
            if better in lowest and worse in highest:
                assert lowest[better] >= highest[worse]


def test_rows_and_columns(tiers):
    features = pd.read_parquet(PROCESSED / "features.parquet")
    cv_rows = features[features["season"].isin([2022, 2023, 2024])]
    assert list(tiers.columns) == KEY + ["position"] + TIER_COLUMNS
    assert not tiers.duplicated(KEY).any()
    assert set(map(tuple, tiers[KEY].to_numpy())) == set(map(tuple, cv_rows[KEY].to_numpy()))
    for column in TIER_COLUMNS:
        assert tiers[column].dropna().isin([1, 2, 3]).all()
    assert tiers[TIER_COLUMNS[:3]].notna().all().all()


def test_actual_and_regression_tiers_follow_ranks(tiers):
    features = pd.read_parquet(PROCESSED / "features.parquet")
    preds = pd.read_parquet(PROCESSED / "preds_regression.parquet")
    rows = tiers.merge(features[KEY + ["fantasy_points_ppr"]], on=KEY).merge(preds[KEY + ["pred"]], on=KEY)
    assert len(rows) == len(tiers)
    assert_tiers_follow_score(rows, "tier_actual", "fantasy_points_ppr")
    assert_tiers_follow_score(rows, "tier_from_regression", "pred")


def test_ecr_tiers_only_where_expert_rank_came_before_kickoff(tiers):
    ecr = pd.read_parquet(PROCESSED / "ecr_weekly.parquet")
    usable = ecr.loc[~ecr["played_before_snapshot"], KEY + ["ecr_pos_rank"]]
    rows = tiers.merge(usable, on=KEY, how="left")
    np.testing.assert_array_equal(rows["tier_ecr"].notna(), rows["ecr_pos_rank"].notna())
    matched = rows.dropna(subset=["tier_ecr"]).assign(ecr_score=lambda d: -d["ecr_pos_rank"])
    assert_tiers_follow_score(matched, "tier_ecr", "ecr_score")


def test_saved_macro_f1_matches_file(tiers, choice):
    def position_week_macro_f1(column):
        return tiers.groupby(GROUP).apply(lambda w: f1_score(w["tier_actual"], w[column], labels=[1, 2, 3],
                                                             average="macro", zero_division=0)).mean()

    assert choice["macro_f1"]["cut from regression"] == pytest.approx(position_week_macro_f1("tier_from_regression"))
    assert choice["macro_f1"]["classifier (raw labels)"] == pytest.approx(position_week_macro_f1("tier_classifier"))


def test_choice_settings(choice):
    assert choice["tier_cuts"] == [12, 24]
    assert choice["refit"] == "every week"
    assert choice["model"] in {"Logistic regression", "Random forest", "Gradient boosting"}
    assert choice["class_weight"] in {None, "balanced"}
