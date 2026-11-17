"""Checks on the Phase H output (2025 walk-forward predictions). Run after the notebook has been executed."""
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
KEY = ["gsis_id", "season", "week"]
TEST_SEASON = 2025


@pytest.fixture(scope="module")
def preds():
    return pd.read_parquet(PROCESSED / "preds_test.parquet")


def test_columns_match_the_contract(preds):
    assert list(preds.columns) == KEY + ["position", "pred", "ecr_pos_rank", "trained_through"]


def test_one_prediction_per_test_player_week(preds):
    features = pd.read_parquet(PROCESSED / "features.parquet", columns=KEY)
    assert set(preds["season"]) == {TEST_SEASON}
    assert not preds.duplicated(KEY).any()
    assert preds["pred"].notna().all()
    assert len(preds) == (features["season"] == TEST_SEASON).sum()


def test_each_week_trained_only_on_earlier_games(preds):
    trained = preds["trained_through"].str.extract(r"(\d{4})-W(\d{2})").astype(int)
    trained_order = trained[0] * 100 + trained[1]
    assert (trained_order < preds["season"] * 100 + preds["week"]).all()
    week_one = preds["week"] == 1
    assert (trained.loc[week_one, 0] == TEST_SEASON - 1).all()
    assert (trained.loc[~week_one, 1] == preds.loc[~week_one, "week"] - 1).all()


def test_expert_ranks_were_made_before_kickoff(preds):
    ecr = pd.read_parquet(PROCESSED / "ecr_weekly.parquet")
    pregame = ecr[~ecr["played_before_snapshot"]].set_index(KEY)["ecr_pos_rank"]
    ranked = preds.dropna(subset=["ecr_pos_rank"]).set_index(KEY)["ecr_pos_rank"]
    assert ranked.index.isin(pregame.index).all()
    assert (ranked == pregame.loc[ranked.index]).all()
