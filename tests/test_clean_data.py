"""Checks on the Phase B outputs in data/processed/. Run after the notebook has been executed."""
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
KEY = ["gsis_id", "season", "week"]


@pytest.fixture(scope="module")
def player_weeks():
    return pd.read_parquet(PROCESSED / "player_weeks.parquet")


@pytest.fixture(scope="module")
def ecr_weekly():
    return pd.read_parquet(PROCESSED / "ecr_weekly.parquet")


def test_one_row_per_player_week(player_weeks):
    assert not player_weeks.duplicated(KEY).any()


def test_no_missing_targets(player_weeks):
    assert player_weeks["fantasy_points_ppr"].notna().all()


def test_no_ecr_columns_in_modeling_table(player_weeks):
    assert not [c for c in player_weeks.columns if "ecr" in c.lower()]


def test_target_seasons_only(player_weeks):
    assert set(player_weeks["season"]) == set(range(2020, 2026))


def test_pool_rule_uses_only_past_games(player_weeks):
    games = pd.read_parquet(PROCESSED / "player_games.parquet", columns=KEY + ["offense_pct"])
    games = games.sort_values(["season", "week"])
    games["order"] = games["season"] * 100 + games["week"]
    sample = player_weeks.sample(200, random_state=0)
    for _, row in sample.iterrows():
        order = row["season"] * 100 + row["week"]
        past = games[(games["gsis_id"] == row["gsis_id"]) & (games["order"] < order)].tail(3)
        assert past["offense_pct"].fillna(0).mean() == pytest.approx(row["recent_snap_share"])


def test_ecr_rows_belong_to_pool(ecr_weekly, player_weeks):
    assert ecr_weekly.set_index(KEY).index.isin(player_weeks.set_index(KEY).index).all()
