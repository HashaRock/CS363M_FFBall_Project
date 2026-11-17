"""Independent checks on the Phase D feature table. Run after the notebook has been executed."""
import json
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
KEY = ["gsis_id", "season", "week"]


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED / "features.parquet")


@pytest.fixture(scope="module")
def families():
    return json.loads((PROCESSED / "feature_families.json").read_text())


@pytest.fixture(scope="module")
def games():
    g = pd.read_parquet(PROCESSED / "player_games.parquet", columns=KEY + ["team", "fantasy_points_ppr"])
    return g.sort_values(["gsis_id", "season", "week"])


def test_same_rows_as_player_weeks(features):
    weeks = pd.read_parquet(PROCESSED / "player_weeks.parquet", columns=KEY)
    assert not features.duplicated(KEY).any()
    assert len(features) == len(weeks)
    assert features.set_index(KEY).index.isin(weeks.set_index(KEY).index).all()


def test_no_ecr_and_no_missing_features(features, families):
    columns = [c for cols in families.values() for c in cols]
    assert not [c for c in columns if "ecr" in c.lower()]
    assert features[columns].notna().all().all()


def test_target_is_not_a_feature(families):
    columns = [c for cols in families.values() for c in cols]
    assert "fantasy_points_ppr" not in columns and "points" not in columns


def past_games(games, row):
    same = games[games["gsis_id"] == row["gsis_id"]]
    before = (same["season"] < row["season"]) | ((same["season"] == row["season"]) & (same["week"] < row["week"]))
    return same[before]


def test_lags_use_only_earlier_games(features, games):
    for _, row in features.sample(300, random_state=1).iterrows():
        past = past_games(games, row)
        assert row["points_last1"] == pytest.approx(past["fantasy_points_ppr"].iloc[-1])
        assert row["points_last3"] == pytest.approx(past["fantasy_points_ppr"].tail(3).mean())
        assert row["games_this_season"] == (past["season"] == row["season"]).sum()


def test_changed_team_flag(features, games):
    for _, row in features.sample(300, random_state=2).iterrows():
        assert row["changed_team"] == int(past_games(games, row)["team"].iloc[-1] != row["team"])
