"""Checks on the regression outputs that later phases read. Run after the notebook has been executed."""
import json
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
KEY = ["gsis_id", "season", "week"]
CV_SEASONS = {2022, 2023, 2024}


@pytest.fixture(scope="module")
def preds():
    return pd.read_parquet(PROCESSED / "preds_regression.parquet")


@pytest.fixture(scope="module")
def choice():
    return json.loads((PROCESSED / "regression_choice.json").read_text())


def test_columns_match_the_contract(preds):
    assert list(preds.columns) == KEY + ["position", "pred", "model"]


def test_one_prediction_per_cv_player_week(preds):
    features = pd.read_parquet(PROCESSED / "features.parquet", columns=KEY)
    cv_rows = features[features["season"].isin(CV_SEASONS)]
    assert not preds.duplicated(KEY).any()
    assert preds["pred"].notna().all()
    assert len(preds) == len(cv_rows)


def test_never_touches_the_test_season(preds):
    assert set(preds["season"]) == CV_SEASONS


def test_choice_file_describes_the_predictions(preds, choice):
    assert set(preds["model"]) == {choice["model"]}
    assert choice["status"] in {"baseline", "final"}
    assert not [c for c in choice["features"] if "ecr" in c.lower()]
