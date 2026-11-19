"""Phase G checks: temporal scope, switches, and independently recomputed CV metrics."""
import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'
KEY = ['gsis_id', 'season', 'week']


def profile_class():
    notebook = nbformat.read(ROOT / 'fantasy_football_rankings.ipynb', 4)
    source = next(c.source for c in notebook.cells if c.cell_type == 'code' and 'class UsageProfiles' in c.source)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'UsageProfiles')
    scope = dict(np=np, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin,
                 make_pipeline=make_pipeline, StandardScaler=StandardScaler, KMeans=KMeans)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<UsageProfiles>', 'exec'), scope)
    return scope['UsageProfiles']


def test_transform_does_not_refit_or_use_targets():
    cls = profile_class()
    x = pd.DataFrame({'usage': [0., .1, .2, 9., 9.1, 9.2],
                      'previous_usage': [0., 0., .1, .2, 9., 9.1],
                      'has_previous': [0, 1, 1, 1, 1, 1]})
    fitted = cls(['usage'], ['usage'], 2, 363).fit(x, np.arange(6))
    other = cls(['usage'], ['usage'], 2, 363).fit(x, np.arange(6)[::-1])
    np.testing.assert_array_equal(fitted.transform(x), other.transform(x))
    centers = fitted.profiles_[-1].cluster_centers_.copy()
    # Transforming future extremes must not change the fit or earlier outputs.
    earlier = fitted.transform(x.iloc[:3])
    future = x.iloc[3:].copy()
    future['usage'] += 10000
    fitted.transform(future)
    np.testing.assert_array_equal(centers, fitted.profiles_[-1].cluster_centers_)
    np.testing.assert_array_equal(earlier, fitted.transform(x.iloc[:3]))
    result = fitted.transform(x)
    np.testing.assert_array_equal(result[:, 1:3].sum(axis=1), np.ones(6))
    np.testing.assert_array_equal(result[:, -1], [0, 0, 0, 1, 0, 0])
    # A row's result must not depend on which other rows share the prediction batch.
    np.testing.assert_array_equal(result[3], fitted.transform(x.iloc[[3]])[0])


def test_export_has_training_rows_and_consistent_switches():
    rows = pd.read_parquet(PROCESSED / 'usage_clusters.parquet').sort_values(KEY)
    features = pd.read_parquet(PROCESSED / 'features.parquet')
    expected = features[features.season.between(2020, 2024)]
    assert list(rows.columns) == KEY + ['cluster', 'switched_cluster']
    assert not rows.duplicated(KEY).any()
    assert set(map(tuple, rows[KEY].to_numpy())) == set(map(tuple, expected[KEY].to_numpy()))
    previous = rows.groupby('gsis_id').cluster.shift(1)
    np.testing.assert_array_equal(rows.switched_cluster, previous.notna() & rows.cluster.ne(previous))
    choice = json.loads((PROCESSED / 'usage_cluster_choice.json').read_text())
    assert rows.cluster.between(0, choice['k'] - 1).all()
    assert choice['selection_seasons'] == [2020, 2021]
    assert choice['evaluation_seasons'] == [2022, 2023, 2024]
    assert choice['changes_frozen_regression'] is False
    families = json.loads((PROCESSED / 'feature_families.json').read_text())
    assert set(choice['usage_columns']) <= set(families['usage'])
    assert all(c.endswith(('_last3', '_recent')) for c in choice['usage_columns'])


def test_saved_cluster_cv_metrics_match_predictions():
    rows = pd.read_parquet(PROCESSED / 'usage_cluster_cv_predictions.parquet')
    features = pd.read_parquet(PROCESSED / 'features.parquet')
    expected = features[features.season.isin([2022, 2023, 2024])]
    assert not rows.duplicated(KEY).any()
    assert set(map(tuple, rows[KEY].to_numpy())) == set(map(tuple, expected[KEY].to_numpy()))
    joined = rows.merge(expected[KEY + ['fantasy_points_ppr']], on=KEY, suffixes=('', '_source'), validate='one_to_one')
    np.testing.assert_allclose(joined.fantasy_points_ppr, joined.fantasy_points_ppr_source)
    assert np.isfinite(rows.pred).all()
    table = pd.read_csv(PROCESSED / 'usage_cluster_comparison.csv', index_col=0)
    actual = table.loc['weekly: clusters + switch']
    err = rows.pred - rows.fantasy_points_ppr
    assert actual.rmse == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert actual.mae == pytest.approx(np.mean(abs(err)))
    groups = rows.groupby(['season', 'week', 'position'])
    spearman = groups.apply(lambda w: w.pred.corr(w.fantasy_points_ppr, method='spearman')).mean()
    hit = groups.apply(lambda w: len(set(w.nlargest(12, 'pred').index) & set(w.nlargest(12, 'fantasy_points_ppr').index)) / 12).mean()
    assert actual.spearman == pytest.approx(spearman)
    assert actual.top12_hit_rate == pytest.approx(hit)
