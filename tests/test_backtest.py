"""Guards on the temporal backtest (train on the previous years, predict the next).

Thresholds sit comfortably above what 2023-2025 actually showed (A3 large-course MAE ~0.4-0.5
per unit, A9 ATAR MAE 0.13-0.48, profile drift <= 1.1), so a failure means something broke.
"""

import pandas as pd
import pytest

from atar_predictor.backtest import atar_curve_backtest, scaled_quantile_backtest, year_drift_backtest
from atar_predictor.data.extract import PROCESSED

TESTS = {t: [y for y in range(t - 4, t) if y >= 2020] for t in (2023, 2024, 2025)}
COURSES = ["English Advanced", "Mathematics Advanced", "Physics", "Chemistry", "Economics"]


@pytest.fixture(scope="module")
def tables(scaling):  # scaling fixture skips when data is not extracted
    return pd.read_csv(PROCESSED / "a3_mark_stats.csv"), pd.read_csv(PROCESSED / "a9_atar_aggregates.csv")


@pytest.mark.parametrize("test_year", TESTS)
def test_scaled_quantiles_stable_for_large_courses(tables, test_year):
    a3, _ = tables
    q = scaled_quantile_backtest(a3, TESTS[test_year], test_year)
    large = a3[(a3.year == test_year) & (a3.number >= 1000)].course
    q = q[q.course.isin(large)]
    assert len(q) > 150
    assert q.error.abs().mean() < 0.8


@pytest.mark.parametrize("test_year", TESTS)
def test_atar_curve_stable(tables, test_year):
    _, a9 = tables
    c = atar_curve_backtest(a9, TESTS[test_year], test_year)
    assert c.error.abs().mean() < 1.0
    assert c.error.abs().max() < 2.5


@pytest.mark.parametrize("test_year", TESTS)
def test_profile_level_drift_small(scaling, converter, test_year):
    d = year_drift_backtest(scaling, converter, COURSES, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99], TESTS[test_year], test_year)
    assert d.error.abs().max() < 2.0
