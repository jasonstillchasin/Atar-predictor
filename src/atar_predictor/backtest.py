"""Temporal backtest: do earlier years' UAC statistics predict a later year?

The simulator samples scaling years uniformly from recent history, so the question is
whether next year's scaled marks and aggregate->ATAR curve land inside that spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import best_ten
from .atar import AtarConverter
from .courses import course_info
from .scaling import ScalingData

QUANTILES = ["p25", "p50", "p75", "p90", "p99"]
Z80 = 1.2816  # two-sided 80% normal interval


def scaled_quantile_backtest(a3: pd.DataFrame, train_years: list[int], test_year: int) -> pd.DataFrame:
    """Per course and published quantile: pooled train-year prediction vs the test year."""
    scaled = a3[a3["mark_type"] == "scaled"]
    test = scaled[scaled["year"] == test_year].set_index("course")
    rows = []
    for course, g in scaled[scaled["year"].isin(train_years)].groupby("course"):
        if course not in test.index:
            continue
        for q in QUANTILES:
            values, actual = g[q].dropna(), test.at[course, q]
            if len(values) < 2 or np.isnan(actual):
                continue
            mean, sd = values.mean(), values.std(ddof=1)
            rows.append({
                "course": course,
                "quantile": q,
                "n_train": len(values),
                "predicted": mean,
                "actual": actual,
                "error": actual - mean,
                "in_train_range": values.min() <= actual <= values.max(),
                "in_normal_80": abs(actual - mean) <= Z80 * sd,
            })
    return pd.DataFrame(rows)


def atar_curve_backtest(
    a9: pd.DataFrame, train_years: list[int], test_year: int, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """ATAR for a grid of aggregates: test-year curve vs the train years' curves."""
    grid = np.arange(170, 481, 5, dtype=float) if grid is None else grid
    actual = AtarConverter(a9[a9["year"] == test_year]).to_atar(grid, test_year)
    train = AtarConverter(a9[a9["year"].isin(train_years)])
    preds = np.stack([train.to_atar(grid, y) for y in train_years])
    return pd.DataFrame({
        "aggregate": grid,
        "actual": actual,
        "pooled": preds.mean(axis=0),
        "low": preds.min(axis=0),
        "high": preds.max(axis=0),
        "error": actual - preds.mean(axis=0),
    })


def year_drift_backtest(
    scaling: ScalingData,
    converter: AtarConverter,
    courses: list[str],
    percentiles: list[float],
    train_years: list[int],
    test_year: int,
) -> pd.DataFrame:
    """A student at the same statewide percentile in every course: ATAR per scaling year.

    Isolates year-to-year scaling drift (no school or rank uncertainty).
    """
    infos = [course_info(c, set(courses)) for c in courses]

    def atar_for(p: float, year: int) -> float:
        scaled = [float(scaling.curve(c, year)(p)) for c in courses]
        return float(converter.to_atar(best_ten(infos, scaled).aggregate, year))

    rows = []
    for p in percentiles:
        train = np.array([atar_for(p, y) for y in train_years])
        actual = atar_for(p, test_year)
        rows.append({
            "percentile": p,
            "actual": actual,
            "pooled": train.mean(),
            "low": train.min(),
            "high": train.max(),
            "error": actual - train.mean(),
            "in_train_range": train.min() <= actual <= train.max(),
        })
    return pd.DataFrame(rows)
