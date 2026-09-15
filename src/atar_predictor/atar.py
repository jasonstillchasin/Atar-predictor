"""Aggregate (out of 500) -> ATAR using UAC Table A9 (lowest aggregate for selected ATARs).

Between published points the relationship is close to linear (it reproduces the report's
Table 3.2 to within 0.1). Below ATAR 50 the lowest segment is extended linearly.
ATARs are truncated to steps of 0.05 and capped at 99.95, as UAC does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data.extract import PROCESSED


class AtarConverter:
    def __init__(self, a9: pd.DataFrame):
        self._knots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for year, g in a9.groupby("year"):
            g = g.sort_values("lowest_aggregate")
            self._knots[int(year)] = (g["lowest_aggregate"].to_numpy(float), g["atar"].to_numpy(float))
        self.years = sorted(self._knots)

    @classmethod
    def load(cls, directory: Path = PROCESSED) -> AtarConverter:
        path = directory / "a9_atar_aggregates.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `uv run python -m atar_predictor.data.extract`")
        return cls(pd.read_csv(path))

    def to_atar(self, aggregate: np.ndarray | float, year: int) -> np.ndarray:
        agg_k, atar_k = self._knots[year]
        agg = np.asarray(aggregate, dtype=float)
        atar = np.interp(agg, agg_k, atar_k)
        slope = (atar_k[1] - atar_k[0]) / (agg_k[1] - agg_k[0])
        atar = np.where(agg < agg_k[0], atar_k[0] + slope * (agg - agg_k[0]), atar)
        atar = np.clip(atar, 0.0, 99.95)
        return np.floor(atar * 20 + 1e-6) / 20


def spline_atar(cohort_percentile: float, participation: float) -> float:
    """ATAR from ATAR-cohort percentile via the one-parameter cubic spline (report section 3.2.9).

    The proportion of people at population percentile x who are ATAR-eligible is
    x^3 / (1000a)^2 for x <= 100a, and 1 - (100 - x)^3 / (1000 - 1000a)^2 above, with
    a = 1.5 - 2 * participation. Used only as a cross-check of Table A8, not for prediction.
    """
    a = 1.5 - 2 * participation
    split = 100 * a

    def eligible_below(x: float) -> float:  # integral of the eligibility proportion from 0 to x
        if x <= split:
            return x**4 / (4 * (1000 * a) ** 2)
        upper = ((100 - split) ** 4 - (100 - x) ** 4) / (4 * (1000 - 1000 * a) ** 2)
        return split**4 / (4 * (1000 * a) ** 2) + (x - split) - upper

    total = eligible_below(100.0)
    target = cohort_percentile / 100 * total
    return brentq(lambda x: eligible_below(x) - target, 0.0, 100.0)
