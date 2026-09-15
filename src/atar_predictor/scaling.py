"""Course percentile -> scaled mark (and HSC mark) curves built from UAC Table A3.

Scaling preserves rank within a course, so the scaled mark at statewide course
percentile p is simply the p-quantile of that course's scaled-mark distribution.
All marks are on a 1-unit basis (0-50).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from .courses import predecessors
from .data.extract import PROCESSED

PUBLISHED = [("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99), ("max", 1.0)]
TAIL_PS = (0.001, 0.02, 0.10)  # below P25 nothing is published; extend the P25-P50 slope in z-space


@dataclass(eq=False)
class MarkCurve:
    course: str
    year: int
    kind: str  # "scaled" or "hsc"
    p: np.ndarray
    v: np.ndarray
    approximate: bool  # True when the course is too small for published percentiles
    _f: PchipInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._f = PchipInterpolator(self.p, self.v)  # monotone data -> monotone curve

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        return np.clip(self._f(np.clip(p, 0.0, 1.0)), 0.0, 50.0)


def _band_knots(bands: pd.Series) -> list[tuple[float, float]]:
    """Table A2 lower-band cumulatives for a 2-unit course, as (percentile, 1-unit HSC mark)."""
    shares = [bands[c] for c in ("pct_b6", "pct_b5", "pct_b4", "pct_b3", "pct_b2")]
    if any(math.isnan(s) for s in shares):
        return []
    above = np.cumsum(shares) / 100  # share at or above bands 6, 5, 4, 3, 2
    cut_marks = [45.0, 40.0, 35.0, 30.0, 25.0]  # 90/80/70/60/50 out of 100
    return [(1 - a, m - 0.25) for a, m in zip(above, cut_marks) if 0 < 1 - a < 0.25]


def _knots(row: pd.Series, extra: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, bool]:
    if any(math.isnan(row[c]) for c, _ in PUBLISHED):
        ps = np.array([0.0, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
        vs = np.clip(row["mean"] + row["sd"] * norm.ppf(np.clip(ps, 0.001, 0.999)), 0.0, row["max"])
        vs[-1] = row["max"]
        return ps, np.maximum.accumulate(vs), True

    points = [(p, row[c]) for c, p in PUBLISHED]
    if extra:
        points += extra + [(0.0, max(0.0, min(v for _, v in extra) - 5.0))]
    else:
        slope = (row["p50"] - row["p25"]) / (norm.ppf(0.5) - norm.ppf(0.25))
        tail = [(tp, max(0.0, row["p25"] + slope * (norm.ppf(tp) - norm.ppf(0.25)))) for tp in TAIL_PS]
        points += tail + [(0.0, tail[0][1])]
    points.sort()
    ps, idx = np.unique([p for p, _ in points], return_index=True)
    vs = np.maximum.accumulate(np.array([v for _, v in points])[idx])
    return ps, vs, False


class ScalingData:
    def __init__(self, a3: pd.DataFrame, a2: pd.DataFrame | None = None):
        self._rows = {(int(r.year), r.course, r.mark_type): r for _, r in a3.iterrows()}
        self._a2 = {} if a2 is None else {(int(r.year), r.course): r for _, r in a2.iterrows()}
        self.years = sorted(int(y) for y in a3["year"].unique())
        self._years = {c: sorted(int(y) for y in g.unique()) for c, g in a3.groupby("course")["year"]}
        self._cache: dict[tuple[str, int, str], MarkCurve] = {}

    @classmethod
    def load(cls, directory: Path = PROCESSED) -> ScalingData:
        a3_path = directory / "a3_mark_stats.csv"
        if not a3_path.exists():
            raise FileNotFoundError(f"{a3_path} missing: run `uv run python -m atar_predictor.data.extract`")
        a2_path = directory / "a2_hsc_bands.csv"
        return cls(pd.read_csv(a3_path), pd.read_csv(a2_path) if a2_path.exists() else None)

    def courses(self) -> set[str]:
        known = set(self._years)
        return known | {new for new, old in predecessors().items() if old in known}

    def courses_in(self, year: int) -> list[str]:
        return sorted(c for c, ys in self._years.items() if year in ys)

    def resolve(self, course: str, year: int) -> tuple[str, int]:
        """Which (course, year) row to use: exact, then predecessor course, then nearest year."""
        own = self._years.get(course, [])
        if year in own:
            return course, year
        pred = predecessors().get(course)
        pred_years = self._years.get(pred, []) if pred else []
        if year in pred_years:
            return pred, year
        if own:
            return course, min(own, key=lambda y: abs(y - year))
        if pred_years:
            return pred, min(pred_years, key=lambda y: abs(y - year))
        raise KeyError(f"No scaling data for course {course!r}")

    def curve(self, course: str, year: int, kind: str = "scaled") -> MarkCurve:
        name, y = self.resolve(course, year)
        key = (name, y, kind)
        if key not in self._cache:
            row = self._rows[(y, name, kind)]
            extra = []
            if kind == "hsc" and "Extension" not in name and (y, name) in self._a2:
                extra = _band_knots(self._a2[(y, name)])
            ps, vs, approx = _knots(row, extra)
            self._cache[key] = MarkCurve(name, y, kind, ps, vs, approx)
        return self._cache[key]
