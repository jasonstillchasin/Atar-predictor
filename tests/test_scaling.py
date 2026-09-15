import numpy as np
import pandas as pd
import pytest

from atar_predictor.data.extract import PROCESSED


def test_curve_passes_through_published_percentiles(scaling):
    a3 = pd.read_csv(PROCESSED / "a3_mark_stats.csv")
    row = a3.query("year == 2025 and course == 'Mathematics Extension 1' and mark_type == 'scaled'").iloc[0]
    curve = scaling.curve("Mathematics Extension 1", 2025)
    for col, p in [("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)]:
        assert float(curve(p)) == pytest.approx(row[col])


def test_all_curves_monotone_and_in_range(scaling):
    ps = np.linspace(0, 1, 401)
    for year in scaling.years:
        for course in scaling.courses_in(year):
            for kind in ("scaled", "hsc"):
                v = scaling.curve(course, year, kind)(ps)
                assert np.all(np.diff(v) >= -1e-9), (year, course, kind)
                assert v.min() >= 0 and v.max() <= 50


def test_small_course_uses_normal_approximation(scaling):
    assert scaling.curve("Croatian Continuers", 2025).approximate
    assert not scaling.curve("Physics", 2025).approximate


def test_new_course_falls_back_to_predecessor(scaling):
    assert scaling.resolve("Software Engineering", 2025) == ("Software Engineering", 2025)
    assert scaling.resolve("Software Engineering", 2022) == ("Software Design & Development", 2022)
    assert "Enterprise Computing" in scaling.courses()
