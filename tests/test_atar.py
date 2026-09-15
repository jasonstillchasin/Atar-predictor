import numpy as np
import pandas as pd
import pytest

from atar_predictor.atar import spline_atar
from atar_predictor.data.extract import PROCESSED


def test_spline_model_reproduces_report_numbers():
    # Section 3.2.9: 2025 participation 56.6% -> alpha 0.37; 93.2% eligible at the 70th percentile
    assert 1.5 - 2 * 0.566 == pytest.approx(0.37, abs=0.005)
    a = 1.5 - 2 * 0.566
    assert 1 - (100 - 70) ** 3 / (1000 - 1000 * a) ** 2 == pytest.approx(0.932, abs=0.002)


@pytest.mark.parametrize("year, participation", [(2025, 0.566), (2024, 0.551)])
def test_spline_model_matches_table_a8(year, participation):
    # UAC fits the spline on NSW+ACT students aged 16-20, while Table A8 covers every NSW ATAR
    # recipient (older and accumulating students sit lower). The model therefore reads slightly
    # high, by ~0.05 at the top rising to ~0.8 at the 30th percentile, in both years.
    path = PROCESSED / "a8_atar_percentiles.csv"
    if not path.exists():
        pytest.skip("UAC data not extracted")
    a8 = pd.read_csv(path)
    rows = a8[(a8["year"] == year) & (a8["percentile"].between(30, 99))]
    for _, r in rows.iterrows():
        tolerance = 0.35 if r["percentile"] >= 70 else 1.0
        diff = spline_atar(r["percentile"], participation) - r["atar"]
        assert -0.1 <= diff <= tolerance, (r["percentile"], diff)


def test_reproduces_table_a9_exactly(converter):
    a9 = pd.read_csv(PROCESSED / "a9_atar_aggregates.csv")
    for year, g in a9.groupby("year"):
        got = converter.to_atar(g["lowest_aggregate"].to_numpy(), int(year))
        np.testing.assert_allclose(got, g["atar"].to_numpy())


# Table 3.2 of the 2025 report (not used to build the converter)
@pytest.mark.parametrize(
    "aggregate, atar",
    [(450, 99.10), (400, 94.40), (350, 86.45), (300, 77.40), (250, 67.55), (200, 57.20), (150, 46.80)],
)
def test_matches_table_3_2_for_2025(converter, aggregate, atar):
    assert abs(float(converter.to_atar(aggregate, 2025)) - atar) <= 0.25


def test_cap_steps_and_monotone(converter):
    aggregates = np.linspace(0, 500, 2001)
    atars = converter.to_atar(aggregates, 2025)
    assert atars.max() == pytest.approx(99.95)
    assert atars.min() >= 0
    assert np.all(np.diff(atars) >= 0)
    assert np.allclose(np.round(atars * 20), atars * 20)
