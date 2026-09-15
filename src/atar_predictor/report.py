"""Serialise a Prediction to plain JSON-ready data (CLI --json and the web app)."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from .assessment import YearSummary
from .averages import averages_from
from .simulate import Prediction


def _year(y: YearSummary | None) -> dict | None:
    if y is None:
        return None
    return {
        "weighted_mark": y.weighted_mark,
        "completed_fraction": y.completed_fraction,
        "total_weight": y.total_weight,
        "projected_sd": y.projected_sd,
        "warnings": y.warnings,
        "tasks": [
            {"name": r.name, "pct": r.pct, "weight": r.weight, "contribution": r.contribution} for r in y.rows
        ],
    }


def _histogram(draws: np.ndarray, bins: int) -> dict:
    lo, hi = math.floor(draws.min()), math.ceil(draws.max())
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(draws, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": counts.tolist()}


def prediction_to_dict(pred: Prediction, histogram_bins: int = 0) -> dict:
    units = {c.course: c.units_used for c in pred.median_breakdown.contributions}
    out = {
        "atar": {"p10": pred.atar_p10, "p50": pred.atar_p50, "p90": pred.atar_p90},
        "aggregate": {"p10": pred.aggregate_p10, "p50": pred.aggregate_p50, "p90": pred.aggregate_p90},
        "scaling_years": pred.years,
        "eligible": pred.median_breakdown.eligible,
        "eligibility_reasons": pred.median_breakdown.reasons,
        "courses": [
            {
                "course": c.info.name,
                "units": c.info.units,
                "units_counted_at_median": units.get(c.info.name, 0),
                "year11": _year(c.assessment.y11),
                "year12": _year(c.assessment.y12),
                "year11_weighted": c.assessment.y11 and c.assessment.y11.weighted_mark,
                "year12_weighted": c.assessment.y12 and c.assessment.y12.weighted_mark,
                "year11_blend_weight": c.assessment.w_y11,
                "blended_mark": c.assessment.blended_mark,
                "school_percentile": c.assessment.rank_percentile,
                "school_percentile_sd": c.assessment.rank_percentile_sd,
                "rank_source": c.assessment.rank_source,
                "state_percentile": c.state_percentile_p50,
                "scaled_per_unit": {"p10": c.scaled_p10, "p50": c.scaled_p50, "p90": c.scaled_p90},
                "hsc_mark_p50": c.hsc_mark_p50,
                "band_p50": c.band_p50,
                "approximate_curve": c.approximate_curve,
            }
            for c in pred.courses
        ],
        "averages": asdict(averages_from([(c.info, c.assessment) for c in pred.courses])),
        "warnings": pred.warnings,
        "draws": int(len(pred.atar_draws)),
    }
    if histogram_bins:
        out["histogram"] = _histogram(pred.atar_draws, histogram_bins)
    return out
