"""Monte Carlo: student profile -> ATAR distribution.

Each draw samples the student's final school position, their school's strength, a scaling
year (from the last few years of UAC data), then runs scaling -> best 10 -> ATAR.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np

from .aggregate import AggregateResult, best_ten, best_ten_batch
from .assessment import AssessmentConfig, CourseAssessment, assess_course
from .atar import AtarConverter
from .courses import CourseInfo, course_info
from .models import StudentProfile
from .moderation import ModerationConfig, school_shift, school_z, state_percentile
from .scaling import ScalingData


@dataclass
class CourseResult:
    info: CourseInfo
    assessment: CourseAssessment
    state_percentile_p50: float
    scaled_p10: float  # per unit, 0-50
    scaled_p50: float
    scaled_p90: float
    hsc_mark_p50: int  # on the course's reported scale
    band_p50: str
    approximate_curve: bool


@dataclass
class Prediction:
    atar_p10: float
    atar_p50: float
    atar_p90: float
    aggregate_p10: float
    aggregate_p50: float
    aggregate_p90: float
    courses: list[CourseResult]
    median_breakdown: AggregateResult
    years: list[int]
    warnings: list[str]
    atar_draws: np.ndarray = field(repr=False)


def hsc_band(course: str, mark_one_unit: float) -> tuple[int, str]:
    if course == "Mathematics Extension 2":
        mark = round(2 * mark_one_unit)
        cuts, lowest = [(90, "E4"), (70, "E3"), (50, "E2")], "E1"
    elif "Extension" in course:
        mark = round(mark_one_unit)
        cuts, lowest = [(45, "E4"), (35, "E3"), (25, "E2")], "E1"
    else:
        mark = round(2 * mark_one_unit)
        cuts, lowest = [(90, "Band 6"), (80, "Band 5"), (70, "Band 4"), (60, "Band 3"), (50, "Band 2")], "Band 1"
    return mark, next((label for cut, label in cuts if mark >= cut), lowest)


def _truncate(atar: float) -> float:
    return float(np.floor(atar * 20 + 1e-6) / 20)


def check_courses(profile: StudentProfile, scaling: ScalingData) -> None:
    known = scaling.courses()
    problems = []
    for c in profile.courses:
        if c.name not in known:
            close = difflib.get_close_matches(c.name, sorted(known), n=3, cutoff=0.5)
            problems.append(f"{c.name!r}" + (f" (did you mean {', '.join(map(repr, close))}?)" if close else ""))
    if problems:
        raise ValueError("Unknown course name(s), use UAC Table A3 names: " + "; ".join(problems))


def predict(
    profile: StudentProfile,
    scaling: ScalingData | None = None,
    converter: AtarConverter | None = None,
    draws: int = 5000,
    seed: int = 0,
    n_years: int = 5,
    assess_cfg: AssessmentConfig = AssessmentConfig(),
    mod_cfg: ModerationConfig = ModerationConfig(),
) -> Prediction:
    scaling = scaling or ScalingData.load()
    converter = converter or AtarConverter.load()
    check_courses(profile, scaling)

    years = [y for y in scaling.years if y in converter.years][-n_years:]
    rng = np.random.default_rng(seed)
    enrolled = {c.name for c in profile.courses}
    infos = [course_info(c.name, enrolled) for c in profile.courses]
    assessments = [assess_course(c, assess_cfg) for c in profile.courses]
    warnings = [w for a in assessments for w in a.warnings]

    year_draw = rng.choice(years, size=draws)
    shift = school_shift(profile.cohort_strength, draws, rng, mod_cfg)
    k = len(infos)
    scaled, hsc, p_state = np.empty((draws, k)), np.empty((draws, k)), np.empty((draws, k))
    for j, (info, a) in enumerate(zip(infos, assessments)):
        z = school_z(a.rank_percentile, a.rank_percentile_sd, draws, rng, mod_cfg)
        p_state[:, j] = state_percentile(z, shift, rng, mod_cfg)
        for y in years:
            m = year_draw == y
            scaled[m, j] = scaling.curve(info.name, y)(p_state[m, j])
            hsc[m, j] = scaling.curve(info.name, y, "hsc")(p_state[m, j])
        borrowed = sorted({scaling.resolve(info.name, y)[0] for y in years} - {info.name})
        if borrowed:
            warnings.append(f"{info.name}: some years use scaling data from {', '.join(borrowed)}")

    aggregate = best_ten_batch(infos, scaled)
    atar = np.empty(draws)
    for y in years:
        m = year_draw == y
        atar[m] = converter.to_atar(aggregate[m], y)

    courses = []
    for j, (info, a) in enumerate(zip(infos, assessments)):
        s10, s50, s90 = np.percentile(scaled[:, j], [10, 50, 90])
        mark, band = hsc_band(info.name, float(np.median(hsc[:, j])))
        approx = scaling.curve(info.name, years[-1]).approximate
        courses.append(
            CourseResult(info, a, float(np.median(p_state[:, j])), s10, s50, s90, mark, band, approx)
        )

    breakdown = best_ten(infos, [c.scaled_p50 for c in courses])
    if not breakdown.eligible:
        warnings.append("Not ATAR-eligible: " + "; ".join(breakdown.reasons))

    a10, a50, a90 = np.percentile(atar, [10, 50, 90])
    g10, g50, g90 = np.percentile(aggregate, [10, 50, 90])
    return Prediction(
        _truncate(a10), _truncate(a50), _truncate(a90), g10, g50, g90,
        courses, breakdown, years, warnings, atar,
    )
