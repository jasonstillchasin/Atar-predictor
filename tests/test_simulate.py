from pathlib import Path

import pytest

from atar_predictor.models import Course, StudentProfile, Task
from atar_predictor.simulate import hsc_band, predict

EXAMPLE = Path(__file__).parents[1] / "examples" / "student.yaml"
SUBJECTS = ["English Advanced", "Mathematics Advanced", "Physics", "Chemistry", "Economics"]


def profile(rank: int, strength=None, names=SUBJECTS) -> StudentProfile:
    full_year = [Task(name="Trial", mark=80, weight=100)]
    courses = [Course(name=n, cohort_size=100, rank=rank, year12=full_year) for n in names]
    return StudentProfile(cohort_strength=strength, courses=courses)


def test_example_profile_runs(scaling, converter):
    pred = predict(StudentProfile.from_yaml(EXAMPLE), scaling, converter, draws=2000)
    assert 0 <= pred.atar_p10 <= pred.atar_p50 <= pred.atar_p90 <= 99.95
    assert pred.median_breakdown.eligible
    assert sum(c.units_used for c in pred.median_breakdown.contributions) == 10


def test_better_rank_gives_higher_atar(scaling, converter):
    top = predict(profile(5), scaling, converter, draws=2000)
    middle = predict(profile(50), scaling, converter, draws=2000)
    assert top.atar_p50 > middle.atar_p50 + 10


def test_selective_cohort_raises_estimate(scaling, converter):
    selective = predict(profile(30, "selective"), scaling, converter, draws=2000)
    below = predict(profile(30, "below"), scaling, converter, draws=2000)
    assert selective.atar_p50 > below.atar_p50


def test_known_strength_narrows_range(scaling, converter):
    unknown = predict(profile(30), scaling, converter, draws=4000)
    known = predict(profile(30, "average"), scaling, converter, draws=4000)
    assert (known.atar_p90 - known.atar_p10) < (unknown.atar_p90 - unknown.atar_p10)


def test_unknown_course_name_suggests_match(scaling, converter):
    with pytest.raises(ValueError, match="Mathematics Advanced"):
        predict(profile(10, names=["Maths Advanced"] + SUBJECTS[2:]), scaling, converter, draws=10)


def test_hsc_bands():
    assert hsc_band("Physics", 45.2) == (90, "Band 6")
    assert hsc_band("History Extension", 44.0) == (44, "E3")
    assert hsc_band("Mathematics Extension 2", 45.0) == (90, "E4")
