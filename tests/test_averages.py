from pathlib import Path

import pytest

from atar_predictor.averages import compute_averages
from atar_predictor.models import Course, StudentProfile, Task

EXAMPLE = Path(__file__).parents[1] / "examples" / "student.yaml"


def tasks(*spec):
    """spec items: (mark or None, weight)"""
    return [Task(name=f"T{i}", mark=m, weight=w) for i, (m, w) in enumerate(spec)]


def one(course: Course):
    return compute_averages(StudentProfile(courses=[course])).subjects[0]


def test_subject_overall_weights_two_complete_years_equally():
    s = one(Course(name="Physics", year11=tasks((70, 50), (80, 50)), year12=tasks((90, 100))))
    assert s.year11 == pytest.approx(75.0)
    assert s.year12 == pytest.approx(90.0)
    assert s.overall == pytest.approx(82.5)


def test_partly_complete_year_counts_in_proportion():
    s = one(Course(name="Physics", year11=tasks((60, 100)), year12=tasks((90, 50), (None, 50))))
    assert s.year12_completed == pytest.approx(0.5)
    assert s.overall == pytest.approx((60 * 1.0 + 90 * 0.5) / 1.5)


def test_single_year_and_no_marks():
    only12 = one(Course(name="Economics", year12=tasks((84, 100))))
    assert only12.year11 is None and only12.overall == pytest.approx(84.0)
    nothing = one(Course(name="Economics", year12=tasks((None, 100))))
    assert nothing.year12 is None and nothing.overall is None


def test_all_subject_averages_simple_and_unit_weighted():
    profile = StudentProfile(courses=[
        Course(name="Mathematics Advanced", year11=tasks((80, 100)), year12=tasks((70, 100))),  # 2 units
        Course(name="Mathematics Extension 1", year12=tasks((40, 100))),  # 1 unit, no Year 11
    ])
    avg = compute_averages(profile)
    assert avg.year11 == pytest.approx(80.0)  # only subjects with a Year 11 mark
    assert avg.year12 == pytest.approx(55.0)
    assert avg.year12_unit_weighted == pytest.approx((70 * 2 + 40 * 1) / 3)
    assert avg.overall == pytest.approx((75.0 + 40.0) / 2)
    assert avg.overall_unit_weighted == pytest.approx((75.0 * 2 + 40.0) / 3)


def test_example_profile_matches_assessment_marks():
    avg = compute_averages(StudentProfile.from_yaml(EXAMPLE))
    english = next(s for s in avg.subjects if s.course == "English Advanced")
    assert english.year11 == pytest.approx(77.0)
    assert english.year12 == pytest.approx(81.5)
    assert english.overall == pytest.approx(79.25)
    assert len(avg.subjects) == 7
