import pytest
from pydantic import ValidationError

from atar_predictor.assessment import assess_course, summarise_year, AssessmentConfig
from atar_predictor.models import Course, Task

CFG = AssessmentConfig()


def tasks(*spec):
    """spec items: (mark or None, weight)"""
    return [Task(name=f"T{i}", mark=m, weight=w) for i, (m, w) in enumerate(spec)]


def test_weighted_mark_hand_computed():
    s = summarise_year(tasks((80, 20), (70, 30), (90, 50)), 12, CFG)
    assert s.weighted_mark == pytest.approx(82.0)
    assert s.completed_fraction == pytest.approx(1.0)
    assert s.projected_sd == 0.0
    assert not s.warnings


def test_raw_score_out_of():
    t = Task(name="Essay", mark=18, out_of=25, weight=100)
    assert summarise_year([t], 12, CFG).weighted_mark == pytest.approx(72.0)


def test_weight_sum_warning_and_normalising():
    s = summarise_year(tasks((80, 20), (60, 20)), 12, CFG)
    assert s.weighted_mark == pytest.approx(70.0)
    assert any("sum to 40%" in w for w in s.warnings)


def test_invalid_task_inputs():
    with pytest.raises(ValidationError):
        Task(name="x", mark=30, out_of=25, weight=10)
    with pytest.raises(ValidationError):
        Task(name="x", mark=20, weight=0)


def test_projection_narrows_as_weighting_completes():
    early = summarise_year(tasks((80, 25), (None, 25), (None, 50)), 12, CFG)
    late = summarise_year(tasks((80, 25), (75, 25), (None, 50)), 12, CFG)
    assert early.completed_fraction < late.completed_fraction
    assert early.projected_sd > late.projected_sd > 0


def test_entered_rank_overrides_task_ranks():
    t = [Task(name="Half-yearly", mark=90, weight=100, rank=40, rank_of=100)]
    course = Course(name="Chemistry", cohort_size=100, rank=1, year12=t)
    a = assess_course(course)
    assert a.rank_source == "entered"
    assert a.rank_percentile == pytest.approx(0.995)


def test_task_rank_percentile_used_when_no_overall_rank():
    t = [
        Task(name="A", mark=90, weight=50, rank=10, rank_of=100),
        Task(name="B", mark=85, weight=50, rank=30, rank_of=100),
    ]
    a = assess_course(Course(name="Biology", year12=t))
    assert a.rank_source == "task_ranks"
    # ranks 10/100 and 30/100 -> percentiles 0.905 and 0.705, equally weighted
    assert a.rank_percentile == pytest.approx(0.805)


def test_year11_weight_shrinks_to_zero_when_year12_complete():
    y11 = tasks((60, 100))
    partial = assess_course(Course(name="Physics", year11=y11, year12=tasks((90, 50), (None, 50))))
    full = assess_course(Course(name="Physics", year11=y11, year12=tasks((90, 50), (90, 50))))
    assert partial.w_y11 == pytest.approx(0.2)
    assert full.w_y11 == 0.0
    assert full.blended_mark == pytest.approx(90.0)
    assert partial.blended_mark == pytest.approx(0.2 * 60 + 0.8 * 90)


def test_marks_only_falls_back_to_default_with_warning():
    a = assess_course(Course(name="Economics", year12=tasks((82, 100))))
    assert a.rank_source == "default"
    assert a.rank_percentile_sd >= 0.18
    assert a.warnings
