"""Averages of school assessment marks.

- Subject, per year: the weighted mark over that year's completed tasks (from assessment.py).
- Subject, overall: its years combined in proportion to how much of each year's weighting is
  complete. Two finished years count equally; a half-finished year counts half as much.
- All subjects, per year and overall: the mean across subjects that have a mark. A unit-weighted
  mean (2-unit courses count double a 1-unit extension) is also given.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assessment import AssessmentConfig, CourseAssessment, YearSummary, assess_course
from .courses import CourseInfo, course_info
from .models import StudentProfile


@dataclass
class SubjectAverage:
    course: str
    units: int
    year11: float | None
    year12: float | None
    overall: float | None
    year11_completed: float  # share of the year's weighting completed, 0-1
    year12_completed: float


@dataclass
class Averages:
    subjects: list[SubjectAverage]
    year11: float | None
    year12: float | None
    overall: float | None
    year11_unit_weighted: float | None
    year12_unit_weighted: float | None
    overall_unit_weighted: float | None


def _mark(year: YearSummary | None) -> tuple[float | None, float]:
    if year is None or year.weighted_mark is None:
        return None, 0.0
    return year.weighted_mark, year.completed_fraction


def _mean(values: list[float | None], weights: list[float] | None = None) -> float | None:
    weights = weights or [1.0] * len(values)
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def subject_average(info: CourseInfo, assessment: CourseAssessment) -> SubjectAverage:
    y11, done11 = _mark(assessment.y11)
    y12, done12 = _mark(assessment.y12)
    overall = _mean([y11, y12], [done11, done12])
    return SubjectAverage(info.name, info.units, y11, y12, overall, done11, done12)


def averages_from(courses: list[tuple[CourseInfo, CourseAssessment]]) -> Averages:
    subjects = [subject_average(info, a) for info, a in courses]
    units = [float(s.units) for s in subjects]
    return Averages(
        subjects=subjects,
        year11=_mean([s.year11 for s in subjects]),
        year12=_mean([s.year12 for s in subjects]),
        overall=_mean([s.overall for s in subjects]),
        year11_unit_weighted=_mean([s.year11 for s in subjects], units),
        year12_unit_weighted=_mean([s.year12 for s in subjects], units),
        overall_unit_weighted=_mean([s.overall for s in subjects], units),
    )


def compute_averages(profile: StudentProfile, cfg: AssessmentConfig = AssessmentConfig()) -> Averages:
    enrolled = {c.name for c in profile.courses}
    return averages_from([(course_info(c.name, enrolled), assess_course(c, cfg)) for c in profile.courses])
