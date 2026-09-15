"""Internal assessment: weighted marks per year, projection of upcoming tasks,
and an estimate of the student's percentile within their school cohort.

Percentiles here are "fraction of the cohort below", so 1.0 is the top of the cohort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from scipy.stats import norm

from .models import Course, Task

RankSource = Literal["entered", "task_ranks", "z_score", "default"]


@dataclass(frozen=True)
class AssessmentConfig:
    # Assumptions, not fitted: no public paired data exists to estimate them.
    alpha_y11: float = 0.4  # Year 11 weight when no Year 12 weighting is complete
    task_sd: float = 8.0  # SD (percentage points) of a future task mark around the current mark
    default_school_mean: float = 70.0  # internal mark distribution used when nothing else is known
    default_school_sd: float = 12.0
    y11_only_penalty: float = 0.08  # extra percentile SD when only Year 11 evidence exists


@dataclass
class TaskRow:
    name: str
    pct: float | None
    weight: float
    contribution: float | None  # percentage points this task adds to the year's final mark


@dataclass
class YearSummary:
    year: int
    weighted_mark: float | None  # over completed tasks, in %
    completed_fraction: float  # share of the year's weighting completed
    total_weight: float  # weights as entered (ideally 100)
    projected_sd: float  # SD of the final year mark once upcoming tasks are sat
    rows: list[TaskRow]
    warnings: list[str] = field(default_factory=list)


@dataclass
class CourseAssessment:
    course: str
    y11: YearSummary | None
    y12: YearSummary | None
    w_y11: float
    blended_mark: float | None
    rank_percentile: float
    rank_percentile_sd: float
    rank_source: RankSource
    warnings: list[str] = field(default_factory=list)

    @property
    def y12_remaining_fraction(self) -> float:
        return 1.0 if self.y12 is None else 1.0 - self.y12.completed_fraction


def summarise_year(tasks: list[Task], year: int, cfg: AssessmentConfig) -> YearSummary | None:
    if not tasks:
        return None
    total = sum(t.weight for t in tasks)
    warnings = []
    if not math.isclose(total, 100.0, abs_tol=0.5):
        warnings.append(f"Year {year} weightings sum to {total:g}%, not 100%; normalising")

    done = [t for t in tasks if t.done]
    done_weight = sum(t.weight for t in done)
    weighted = sum(t.pct * t.weight for t in done) / done_weight if done else None

    upcoming_sq = sum((t.weight / total) ** 2 for t in tasks if not t.done)
    projected_sd = cfg.task_sd * math.sqrt(upcoming_sq)

    rows = [
        TaskRow(t.name, t.pct, t.weight, None if t.pct is None else t.pct * t.weight / total)
        for t in tasks
    ]
    return YearSummary(year, weighted, done_weight / total, total, projected_sd, rows, warnings)


def _year11_weight(y11: YearSummary | None, y12: YearSummary | None, cfg: AssessmentConfig) -> float:
    has11 = y11 is not None and y11.weighted_mark is not None
    has12 = y12 is not None and y12.weighted_mark is not None
    if not has11:
        return 0.0
    if not has12:
        return 1.0
    return cfg.alpha_y11 * (1.0 - y12.completed_fraction)


def _task_rank_percentile(tasks: list[Task]) -> float | None:
    ranked = [t for t in tasks if t.rank is not None and t.rank_of is not None]
    if not ranked:
        return None
    w = sum(t.weight for t in ranked)
    return sum((1 - (t.rank - 0.5) / t.rank_of) * t.weight for t in ranked) / w


def _task_z(tasks: list[Task]) -> float | None:
    scored = [t for t in tasks if t.cohort_mean is not None and t.cohort_sd is not None]
    if not scored:
        return None
    w = sum(t.weight for t in scored)
    return sum((t.mark - t.cohort_mean) / t.cohort_sd * t.weight for t in scored) / w


def _blend(a11: float | None, a12: float | None, w11: float) -> float | None:
    if a11 is None:
        return a12
    if a12 is None:
        return a11
    return w11 * a11 + (1 - w11) * a12


def assess_course(course: Course, cfg: AssessmentConfig = AssessmentConfig()) -> CourseAssessment:
    y11 = summarise_year(course.year11, 11, cfg)
    y12 = summarise_year(course.year12, 12, cfg)
    warnings = [f"{course.name}: {w}" for y in (y11, y12) if y for w in y.warnings]
    w11 = _year11_weight(y11, y12, cfg)
    blended = _blend(y11 and y11.weighted_mark, y12 and y12.weighted_mark, w11)

    remaining12 = 1.0 if y12 is None else 1.0 - y12.completed_fraction
    y11_only = w11 == 1.0

    # Rank evidence, strongest first. An entered overall rank always wins.
    if course.rank is not None:
        p = 1 - (course.rank - 0.5) / course.cohort_size
        sd, source = 0.02 + 0.10 * remaining12, "entered"
    elif (
        pr := _blend(_task_rank_percentile(course.year11), _task_rank_percentile(course.year12), w11)
    ) is not None:
        p, sd, source = pr, 0.04 + 0.12 * remaining12, "task_ranks"
    elif (z := _blend(_task_z(course.year11), _task_z(course.year12), w11)) is not None:
        p, sd, source = float(norm.cdf(z)), 0.06 + 0.12 * remaining12, "z_score"
    elif blended is not None:
        z = (blended - cfg.default_school_mean) / cfg.default_school_sd
        p, sd, source = float(norm.cdf(z)), 0.18 + 0.08 * remaining12, "default"
        warnings.append(
            f"{course.name}: no rank or cohort stats; percentile guessed from marks alone (wide range)"
        )
    else:
        p, sd, source = 0.5, 0.25, "default"
        warnings.append(f"{course.name}: no completed tasks; assuming the cohort median")

    if y11_only and source != "entered":
        sd += cfg.y11_only_penalty
    p = min(max(p, 0.001), 0.999)
    return CourseAssessment(course.name, y11, y12, w11, blended, p, sd, source, warnings)
