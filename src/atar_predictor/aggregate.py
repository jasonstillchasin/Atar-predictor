"""Best-10-units aggregate: best 2 units of English + best 8 of the remaining units.

Scaled marks are per unit (0-50), so the aggregate is out of 500. A course may contribute
only some of its units: UAC's 2025 scaling report (section 5.3) notes that students with more
than 10 units have "at least 1 unit" omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .courses import CourseInfo


@dataclass
class Contribution:
    course: str
    units_used: int
    scaled_per_unit: float

    @property
    def points(self) -> float:
        return self.units_used * self.scaled_per_unit


@dataclass
class AggregateResult:
    aggregate: float
    contributions: list[Contribution]
    eligible: bool
    reasons: list[str]


def eligibility(courses: list[CourseInfo]) -> list[str]:
    """Reasons the pattern of study is not ATAR-eligible (empty list = eligible)."""
    reasons = []
    if sum(c.units for c in courses) < 10:
        reasons.append("fewer than 10 units of ATAR courses")
    if sum(c.units for c in courses if c.english) < 2:
        reasons.append("fewer than 2 units of English")
    if sum(1 for c in courses if c.units >= 2) < 3:
        reasons.append("fewer than three courses of 2 units or more")
    if len({c.subject for c in courses}) < 4:
        reasons.append("fewer than four subjects")
    return reasons


def best_ten(courses: list[CourseInfo], scaled: list[float]) -> AggregateResult:
    """Aggregate for one draw, with the breakdown of which units counted."""
    slots = [(s, c) for c, s in zip(courses, scaled) for _ in range(c.units)]
    english = sorted((sl for sl in slots if sl[1].english), key=lambda sl: -sl[0])
    chosen = english[:2]
    rest = sorted(english[2:] + [sl for sl in slots if not sl[1].english], key=lambda sl: -sl[0])
    chosen += rest[:8]

    used: dict[str, Contribution] = {}
    for s, c in chosen:
        used.setdefault(c.name, Contribution(c.name, 0, s)).units_used += 1
    total = sum(s for s, _ in chosen)
    reasons = eligibility(courses)
    return AggregateResult(total, list(used.values()), not reasons, reasons)


def best_ten_batch(courses: list[CourseInfo], scaled: np.ndarray) -> np.ndarray:
    """Vectorised aggregate. scaled: (draws, n_courses) per-unit scaled marks -> (draws,)."""
    reps = np.array([c.units for c in courses])
    is_english = np.repeat(np.array([c.english for c in courses]), reps)
    slots = np.repeat(scaled, reps, axis=1)

    eng = -np.sort(-slots[:, is_english], axis=1)
    eng_top, eng_rest = eng[:, :2], eng[:, 2:]
    others = np.concatenate([eng_rest, slots[:, ~is_english]], axis=1)
    others_top = -np.sort(-others, axis=1)[:, :8]
    return eng_top.sum(axis=1) + others_top.sum(axis=1)
