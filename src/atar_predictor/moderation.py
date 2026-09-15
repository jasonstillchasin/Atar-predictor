"""School cohort percentile -> statewide course percentile.

NESA moderation preserves a school's rank order, so a student's HSC position is driven by
(a) their rank in the school cohort and (b) how that cohort performs relative to the state.
We model this in z-space:

    z_state = school_shift + course_shift + spread_ratio * z_school

With the defaults, an average school gives Var(z_state) = 0.35² + 0.26² + 0.9² ≈ 1, i.e. the
statewide distribution is recovered. All parameters are assumptions (no public data to fit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .models import CohortStrength

STRENGTH_SHIFT: dict[CohortStrength, float] = {
    "below": -0.4,
    "average": 0.0,
    "above": 0.4,
    "selective": 1.0,
}


@dataclass(frozen=True)
class ModerationConfig:
    spread_ratio: float = 0.9  # within-school SD relative to the state SD
    sigma_school: float = 0.35  # school strength spread when cohort strength is unknown
    sigma_school_known: float = 0.2  # residual spread once the user picks a strength
    sigma_course: float = 0.26  # course-specific deviation from the school's overall strength
    max_sd_z: float = 1.0


def school_shift(
    strength: CohortStrength | None, n: int, rng: np.random.Generator, cfg: ModerationConfig
) -> np.ndarray:
    """One draw per simulation, shared by all of the student's courses."""
    mean = STRENGTH_SHIFT[strength] if strength else 0.0
    sigma = cfg.sigma_school if strength is None else cfg.sigma_school_known
    return mean + sigma * rng.standard_normal(n)


def school_z(p: float, p_sd: float, n: int, rng: np.random.Generator, cfg: ModerationConfig) -> np.ndarray:
    """Sample the student's final within-school position, given the estimate and its uncertainty."""
    z0 = norm.ppf(p)
    sd_z = min(p_sd / norm.pdf(z0), cfg.max_sd_z)  # delta method: percentile SD -> z SD
    return z0 + sd_z * rng.standard_normal(n)


def state_percentile(
    z_school: np.ndarray, shift: np.ndarray, rng: np.random.Generator, cfg: ModerationConfig
) -> np.ndarray:
    course_shift = cfg.sigma_course * rng.standard_normal(z_school.shape)
    return norm.cdf(shift + course_shift + cfg.spread_ratio * z_school)
