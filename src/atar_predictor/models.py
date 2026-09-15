"""Student profile schema: courses -> Year 11 / Year 12 assessment tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CohortStrength = Literal["below", "average", "above", "selective"]


class Task(BaseModel):
    """One internal assessment task from a school's assessment schedule."""

    name: str
    weight: float = Field(gt=0, le=100, description="Weighting in % for that year")
    mark: float | None = Field(default=None, ge=0, description="Raw score; None = upcoming")
    out_of: float = Field(default=100, gt=0)
    rank: int | None = Field(default=None, ge=1, description="Rank in this task")
    rank_of: int | None = Field(default=None, ge=1, description="Students ranked in this task")
    cohort_mean: float | None = Field(default=None, ge=0, description="Same scale as mark")
    cohort_sd: float | None = Field(default=None, gt=0, description="Same scale as mark")

    @model_validator(mode="after")
    def _check(self) -> Task:
        if self.mark is not None and self.mark > self.out_of:
            raise ValueError(f"{self.name}: mark {self.mark} exceeds out_of {self.out_of}")
        if self.rank is not None and self.rank_of is not None and self.rank > self.rank_of:
            raise ValueError(f"{self.name}: rank {self.rank} exceeds rank_of {self.rank_of}")
        if self.mark is None and (self.rank is not None or self.cohort_mean is not None):
            raise ValueError(f"{self.name}: rank/cohort stats given for an upcoming task")
        return self

    @property
    def done(self) -> bool:
        return self.mark is not None

    @property
    def pct(self) -> float | None:
        return None if self.mark is None else 100 * self.mark / self.out_of


class Course(BaseModel):
    """A course (by UAC course name) with its assessment tasks per year."""

    name: str
    cohort_size: int | None = Field(default=None, ge=1)
    rank: int | None = Field(default=None, ge=1, description="Current overall course rank")
    year11: list[Task] = Field(default_factory=list)
    year12: list[Task] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Course:
        if self.rank is not None and self.cohort_size is None:
            raise ValueError(f"{self.name}: rank given without cohort_size")
        if self.rank is not None and self.cohort_size is not None and self.rank > self.cohort_size:
            raise ValueError(f"{self.name}: rank {self.rank} exceeds cohort_size {self.cohort_size}")
        return self

    def tasks(self, year: Literal[11, 12]) -> list[Task]:
        return self.year11 if year == 11 else self.year12


class StudentProfile(BaseModel):
    name: str | None = None
    cohort_strength: CohortStrength | None = Field(
        default=None, description="School cohort strength vs the state; None = unknown (widest range)"
    )
    courses: list[Course]

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudentProfile:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
