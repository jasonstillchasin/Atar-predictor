"""Course metadata: unit values, English flag, subject grouping, predecessor courses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import yaml


@dataclass(frozen=True)
class CourseInfo:
    name: str
    units: int
    english: bool
    subject: str


@lru_cache
def _config() -> dict:
    return yaml.safe_load(files("atar_predictor.data").joinpath("courses.yaml").read_text())


def predecessors() -> dict[str, str]:
    return dict(_config().get("predecessors", {}))


def course_info(name: str, enrolled: set[str] | frozenset[str] = frozenset()) -> CourseInfo:
    """Rules for one course, given the student's other enrolments (Maths Ext 1 depends on Ext 2)."""
    spec = _config()["courses"].get(name, {})
    units = spec.get("units", 1 if "Extension" in name else 2)
    for other, override in spec.get("units_with", {}).items():
        if other in enrolled:
            units = override
    return CourseInfo(name, units, spec.get("english", False), spec.get("subject", name))
