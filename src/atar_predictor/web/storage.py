"""Student profiles saved as YAML files, one per student (same format as examples/student.yaml)."""

from __future__ import annotations

import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.extract import ROOT
from ..models import StudentProfile

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def default_dir() -> Path:
    return Path(os.environ.get("ATAR_STUDENTS_DIR", ROOT / "students"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


class StudentStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, student_id: str) -> Path:
        if not ID_RE.match(student_id):
            raise KeyError(student_id)
        return self.dir / f"{student_id}.yaml"

    def list(self) -> list[dict]:
        students = []
        for path in self.dir.glob("*.yaml"):
            if not ID_RE.match(path.stem):
                continue
            updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            try:
                data = yaml.safe_load(path.read_text()) or {}
                name, courses = data.get("name") or path.stem, len(data.get("courses") or [])
            except yaml.YAMLError:
                name, courses = path.stem, 0
            students.append({"id": path.stem, "name": name, "courses": courses, "updated": updated})
        return sorted(students, key=lambda s: s["updated"], reverse=True)

    def get(self, student_id: str) -> StudentProfile:
        path = self._path(student_id)
        if not path.exists():
            raise KeyError(student_id)
        return StudentProfile.from_yaml(path)

    def save(self, student_id: str, profile: StudentProfile) -> None:
        path = self._path(student_id)
        data = profile.model_dump(mode="json", exclude_defaults=True)
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        tmp.replace(path)

    def create(self, profile: StudentProfile) -> str:
        base = _slug(profile.name or "") or "student"
        student_id = base
        while self._path(student_id).exists():
            student_id = f"{base}-{secrets.token_hex(2)}"
        self.save(student_id, profile)
        return student_id

    def delete(self, student_id: str) -> None:
        path = self._path(student_id)
        if not path.exists():
            raise KeyError(student_id)
        path.unlink()

    def seed_example(self, example: Path) -> None:
        """Give a fresh students folder one example to explore."""
        if example.exists() and not any(self.dir.glob("*.yaml")):
            shutil.copy(example, self.dir / "example-student.yaml")
