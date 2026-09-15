"""Local web app: edit student profiles, save and switch between students, run the simulator.

    uv run atar-web                # http://127.0.0.1:8765

Binds to localhost only: saved students are personal data and the model uses UAC data
that must not be served publicly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..atar import AtarConverter
from ..courses import course_info
from ..data.extract import ROOT
from ..models import StudentProfile
from ..report import prediction_to_dict
from ..scaling import ScalingData
from ..simulate import predict
from .storage import StudentStore, default_dir

STATIC = Path(__file__).parent / "static"
EXAMPLE = ROOT / "examples" / "student.yaml"


class PredictRequest(BaseModel):
    profile: StudentProfile
    draws: int = Field(default=4000, ge=200, le=20000)
    seed: int = 0
    years: int = Field(default=5, ge=1, le=10)


def create_app(students_dir: Path | None = None) -> FastAPI:
    store = StudentStore(students_dir or default_dir())
    store.seed_example(EXAMPLE)
    model: dict = {}

    def load_model() -> tuple[ScalingData, AtarConverter]:
        if not model:
            try:
                model["scaling"], model["converter"] = ScalingData.load(), AtarConverter.load()
            except FileNotFoundError as exc:
                raise HTTPException(503, str(exc)) from exc
        return model["scaling"], model["converter"]

    app = FastAPI(title="ATAR Workbook", docs_url=None, redoc_url=None)

    @app.get("/api/courses")
    def courses() -> list[dict]:
        scaling, _ = load_model()
        return [
            {"name": name, "units": info.units, "english": info.english}
            for name in sorted(scaling.courses())
            for info in [course_info(name)]
        ]

    @app.get("/api/students")
    def list_students() -> list[dict]:
        return store.list()

    @app.get("/api/students/{student_id}")
    def get_student(student_id: str) -> dict:
        try:
            return {"id": student_id, "profile": store.get(student_id).model_dump(mode="json")}
        except KeyError:
            raise HTTPException(404, "Student not found") from None

    @app.post("/api/students", status_code=201)
    def create_student(profile: StudentProfile) -> dict:
        return {"id": store.create(profile)}

    @app.put("/api/students/{student_id}")
    def save_student(student_id: str, profile: StudentProfile) -> dict:
        try:
            store.get(student_id)  # must already exist; new students go through POST
            store.save(student_id, profile)
        except KeyError:
            raise HTTPException(404, "Student not found") from None
        return {"id": student_id}

    @app.delete("/api/students/{student_id}", status_code=204)
    def delete_student(student_id: str) -> Response:
        try:
            store.delete(student_id)
        except KeyError:
            raise HTTPException(404, "Student not found") from None
        return Response(status_code=204)

    @app.post("/api/predict")
    def run_prediction(req: PredictRequest) -> dict:
        scaling, converter = load_model()
        if not req.profile.courses:
            raise HTTPException(422, "Add at least one course")
        try:
            pred = predict(req.profile, scaling, converter, draws=req.draws, seed=req.seed, n_years=req.years)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return prediction_to_dict(pred, histogram_bins=40)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ATAR Workbook web app (local only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--students-dir", type=Path, default=None, help="Where student YAML files are saved")
    args = parser.parse_args()

    import uvicorn

    print(f"ATAR Workbook: http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.students_dir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
