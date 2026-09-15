from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from atar_predictor.web.app import create_app

SUBJECTS = ["English Advanced", "Mathematics Advanced", "Physics", "Chemistry", "Economics"]
PROFILE = {
    "name": "Test Student",
    "courses": [
        {"name": n, "cohort_size": 100, "rank": 10, "year12": [{"name": "Trial", "mark": 80, "weight": 100}]}
        for n in SUBJECTS
    ],
}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "students"))


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ATAR Workbook" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_fresh_folder_is_seeded_with_example(client):
    students = client.get("/api/students").json()
    assert [s["id"] for s in students] == ["example-student"]
    assert students[0]["courses"] == 7


def test_create_read_update_delete(client, tmp_path):
    sid = client.post("/api/students", json=PROFILE).json()["id"]
    assert sid == "test-student"
    assert (tmp_path / "students" / "test-student.yaml").exists()

    profile = client.get(f"/api/students/{sid}").json()["profile"]
    assert profile["courses"][0]["year12"][0]["mark"] == 80
    assert profile["courses"][0]["year11"] == []

    renamed = {**PROFILE, "name": "Renamed"}
    assert client.put(f"/api/students/{sid}", json=renamed).status_code == 200
    assert any(s["name"] == "Renamed" for s in client.get("/api/students").json())

    assert client.delete(f"/api/students/{sid}").status_code == 204
    assert client.get(f"/api/students/{sid}").status_code == 404


def test_duplicate_names_get_distinct_ids(client):
    first = client.post("/api/students", json=PROFILE).json()["id"]
    second = client.post("/api/students", json=PROFILE).json()["id"]
    assert first != second and second.startswith("test-student-")


def test_bad_ids_and_unknown_students(client):
    assert client.get("/api/students/Not_Valid").status_code == 404
    assert client.put("/api/students/nobody", json=PROFILE).status_code == 404
    assert client.delete("/api/students/nobody").status_code == 404


def test_invalid_task_rejected(client):
    bad = deepcopy(PROFILE)
    bad["courses"][0]["year12"][0]["mark"] = 120
    r = client.post("/api/students", json=bad)
    assert r.status_code == 422
    assert "exceeds" in str(r.json())


def test_predict(client, scaling):
    r = client.post("/api/predict", json={"profile": PROFILE, "draws": 500})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["atar"]["p10"] <= body["atar"]["p50"] <= body["atar"]["p90"] <= 99.95
    assert sum(body["histogram"]["counts"]) == 500
    assert body["courses"][0]["year12"]["weighted_mark"] == 80
    assert sum(c["units_counted_at_median"] for c in body["courses"]) == 10
    assert body["averages"]["year12"] == pytest.approx(80.0)
    assert body["averages"]["year11"] is None
    assert [s["overall"] for s in body["averages"]["subjects"]] == pytest.approx([80.0] * 5)


def test_predict_unknown_course(client, scaling):
    bad = deepcopy(PROFILE)
    bad["courses"][1]["name"] = "Maths Advanced"
    r = client.post("/api/predict", json={"profile": bad, "draws": 200})
    assert r.status_code == 422
    assert "Mathematics Advanced" in r.json()["detail"]


def test_courses_listed(client, scaling):
    courses = {c["name"]: c for c in client.get("/api/courses").json()}
    assert courses["Mathematics Extension 2"]["units"] == 2
    assert courses["English Extension 1"]["english"]
