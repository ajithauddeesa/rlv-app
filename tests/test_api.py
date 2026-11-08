import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "SIGNAL_ACTIVE", tmp_path / "signal.json")
    monkeypatch.setattr(main, "LINE_FILE", tmp_path / "line.json")
    monkeypatch.setattr(main, "OUT_PREFIX", str(tmp_path / "events"))
    return TestClient(main.app)


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_signal_rejects_bad_state(client):
    r = client.post("/api/signal", json={"phases": [{"start": 0, "end": 5, "state": "PURPLE"}]})
    assert r.status_code == 400


def test_signal_round_trip(client):
    body = {"clock_offset_s": 1.5, "phases": [{"start": 0, "end": 10, "state": "green"}]}
    assert client.post("/api/signal", json=body).status_code == 200
    got = client.get("/api/signal").json()
    assert got["clock_offset_s"] == 1.5 and got["phases"][0]["state"] == "GREEN"


def test_stopline_is_ordered_left_to_right(client):
    r = client.post("/api/stopline", json={"x1": 900, "y1": 500, "x2": 100, "y2": 520}).json()
    assert r["x1"] < r["x2"]


def test_stopline_rejects_zero_length(client):
    assert client.post("/api/stopline", json={"x1": 5, "y1": 5, "x2": 5, "y2": 5}).status_code == 400


def test_events_empty_before_analysis(client):
    assert client.get("/api/events").json()["events"] == []


def test_analyze_requires_video(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "VIDEO", tmp_path / "missing.mp4")
    assert client.post("/api/analyze").status_code == 400