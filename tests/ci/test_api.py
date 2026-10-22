"""The REST API of the web app with a stand-in engine (no model): routes, validation, OpenAPI schema."""
import pytest
from fastapi.testclient import TestClient

from assistant_web import server


class StandInEngine:
    def check(self, codes, target, margin=None):
        if len(codes) != 52:
            raise ValueError("a placement has 52 slots")
        return {"kind": "model", "target": target, "margin": margin}

    def predict(self, placements, target=None):
        if not placements:
            raise ValueError("1..4096 placements per request")
        return {"n": len(placements), "spectra": [[1.0] * 231 for _ in placements], "target": target}

    def map(self, codes, index):
        return {"index": index}

    def fixes(self, codes, target, mode, max_radius, margin=None, remove=True):
        return {"status": "none", "max_radius": max_radius, "remove": remove}

    def improve(self, codes, target):
        return {"status": "none"}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setitem(server.STATE, "engine", StandInEngine())
    monkeypatch.setitem(server.STATE, "setup", {"targets": ["0.8", "0.6", "0.5"]})
    return TestClient(server.app)              # no "with": the lifespan (model loading) does not run


def test_page_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "PDN Design Assistant" in r.text
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/setup").json()["targets"] == ["0.8", "0.6", "0.5"]


def test_predict_and_validation(client):
    r = client.post("/api/predict", json={"placements": [[1] + [0] * 51], "target": "0.8"})
    assert r.status_code == 200 and r.json()["n"] == 1 and len(r.json()["spectra"][0]) == 231
    assert client.post("/api/predict", json={"placements": []}).status_code == 400       # engine ValueError -> 400
    assert client.post("/api/predict", json={"placements": "x"}).status_code == 422      # schema error
    assert client.post("/api/check", json={"codes": [0] * 10, "target": "0.8"}).status_code == 400


def test_fix_radius_is_clamped(client):
    r = client.post("/api/fixes", json={"codes": [1] + [0] * 51, "target": "0.8", "max_radius": 9})
    assert r.json()["max_radius"] == 3 and r.json()["remove"] is True


def test_openapi_lists_the_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/check", "/api/map", "/api/fixes", "/api/improve", "/api/predict",
              "/api/ecad/run", "/api/ecad/status", "/api/ecad/result"):
        assert p in paths, p
