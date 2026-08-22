import os
import tempfile

# Must be set before importing api.main (store reads env lazily, but keep it explicit).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_predictions.db")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)  # context manager below triggers lifespan (store init)


def _c():
    with client:
        yield client


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_batch_returns_labels():
    with TestClient(app) as c:
        r = c.post(
            "/v1/predict-batch",
            json={"comments": [{"id": "1", "text": "great video, loved it"}, {"id": "2", "text": "terrible waste"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 2
        for res in body["results"]:
            assert res["label"] in {"positive", "neutral", "negative"}
            assert 0.0 <= res["confidence"] <= 1.0
            assert res["model_version"]
        assert body["cache_hits"] == 0


def test_predict_batch_cache_hit_on_second_call():
    with TestClient(app) as c:
        payload = {"comments": [{"id": "7", "text": "same text again"}]}
        c.post("/v1/predict-batch", json=payload)
        r2 = c.post("/v1/predict-batch", json=payload)
        assert r2.json()["cache_hits"] == 1


def test_predict_batch_over_limit_is_422():
    items = [{"id": str(i), "text": "x"} for i in range(51)]
    with TestClient(app) as c:
        r = c.post("/v1/predict-batch", json={"comments": items})
        assert r.status_code == 422


def test_predict_text_over_500_chars_is_422():
    with TestClient(app) as c:
        r = c.post("/v1/predict-batch", json={"comments": [{"id": "1", "text": "a" * 501}]})
        assert r.status_code == 422


def test_feedback_saved():
    with TestClient(app) as c:
        r = c.post(
            "/v1/feedback",
            json={"comment_id": "1", "text": "nice", "predicted": "neutral", "actual": "positive"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "saved"


def test_model_current():
    with TestClient(app) as c:
        r = c.get("/v1/model/current")
        assert r.status_code == 200
        assert "model_version" in r.json()
