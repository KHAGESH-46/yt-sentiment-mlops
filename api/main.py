"""PHASE 3 — FastAPI serving app.

Endpoints: /v1/predict-batch · /v1/feedback · /v1/model/current · /health
Behaviors: batch caps (pydantic), per-IP rate limit, LRU cache, SQLite logging.
Run:  uvicorn api.main:app --reload
"""
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.cache import LRUCache
from api.inference import get_model
from api.schemas import (
    FeedbackRequest,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictResult,
)
from api import store

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "10000"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init()
    print("startup: store ready, model =",
          get_model().version, f"(dummy={get_model().version.endswith('-dummy')})")
    yield


app = FastAPI(title="YT Comment Sentiment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # extension fetches come from the service worker; keep permissive for dev
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = LRUCache(CACHE_SIZE)
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": get_model().version,
        "cache_hit_rate": round(cache.hit_rate, 3),
        "store": store.counts(),
    }


@app.post("/v1/predict-batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")

    model = get_model()
    keys = [cache.key_for(c.text) for c in req.comments]

    results: dict[int, PredictResult] = {}
    miss_idx: list[int] = []
    for i, k in enumerate(keys):
        cached = cache.get(k)
        if cached:
            label, conf, ver = cached
            results[i] = PredictResult(
                id=req.comments[i].id, label=label, confidence=conf, model_version=ver
            )
        else:
            miss_idx.append(i)

    if miss_idx:
        t0 = time.perf_counter()
        preds = model.predict([req.comments[i].text for i in miss_idx])
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(miss_idx), 1)
        for j, i in enumerate(miss_idx):
            label, conf = preds[j]
            ver = model.version
            cache.set(keys[i], (label, conf, ver))
            results[i] = PredictResult(
                id=req.comments[i].id, label=label, confidence=conf, model_version=ver
            )
            try:  # logging must never break serving
                store.log_prediction(
                    text_hash=keys[i], text=req.comments[i].text, label=label,
                    confidence=conf, model_version=ver, latency_ms=round(latency_ms, 2),
                )
            except Exception as exc:
                print(f"store error (ignored): {exc}")

    return PredictBatchResponse(
        results=[results[i] for i in range(len(req.comments))],
        model_version=model.version,
        cache_hits=len(results) - len(miss_idx),
    )


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    store.log_feedback(
        comment_id=req.comment_id, text=req.text, predicted=req.predicted,
        actual=req.actual, model_version=req.model_version,
    )
    return {"status": "saved"}


@app.get("/v1/model/current")
def model_current():
    return {
        "model_version": get_model().version,
        "registry_stage": os.getenv("REGISTRY_STAGE", "Production"),
    }
