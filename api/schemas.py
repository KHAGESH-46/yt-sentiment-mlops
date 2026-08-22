"""PHASE 3 — API schemas. This file IS the frozen /v1 contract.

Extension and backend evolve independently, so changes here must be
backward-compatible (add fields, never remove/rename).
"""
from pydantic import BaseModel, Field


class PredictItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=500)


class PredictBatchRequest(BaseModel):
    comments: list[PredictItem] = Field(..., min_length=1, max_length=50)


class PredictResult(BaseModel):
    id: str
    label: str
    confidence: float
    model_version: str


class PredictBatchResponse(BaseModel):
    results: list[PredictResult]
    model_version: str
    cache_hits: int


class FeedbackRequest(BaseModel):
    comment_id: str = Field(..., max_length=128)
    text: str = Field(..., max_length=500)
    predicted: str
    actual: str
    model_version: str = ""
