"""Model loading + prediction — pluggable backends.

MODEL_BACKEND env var:
  auto        (default) transformer -> sklearn -> dummy, whatever loads
  transformer pre-trained HF sentiment model (real deal, needs torch)
  sklearn     models/model.joblib (TF-IDF baseline, our Phase-2 starter)
  dummy       deterministic hash placeholder (tests / offline)

HONEST EXPECTATIONS: the transformer is ~85-92% accurate on informal
English — NOT 100%. Sarcasm, rare slang, and mixed-language comments stay
hard; the confidence score + "?" badge exist precisely for that tail.
"""
import hashlib
import os
from pathlib import Path

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
_env_version = os.getenv("MODEL_VERSION")
HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
LABELS = ["positive", "neutral", "negative"]


class DummyModel:
    """Deterministic hash-based placeholder — same text always gets the same label."""

    version = _env_version or "0.0.0-dummy"

    def predict(self, texts):
        out = []
        for t in texts:
            h = hashlib.md5(t.lower().strip().encode()).hexdigest()
            idx = int(h, 16) % 3
            conf = 0.55 + (int(h[:2], 16) / 255) * 0.4  # 0.55 - 0.95
            out.append((LABELS[idx], round(conf, 4)))
        return out


class SklearnWrapper:
    def __init__(self, pipeline) -> None:
        self.pipe = pipeline
        self.version = _env_version or "0.1.0-tfidf"

    def predict(self, texts):
        probs = self.pipe.predict_proba(list(texts))
        classes = [str(c) for c in self.pipe.classes_]
        out = []
        for row in probs:
            i = int(row.argmax())
            out.append((classes[i], round(float(row[i]), 4)))
        return out


class TransformersBackend:
    """Pre-trained HF sentiment model (default: twitter-roberta-base-sentiment)."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline  # heavy import — only when selected

        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=256,
        )
        self.version = _env_version or "1.0.0-twt-roberta"

    def predict(self, texts):
        raw = self._pipe(list(texts))
        out = []
        for row in raw:
            best = row[0] if isinstance(row, list) else row
            label = str(best["label"]).lower()
            if label in ("pos", "positive", "label_pos"):
                label = "positive"
            elif label in ("neg", "negative", "label_neg"):
                label = "negative"
            else:
                label = "neutral"
            out.append((label, round(float(best["score"]), 4)))
        return out


_model = None


def _try_transformer():
    try:
        return TransformersBackend(HF_MODEL_NAME)
    except Exception as exc:
        print(f"transformer backend unavailable ({exc}) — falling back")
        return None


def get_model():
    global _model
    if _model is not None:
        return _model
    choice = os.getenv("MODEL_BACKEND", "auto").lower()

    if choice == "dummy":
        _model = DummyModel()
    elif choice == "sklearn":
        _model = SklearnWrapper(__import__("joblib").load(MODEL_PATH))
    elif choice == "transformer":
        _model = _try_transformer() or DummyModel()
    else:  # auto
        _model = _try_transformer()
        if _model is None:
            if MODEL_PATH.exists():
                _model = SklearnWrapper(__import__("joblib").load(MODEL_PATH))
            else:
                print(f"WARNING: no backend available — serving DummyModel")
                _model = DummyModel()

    print(f"model backend: {_model.version}")
    return _model
