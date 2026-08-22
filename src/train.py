"""PHASE 2 — Train (baseline: TF-IDF + Logistic Regression) + MLflow tracking.

Swap in DistilBERT later (train.model_type=distilbert); keep the MLflow
logging contract identical so run comparison works across both.
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def build_model(cfg: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg["tfidf_max_features"],
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("clf", LogisticRegression(C=cfg["logreg_c"], max_iter=1000)),
        ]
    )


def log_to_mlflow(cfg: dict, f1: float, acc: float) -> None:
    """Optional: only logs when MLFLOW_TRACKING_URI is set (e.g. Dagshub)."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("mlflow: MLFLOW_TRACKING_URI not set — skipping tracking")
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        with mlflow.start_run(run_name="tfidf_logreg"):
            mlflow.log_params(
                {
                    "model_type": cfg["model_type"],
                    "tfidf_max_features": cfg["tfidf_max_features"],
                    "logreg_c": cfg.get("logreg_c", cfg.get("logreg_C")),
                }
            )
            mlflow.log_metrics({"val_macro_f1": f1, "val_accuracy": acc})
            mlflow.sklearn.log_model(_MODEL, "model")
        print("mlflow: run logged")
    except Exception as exc:  # tracking must never break training
        print(f"mlflow logging skipped: {exc}")


_MODEL = None


def main() -> None:
    global _MODEL
    cfg = PARAMS["train"]
    proc = ROOT / PARAMS["data"]["processed_dir"]
    train = pd.read_csv(proc / "train.csv").dropna()
    val = pd.read_csv(proc / "val.csv").dropna()

    model = build_model(cfg)
    model.fit(train["text"], train["label"])

    from sklearn.metrics import accuracy_score, f1_score

    preds = model.predict(val["text"])
    f1 = float(f1_score(val["label"], preds, average="macro"))
    acc = float(accuracy_score(val["label"], preds))
    print(f"val macro-F1={f1:.3f}  val accuracy={acc:.3f}")

    _MODEL = model
    log_to_mlflow(cfg, f1, acc)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "metrics.json").write_text(
        json.dumps({"val_macro_f1": round(f1, 4), "val_accuracy": round(acc, 4)}, indent=2)
    )
    print(f"saved -> {models_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
