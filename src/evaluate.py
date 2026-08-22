"""PHASE 2c — Evaluation GATE.

Writes models/eval_metrics.json and exits non-zero if macro-F1 is below
params.evaluate.min_macro_f1. The weekly retrain job reuses this gate:
a new model is only promoted if it beats the threshold/current model.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def main() -> None:
    proc = ROOT / PARAMS["data"]["processed_dir"]
    test = pd.read_csv(proc / "test.csv").dropna()
    model = joblib.load(ROOT / PARAMS["serve"]["model_path"])

    preds = model.predict(test["text"])
    f1 = float(f1_score(test["label"], preds, average="macro"))
    acc = float(accuracy_score(test["label"], preds))
    report = classification_report(test["label"], preds, output_dict=True)

    print(f"test macro-F1={f1:.3f}  accuracy={acc:.3f}")
    print(classification_report(test["label"], preds))

    out = ROOT / "models" / "eval_metrics.json"
    out.write_text(
        json.dumps({"test_macro_f1": round(f1, 4), "test_accuracy": round(acc, 4), "report": report}, indent=2)
    )

    threshold = PARAMS["evaluate"]["min_macro_f1"]
    if f1 < threshold:
        sys.exit(f"GATE FAILED: test macro-F1 {f1:.3f} < required {threshold}")
    print(f"GATE PASSED (>= {threshold})")


if __name__ == "__main__":
    main()
