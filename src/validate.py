"""PHASE 1b — Data validation gate.

Fails loudly (exit 1) if the raw data is broken. This becomes a pipeline
gate: dvc repro / CI should never train on silently-corrupted data.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def main() -> None:
    path = ROOT / PARAMS["data"]["raw_path"]
    if not path.exists():
        sys.exit("FAIL: raw data missing — run `python src/ingestion.py` first.")

    df = pd.read_csv(path)
    checks = {
        f"min_rows (>={PARAMS['data']['min_rows']})": len(df) >= PARAMS["data"]["min_rows"],
        "no_null_text": df["text_raw"].notna().all(),
        "no_duplicate_comment_ids": df["comment_id"].is_unique,
        "text_len_bounds": df["text_raw"].astype(str).str.len().between(1, 5000).all(),
    }
    ok = True
    for name, passed in checks.items():
        print(("PASS " if passed else "FAIL ") + name)
        ok &= bool(passed)
    if not ok:
        sys.exit(1)
    print(f"validation OK — {len(df)} rows, {df['video_id'].nunique()} video(s)")


if __name__ == "__main__":
    main()
