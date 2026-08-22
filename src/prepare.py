"""PHASE 1c — Prepare: clean -> weak-label -> stratified split.

LABELING NOTE (important): `weak_label` is a keyword/emoji heuristic that
gives NOISY labels. It exists so the pipeline runs end-to-end from day one.
For a real model, swap in a labeled Kaggle dataset (primary source) and use
API-scraped data mainly for inference/drift. Update evaluate.min_macro_f1
afterwards (0.40 -> 0.75).
"""
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

POSITIVE = {
    "good", "great", "love", "loved", "awesome", "best", "amazing", "excellent",
    "nice", "helpful", "perfect", "wonderful", "brilliant", "fire", "thanks",
    "🔥", "❤", "😍", "😊", "👍",
}
NEGATIVE = {
    "bad", "worst", "hate", "hated", "trash", "boring", "waste", "awful",
    "dislike", "stupid", "scam", "garbage", "terrible", "cringe", "disgusting",
    "💀", "😡", "👎",
}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = r"\w+|[🔥❤😍😊👍💀😡👎]"


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, collapse whitespace. KEEP emojis (they carry signal)."""
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def weak_label(text: str) -> str:
    tokens = set(re.findall(EMOJI_RE, str(text)))
    pos, neg = len(tokens & POSITIVE), len(tokens & NEGATIVE)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def main() -> None:
    prep = PARAMS["prepare"]
    raw = pd.read_csv(ROOT / PARAMS["data"]["raw_path"])

    df = raw.copy()
    df["text"] = df["text_raw"].map(clean_text)
    df = df[df["text"].str.len().between(prep["min_text_len"], prep["max_text_len"])]
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["text"].map(weak_label)
    df = df[["text", "label", "video_id", "likes"]]
    df = df.dropna(subset=["text", "label"])

    seed = PARAMS["train"]["random_state"]
    train, tmp = train_test_split(
        df,
        test_size=prep["val_size"] + prep["test_size"],
        random_state=seed,
        stratify=df["label"],
    )
    rel_test = prep["test_size"] / (prep["val_size"] + prep["test_size"])
    val, test = train_test_split(tmp, test_size=rel_test, random_state=seed, stratify=tmp["label"])

    out = ROOT / PARAMS["data"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train[["text", "label"]].to_csv(out / "train.csv", index=False)
    val[["text", "label"]].to_csv(out / "val.csv", index=False)
    test[["text", "label"]].to_csv(out / "test.csv", index=False)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("label distribution (train):")
    print(train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
