"""Test the pipeline on REAL YouTube comments — no API key needed.

Uses `youtube-comment-downloader` (YouTube's internal innertube API) to fetch
real comments, sends them to the live sentiment API in batches of 50, and
writes:
  - data/real/real_comments_scored.csv   (full results)
  - reports/real_comments_test.html      (visual report = what the extension shows)
"""
import html
import json
import time
from collections import Counter
from pathlib import Path

import requests
import yaml
from youtube_comment_downloader import YoutubeCommentDownloader

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
API = "http://localhost:8000"
PER_VIDEO = 100

VIDEOS = {v: v for v in PARAMS["data"]["video_ids"]}


def fetch_comments() -> list[dict]:
    dl = YoutubeCommentDownloader()
    out = []
    for vid in VIDEOS:
        print(f"fetching: {vid} ...")
        n = 0
        try:
            for c in dl.get_comments_from_url(f"https://www.youtube.com/watch?v={vid}"):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                out.append(
                    {
                        "video_id": vid,
                        "comment_id": c.get("cid", ""),
                        "text": text,
                        "likes": c.get("likes", 0) or 0,
                        "author": c.get("author", ""),
                    }
                )
                n += 1
                if n >= PER_VIDEO:
                    break
        except Exception as exc:
            print(f"  !! {vid} failed: {exc}")
        print(f"  got {n} comments")
    return out


def score_via_api(comments: list[dict]) -> list[dict]:
    results = []
    for i in range(0, len(comments), 50):
        batch = comments[i : i + 50]
        payload = {
            "comments": [
                {"id": f"i{i + j}", "text": c["text"][:500]} for j, c in enumerate(batch)
            ]
        }
        t0 = time.perf_counter()
        resp = requests.post(f"{API}/v1/predict-batch", json=payload, timeout=60)
        dt = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        by_id = {r["id"]: r for r in resp.json()["results"]}
        for j, c in enumerate(batch):
            r = by_id[f"i{i + j}"]
            results.append({**c, "label": r["label"], "confidence": r["confidence"]})
        print(
            f"batch {i // 50 + 1}: {len(batch)} comments scored in {dt:.0f} ms "
            f"({dt / len(batch):.1f} ms/comment)"
        )
    return results


BADGE = {
    "positive": ("#e6f6e9", "#137333", "😊"),
    "neutral": ("#f1f3f4", "#5f6368", "😐"),
    "negative": ("#fde7e9", "#c5221f", "😡"),
}


def badge_html(label: str, conf: float) -> str:
    bg, fg, emoji = BADGE[label]
    mark = "?" if conf < 0.6 else ""
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 8px;border-radius:9px;'
        f'font-size:12px;white-space:nowrap">{emoji}{mark} {label} '
        f'<span style="opacity:.65">{conf:.0%}</span></span>'
    )


def build_report(rows: list[dict], out_path: Path) -> None:
    dist = Counter(r["label"] for r in rows)
    total = len(rows)
    avg_conf = sum(r["confidence"] for r in rows) / max(total, 1)
    low_conf = sum(1 for r in rows if r["confidence"] < 0.6)
    top = sorted(rows, key=lambda r: -r["likes"])[:20]
    uncertain = sorted(rows, key=lambda r: r["confidence"])[:5]
    try:
        model_version = requests.get(f"{API}/v1/model/current", timeout=10).json()["model_version"]
    except Exception:
        model_version = "unknown"

    def comment_html(r: dict) -> str:
        return (
            f'<div style="display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #eee">'
            f'<div style="flex:1"><b style="font-size:13px">{html.escape(r["author"] or "unknown")}'
            f'</b> <span style="color:#999;font-size:11px">👍 {r["likes"]}</span>'
            f'<div style="font-size:13px;margin-top:3px">{html.escape(r["text"][:300])}</div></div>'
            f'<div style="align-self:center">{badge_html(r["label"], r["confidence"])}</div></div>'
        )

    bars = ""
    colors = {"positive": "#137333", "neutral": "#5f6368", "negative": "#c5221f"}
    for lbl in ("positive", "neutral", "negative"):
        pct = 100 * dist.get(lbl, 0) / max(total, 1)
        bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<div style="width:70px;font-size:13px">{lbl}</div>'
            f'<div style="flex:1;background:#f1f3f4;border-radius:6px">'
            f'<div style="width:{pct:.0f}%;background:{colors[lbl]};border-radius:6px;'
            f'height:14px"></div></div>'
            f'<div style="width:90px;font-size:12px;color:#555">{dist.get(lbl, 0)} ({pct:.0f}%)</div></div>'
        )

    rows_html = "".join(comment_html(r) for r in top)
    unc_html = "".join(comment_html(r) for r in uncertain)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Real YouTube Comments — Sentiment Test</title></head>
<body style="font-family:Roboto,Arial,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#202124">
<h1 style="font-size:22px">Real YouTube Comments — Sentiment Test</h1>
<p style="color:#5f6368;font-size:13px">{total} real comments scored by the live API
(model v{rows[0]["confidence"] and "0.1.0"}, trained on synthetic data — see note below)</p>

<div style="background:#f8f9fa;border-radius:10px;padding:14px 16px;margin:14px 0">
<h2 style="font-size:15px;margin:0 0 8px">Distribution</h2>{bars}
<div style="font-size:12px;color:#5f6368;margin-top:8px">
avg confidence <b>{avg_conf:.0%}</b> · low-confidence (&lt;60%, shown with "?") <b>{low_conf}</b> of {total}</div></div>

<h2 style="font-size:15px">Top-liked comments — as the extension would show them</h2>
<div>{rows_html}</div>

<h2 style="font-size:15px;margin-top:20px">Least confident predictions (the "?" badges)</h2>
<div>{unc_html}</div>

<div style="background:#fef7e0;border-radius:10px;padding:12px 16px;margin:18px 0;font-size:13px">
<b>Honest note:</b> this model was trained on 3,000 synthetic English comments.
Real YouTube comments include sarcasm, Hinglish, and spam — expect weaker accuracy here.
This baseline is exactly why the next step is training on a real labeled dataset.</div>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


def main() -> None:
    comments = fetch_comments()
    if not comments:
        raise SystemExit("No comments fetched — YouTube unreachable from sandbox?")
    print(f"total fetched: {len(comments)}")
    scored = score_via_api(comments)

    import csv

    out_csv = ROOT / "data" / "real" / "real_comments_scored.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)

    build_report(scored, ROOT / "reports" / "real_comments_test.html")
    dist = Counter(r["label"] for r in scored)
    print("done. distribution:", dict(dist))
    print(f"saved: {out_csv} + reports/real_comments_test.html")


if __name__ == "__main__":
    main()
