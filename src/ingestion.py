"""PHASE 1a — Ingestion: YouTube Data API v3 -> raw CSV.

Usage:  python src/ingestion.py        (needs YT_API_KEY in .env / env)
Notes:
  - Disk cache: videos already present in the CSV are never re-fetched
    (quota discipline — free tier is ~10,000 units/day).
  - Incremental: new videos are appended + deduped by comment_id.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
OUT = ROOT / PARAMS["data"]["raw_path"]
API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch_for_video(video_id: str, api_key: str, max_comments: int) -> list[dict]:
    items, token = [], None
    while len(items) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance",
        }
        if token:
            params["pageToken"] = token
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for it in data.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            items.append(
                {
                    "comment_id": it["snippet"]["topLevelComment"]["id"],
                    "text_raw": s.get("textDisplay", ""),
                    "likes": s.get("likeCount", 0),
                    "published_at": s.get("publishedAt"),
                    "video_id": video_id,
                    "author": s.get("authorDisplayName", ""),
                }
            )
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)  # be polite
    return items[:max_comments]


def main() -> None:
    api_key = os.getenv("YT_API_KEY")
    if not api_key:
        sys.exit("YT_API_KEY not set — copy .env.example to .env and fill it in.")

    existing_videos: set[str] = set()
    if OUT.exists():
        existing_videos = set(pd.read_csv(OUT)["video_id"].unique())
        print(f"cache: {OUT} already has data for {len(existing_videos)} video(s)")

    rows: list[dict] = []
    for vid in PARAMS["data"]["video_ids"]:
        if vid in existing_videos:
            print(f"skip {vid} (already cached)")
            continue
        print(f"fetching comments for {vid} ...")
        rows.extend(
            fetch_for_video(vid, api_key, PARAMS["ingestion"]["max_comments_per_video"])
        )

    if rows:
        new_df = pd.DataFrame(rows)
        old_df = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["comment_id"])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
    print(f"done: +{len(rows)} new rows -> {OUT}")


if __name__ == "__main__":
    main()
