# Architecture Diagrams (Mermaid) — Option A

> Paste these into any Mermaid renderer (GitHub READMEs render `.mermaid`/```` ```mermaid ```` blocks natively).
> Diagram 1 = full system · Diagram 2 = training pipeline (dvc.yaml stages) · Diagram 3 = closed retrain loop.

---

## 1. Main System Architecture

```mermaid
flowchart LR

    %% ================= CLIENT =================
    subgraph EXT["🖥️ Chrome Extension — Manifest V3"]
        direction TB
        OBS["content.js<br/>MutationObserver captures<br/>new comment nodes"]
        SW["background.js — service worker<br/>dedupe · debounce 300ms · batch ≤50"]
        LC[("chrome.storage.local<br/>LRU cache — text hash → result")]
        BADGE["Shadow DOM badge injection<br/>😊 😐 😡 + 👍/👎 feedback"]
        POP["popup<br/>on/off · API URL · model version"]
    end

    YT["📺 YouTube watch page<br/>comments lazy-load on scroll"]

    %% ================= BACKEND =================
    subgraph SRV["☁️ Backend — FastAPI in Docker (Render/EC2)"]
        direction TB
        PB["POST /v1/predict-batch<br/>rate-limited per IP · caps: 50 items, 500 chars"]
        RC[("Redis / in-mem LRU<br/>text hash → label")]
        ONNX["ONNX model<br/>(Production build baked into image)"]
        FB["POST /v1/feedback"]
        MV["GET /v1/model/current"]
        LOGS[("SQLite / Postgres<br/>prediction logs + feedback")]
    end

    %% ================= MLOPS =================
    subgraph OPS["🔧 MLOps Backbone"]
        direction TB
        YTAPI["YouTube Data API v3"]
        DVC["DVC + dvc.yaml<br/>ingest → validate → prepare"]
        TRN["train.py / evaluate.py<br/>MLflow-tracked · metric gate"]
        REG["MLflow Model Registry<br/>staging → Production"]
        GHA["GitHub Actions<br/>ci.yml · deploy.yml · retrain.yml"]
        EVD["Evidently<br/>drift report (weekly)"]
        ISS["auto GitHub issue on drift"]
    end

    %% ---- runtime flows (user watching YouTube) ----
    YT --> OBS
    OBS --> SW
    SW <--> LC
    SW -->|"POST batch"| PB
    PB --> RC
    RC -.->|"cache miss → infer"| ONNX
    PB --> LOGS
    BADGE -.->|"👍/👎"| FB
    FB --> LOGS
    SW -->|"startup version check"| MV
    POP -.-> SW

    %% ---- data / build-time flows ----
    YTAPI --> DVC
    DVC --> TRN
    TRN --> REG
    REG -.->|"bake into Docker image"| ONNX
    GHA -.->|"runs"| TRN
    GHA -.->|"deploys"| SRV

    %% ---- monitoring flows ----
    LOGS --> EVD
    EVD --> ISS
    ISS -.-> GHA
```

---

## 2. Training / Data Pipeline

> Each box = one stage in `dvc.yaml` (with `deps`, `params`, `outs`). `dvc repro` runs the whole chain.

```mermaid
flowchart LR
    A["ingestion.py<br/>YouTube API → raw.csv<br/>(quota-budgeted, disk-cached)"] --> B["validate.py<br/>row count · nulls · dupes"]
    B --> C["prepare.py<br/>clean · label · stratified split"]
    C --> D["train.py<br/>TF-IDF baseline → DistilBERT"]
    D --> E["evaluate.py<br/>gate: macro-F1 ≥ threshold"]
    E --> F["export_onnx.py"]
    F --> G[("MLflow Registry<br/>Production")]

    D -.-> M["MLflow Tracking (Dagshub)"]
    C -.-> V["DVC push — versioned data (Dagshub)"]
```

---

## 3. Closed Retrain Loop (flagship feature)

```mermaid
flowchart TB
    L[("prediction logs")] --> E["Evidently drift report<br/>(weekly job)"]
    U["👍/👎 user feedback<br/>via /v1/feedback"] --> S[("feedback store")]
    E -->|"drift > threshold"| I["auto GitHub issue"]
    E -.-> R
    S --> R["retrain.yml — weekly cron<br/>merge training data + corrections"]
    R --> G{"gate: new model<br/>beats Production?"}
    G -->|"yes"| P["MLflow Registry<br/>promote to Production"]
    G -->|"no"| X["stop — keep current model"]
    P --> T["tag release → deploy.yml<br/>build + deploy new image"]
    T --> A2["API serves new model_version"]
    A2 --> X2["extension polls /v1/model/current<br/>badges show updated version"]
```

---

## How to read Diagram 1 (3-minute demo script)

1. **Runtime (solid arrows, top):** user scrolls → content script catches new comments → service worker batches them → API checks cache → cache miss hits the ONNX model → label + confidence return → badge injected next to username. Every prediction is logged.
2. **Build-time (middle):** YouTube API → DVC-versioned pipeline → MLflow-tracked training → gated evaluation → Production model in registry → CI bakes it into the Docker image and deploys.
3. **Monitoring (bottom):** logs feed Evidently weekly → drift over threshold opens a GitHub issue → retrain workflow can consume logs + user feedback → only promotes if it beats the current model.
