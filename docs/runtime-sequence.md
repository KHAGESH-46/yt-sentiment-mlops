# Runtime Sequence — comment → badge (explain the debounce/batch behavior)

```mermaid
sequenceDiagram
    autonumber
    participant U as User (scrolls)
    participant C as content.js<br/>(YouTube tab)
    participant B as background.js<br/>(service worker)
    participant A as FastAPI<br/>(Docker, cloud)
    participant M as Model<br/>(joblib/ONNX)
    participant S as SQLite<br/>(logs + feedback)

    U->>C: new comments render on screen
    C->>C: MutationObserver fires<br/>dedupe (text hash) · debounce 300 ms
    C->>B: PREDICT_BATCH [{id, text}, ...]
    B->>B: chrome.storage cache lookup
    B->>A: POST /v1/predict-batch (cache misses only)
    A->>A: rate-limit check · pydantic caps (≤50 × 500 chars)
    A->>A: server LRU cache lookup
    A->>M: inference on remaining misses
    M-->>A: labels + confidence
    A->>S: log predictions (drift fuel)
    A-->>B: results + model_version (+ cache_hits)
    B->>B: merge cache + API results · update local cache
    B-->>C: results map
    C->>C: inject badge next to author (Shadow DOM)
    Note over C,S: hover badge → 👍/👎 → POST /v1/feedback → feedback store → weekly retrain
```

Why it feels real-time: the observer catches comments the moment YouTube renders
them, batching means one API call per scroll-burst (not one per comment), and the
two-level cache means repeat comments cost zero inference.
