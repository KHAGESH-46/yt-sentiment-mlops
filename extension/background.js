// ============================================================
// background.js v0.2.1 — service worker
// Fixes the 422 bug you found on YouTube:
//  - batches are SPLIT client-side into chunks of ≤50 (API contract cap)
//  - text is TRUNCATED to 500 chars client-side (API contract cap)
//  - retry only on network errors / 5xx — never on 4xx (a 422 retry
//    would fail identically and just double the request)
// ============================================================

const CACHE_KEY = "pred_cache";
const MAX_CACHE = 5000;
const MAX_BATCH = 50;   // must match api/schemas.py
const MAX_TEXT = 500;   // must match api/schemas.py

function hashId(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return "c" + Math.abs(h).toString(36);
}

const keyOf = (text) => "h" + hashId(text.toLowerCase().trim());

async function getApiBase() {
  const { apiBase } = await chrome.storage.sync.get({ apiBase: "http://localhost:8000" });
  return apiBase.replace(/\/+$/, "");
}

async function setLastError(msg) {
  try {
    await chrome.storage.local.set({ lastError: msg || "" });
  } catch (_e) {}
}

async function cacheGet(keys) {
  const store = (await chrome.storage.local.get({ [CACHE_KEY]: {} }))[CACHE_KEY];
  return keys.map((k) => store[k] || null);
}

async function cachePut(pairs) {
  const got = await chrome.storage.local.get({ [CACHE_KEY]: {} });
  const store = got[CACHE_KEY];
  for (const { key, value } of pairs) store[key] = { ...value, ts: Date.now() };
  const entries = Object.entries(store);
  if (entries.length > MAX_CACHE) {
    entries.sort((a, b) => a[1].ts - b[1].ts);
    for (const [k] of entries.slice(0, entries.length - MAX_CACHE)) delete store[k];
  }
  await chrome.storage.local.set({ [CACHE_KEY]: store });
}

async function callApi(base, chunk) {
  const res = await fetch(base + "/v1/predict-batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      comments: chunk.map(({ id, text }) => ({ id, text: String(text).slice(0, MAX_TEXT) })),
    }),
  });
  if (!res.ok) {
    const err = new Error("API HTTP " + res.status);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  const map = {};
  for (const r of data.results || []) map[r.id] = r;
  return map;
}

function isRetryable(err) {
  return !err.status || err.status >= 500; // network failure or server error only
}

async function callApiWithRetry(base, chunk) {
  try {
    return await callApi(base, chunk);
  } catch (e) {
    if (!isRetryable(e)) throw e;
    return await callApi(base, chunk);
  }
}

// Sends ANY number of items, respecting the ≤50 contract via chunking.
async function fetchAll(base, items) {
  const results = {};
  let lastErr = null;
  let ok = 0, failed = 0;
  for (let i = 0; i < items.length; i += MAX_BATCH) {
    try {
      const map = await callApiWithRetry(base, items.slice(i, i + MAX_BATCH));
      Object.assign(results, map);
      ok++;
    } catch (e) {
      failed++;
      lastErr = e;
    }
  }
  if (failed && !ok) throw lastErr;               // total failure -> report it
  else if (failed) await setLastError(             // partial failure -> note it
    `${failed} of ${ok + failed} batch(es) failed (${lastErr}) — those badges show "–"`);
  else await setLastError("");
  return results;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== "PREDICT_BATCH") return;
  (async () => {
    try {
      const base = await getApiBase();
      const cached = await cacheGet(msg.items.map((i) => keyOf(i.text)));
      const misses = msg.items.filter((_i, idx) => !cached[idx]);

      let fetched = {};
      if (misses.length) {
        try {
          fetched = await fetchAll(base, misses);
        } catch (e) {
          fetched = {};
          await setLastError(`API unreachable at ${base} (${e})`);
        }
      }

      const results = {};
      const toCache = [];
      msg.items.forEach((item, idx) => {
        const r = cached[idx] || fetched[item.id] || null;
        results[item.id] = r;
        if (!cached[idx] && r) toCache.push({ key: keyOf(item.text), value: r });
      });
      if (toCache.length) await cachePut(toCache);

      const { analyzed } = await chrome.storage.local.get({ analyzed: 0 });
      await chrome.storage.local.set({ analyzed: analyzed + msg.items.length });

      sendResponse({ results });
    } catch (e) {
      await setLastError("background error: " + e);
      sendResponse({ results: {}, error: String(e) });
    }
  })();
  return true;
});

// Model-version handshake at startup -> popup + version display.
(async () => {
  try {
    const base = await getApiBase();
    const res = await fetch(base + "/v1/model/current");
    const data = await res.json();
    await chrome.storage.local.set({ modelVersion: data.model_version });
    await setLastError("");
  } catch (e) {
    await chrome.storage.local.set({ modelVersion: "unknown" });
    await setLastError(`startup handshake failed (${e}) — is "API base" set in the popup?`);
  }
})();
