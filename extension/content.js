// ============================================================
// content.js v2 — YouTube watch pages
// v2 changes:
//  - 3-layer fallback detection (YouTube keeps changing markup):
//      1) thread containers -> #author-text (badge next to username)
//      2) thread containers, no author found -> badge above comment text
//      3) no thread containers at all -> badge ANY #content-text on page
//  - marks the text element itself (dataset) -> no double badges after re-renders
//  - diagnostics -> popup (threads seen / queued / last scan) + console logs [ytcx]
// ============================================================

const DEFAULT_SELECTORS = {
  commentsSection: "#comments",
  // comma list is fine for querySelectorAll — first match wins per element
  commentThread:
    "ytd-comment-thread-renderer, ytd-comment-view-model, ytd-comment-renderer",
  author: "#author-text",
  text: "#content-text",
};

let SELECTORS = { ...DEFAULT_SELECTORS };
let ENABLED = true;
const BATCH_DELAY_MS = 300;
const DEBUG = true;

const processed = new Set();
let batchTimer = null;
let queue = [];
let totalQueued = 0;
let lastSig = "";

const log = (...a) => DEBUG && console.log("[ytcx]", ...a);

function setDiag(extra = {}) {
  try {
    chrome.storage.local.set({
      diag: { threads: scan.threadsFound || 0, queued: totalQueued, ts: Date.now(), ...extra },
    });
  } catch (_e) {}
}

async function loadSelectors() {
  try {
    const url = chrome.runtime.getURL("selectors.json");
    SELECTORS = { ...DEFAULT_SELECTORS, ...(await (await fetch(url)).json()) };
  } catch (_e) {
    log("selectors.json unavailable, using defaults");
  }
}

function hashId(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return "c" + Math.abs(h).toString(36);
}

function makeBadgeHost() {
  const host = document.createElement("span");
  host.className = "ytcx-badge-host";
  host.style.display = "inline-block";
  host.style.marginLeft = "6px";
  const root = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    .badge{font-size:11px;line-height:16px;padding:2px 8px;border-radius:9px;
           font-family:Roboto,Arial,sans-serif;user-select:none;cursor:default;
           font-weight:500;white-space:nowrap;display:inline-flex;align-items:center;gap:3px}
    .pos{background:#e6f6e9;color:#137333}
    .neu{background:#f1f3f4;color:#5f6368}
    .neg{background:#fde7e9;color:#c5221f}
    .unk{background:#f1f3f4;color:#9aa0a6}
    .label-text{text-transform:capitalize}`;
  const badge = document.createElement("span");
  badge.className = "badge unk";
  badge.textContent = "Analyzing…";
  root.append(style, badge);
  return host;
}

function setBadge(host, result) {
  const badge = host.shadowRoot.querySelector(".badge");
  if (!result) {
    badge.className = "badge unk";
    badge.textContent = "No result";
    badge.title = "Sentiment API did not return a result for this comment";
    return;
  }
  const cls = { positive: "pos", neutral: "neu", negative: "neg" }[result.label] || "unk";
  const emoji = { positive: "\u{1F60A}", neutral: "\u{1F610}", negative: "\u{1F621}" }[result.label] || "";
  const confPct = ((result.confidence * 100) | 0) + "%";
  const labelText = result.label.charAt(0).toUpperCase() + result.label.slice(1);
  badge.className = "badge " + cls;
  badge.textContent = (emoji ? emoji + " " : "") + labelText + " \u00B7 " + confPct;
  badge.title = result.label + " · " + confPct + " confidence · model " + result.model_version;
}

// returns true if a comment was newly queued
function processScope(scope, scopeIsTextEl) {
  const textEl = scopeIsTextEl ? scope : scope.querySelector(SELECTORS.text);
  if (!textEl || textEl.dataset.ytcx) return false;
  const text = (textEl.textContent || "").trim();
  if (!text) return false;

  textEl.dataset.ytcx = "1"; // mark the text el itself — survives markup shifts
  const id = hashId(text);
  processed.add(id);

  const host = makeBadgeHost();
  const authorEl = scopeIsTextEl ? null : scope.querySelector(SELECTORS.author);
  try {
    if (authorEl && authorEl.parentNode) {
      authorEl.insertAdjacentElement("afterend", host); // next to username
    } else {
      textEl.parentNode.insertBefore(host, textEl); // fallback: above comment text
    }
  } catch (e) {
    return false;
  }
  queue.push({ id, text, host });
  return true;
}

function scan() {
  if (!ENABLED) return;

  let threads = [];
  try {
    threads = document.querySelectorAll(SELECTORS.commentThread);
  } catch (_e) {
    threads = [];
  }
  scan.threadsFound = threads.length;

  let added = 0;
  if (threads.length) {
    for (const th of threads) if (processScope(th, false)) added++;
  } else {
    // layer-3 fallback: no known thread containers — badge every visible comment text
    for (const el of document.querySelectorAll(SELECTORS.text)) {
      if (processScope(el, true)) added++;
    }
  }

  const sig = `${threads.length}:${totalQueued}`;
  if (added || sig !== lastSig) {
    totalQueued += added;
    log(`scan: threads=${threads.length} new=${added} totalQueued=${totalQueued}`);
    lastSig = sig;
    setDiag();
  }
  if (queue.length) scheduleFlush();
}

function scheduleFlush() {
  clearTimeout(batchTimer);
  batchTimer = setTimeout(flush, BATCH_DELAY_MS);
}

async function flush() {
  const items = queue;
  queue = [];
  if (!items.length) return;
  let results = {};
  try {
    const resp = await chrome.runtime.sendMessage({
      type: "PREDICT_BATCH",
      items: items.map(({ id, text }) => ({ id, text })),
    });
    results = resp && resp.results ? resp.results : {};
  } catch (e) {
    log("prediction unavailable:", e);
    setDiag({ lastError: "content->background message failed: " + e });
  }
  for (const { id, host } of items) setBadge(host, results[id]);
}

// --- wiring -------------------------------------------------
chrome.storage.sync.get({ enabled: true }, (s) => (ENABLED = s.enabled));
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.enabled) ENABLED = changes.enabled.newValue;
});

let observer = null;
function attachObserver() {
  if (observer) observer.disconnect();
  const target = document.querySelector(SELECTORS.commentsSection) || document.body;
  observer = new MutationObserver(scan);
  observer.observe(target, { childList: true, subtree: true });
  log("observer attached to", target.id || target.tagName);
}

loadSelectors().then(() => {
  attachObserver();
  scan();
  // YouTube is an SPA — re-attach + rescan on in-site navigation
  document.addEventListener("yt-navigate-finish", () => {
    attachObserver();
    scan();
  });
});
