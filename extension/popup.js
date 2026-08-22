const $ = (id) => document.getElementById(id);

$("ver").textContent = "v" + chrome.runtime.getManifest().version;

chrome.storage.sync.get({ enabled: true, apiBase: "http://localhost:8000" }, (s) => {
  $("enabled").checked = s.enabled;
  $("apiBase").value = s.apiBase;
});
$("enabled").addEventListener("change", (e) =>
  chrome.storage.sync.set({ enabled: e.target.checked })
);
$("apiBase").addEventListener("change", (e) =>
  chrome.storage.sync.set({ apiBase: e.target.value.trim() })
);

$("testApi").addEventListener("click", async () => {
  const base = $("apiBase").value.trim().replace(/\/+$/, "");
  $("testResult").textContent = "testing…";
  $("testResult").className = "muted";
  try {
    const res = await fetch(base + "/health");
    const data = await res.json();
    $("testResult").textContent = `✔ API OK — model ${data.model_version}`;
    $("testResult").className = "ok";
  } catch (e) {
    $("testResult").textContent = "✖ cannot reach API: " + e.message;
    $("testResult").className = "err";
  }
});

function refreshDiag() {
  chrome.storage.local.get({ analyzed: 0, modelVersion: "unknown", diag: null, lastError: "" }, (s) => {
    $("analyzed").textContent = s.analyzed;
    $("modelVersion").textContent = s.modelVersion;
    $("diagQueued").textContent = s.diag ? s.diag.queued : 0;
    $("lastError").textContent = s.lastError || "none";
  });
}
refreshDiag();
chrome.storage.onChanged.addListener((_c, area) => area === "local" && refreshDiag());

$("clearCache").addEventListener("click", async () => {
  await chrome.storage.local.remove("pred_cache");
  $("clearCache").textContent = "Cleared!";
});
