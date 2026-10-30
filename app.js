/**
 * EuroSAT Decision Demo -- app.js
 *
 * Runs the committed ResNet-18 model (exported to ONNX) in the browser via
 * onnxruntime-web, loaded entirely from this local folder. No network calls at runtime.
 *
 * Preprocessing mirrors the training/validation transform:
 *   - resize to 64x64
 *   - convert to RGB float tensor [1, 3, 64, 64]
 *   - normalize with ImageNet mean/std
 *
 * Decision rule: answered if max softmax probability >= 0.50 (frozen threshold),
 * otherwise "abstained (route to review)".
 */

const CLASS_NAMES = [
  "AnnualCrop",
  "Forest",
  "HerbaceousVegetation",
  "Highway",
  "Industrial",
  "Pasture",
  "PermanentCrop",
  "Residential",
  "River",
  "SeaLake",
];

const FROZEN_THRESHOLD = 0.50;
const MODEL_PATH = "model.onnx";

// Debug hook: stores softmax results keyed by example file, for verification only.
window.__debugResults = window.__debugResults || {};
function storeDebugResult(file, probs, pred) {
  window.__debugResults[file] = { probs: probs.map(Number), pred: pred };
}

let session = null;
let modelResolve = null;
const modelReady = new Promise((resolve) => { modelResolve = resolve; });

// ----- element helpers -----
function el(id) {
  return document.getElementById(id);
}

// ----- helpers -----
function setStatus(msg, kind) {
  const s = el("status");
  if (s) {
    s.textContent = msg;
    s.className = "status " + (kind || "");
  }
}

function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function top3(probs) {
  const indexed = probs.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => b.p - a.p);
  return indexed.slice(0, 3);
}

function formatPct(p) {
  return (p * 100).toFixed(1) + "%";
}

// ----- image preprocessing -----
function loadAndPreprocess(fileOrUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const resized = img.width !== 64 || img.height !== 64;
        const canvas = document.createElement("canvas");
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, 64, 64);
        const rn = el("resizeNotice");
        if (rn) {
          if (resized) { rn.style.display = "block"; } else { rn.style.display = "none"; }
        }
        const imageData = ctx.getImageData(0, 0, 64, 64);
        const data = imageData.data;
        const tensor = new Float32Array(1 * 3 * 64 * 64);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        const CHANNEL_STRIDE = 64 * 64; // 4096 pixels per channel (NCHW layout)
        for (let y = 0; y < 64; y++) {
          for (let x = 0; x < 64; x++) {
            const off = (y * 64 + x) * 4;
            const r = data[off] / 255;
            const g = data[off + 1] / 255;
            const b = data[off + 2] / 255;
            const pixel_idx = y * 64 + x;
            tensor[pixel_idx] = (r - mean[0]) / std[0];          // channel 0: R
            tensor[CHANNEL_STRIDE + pixel_idx] = (g - mean[1]) / std[1];   // channel 1: G
            tensor[2 * CHANNEL_STRIDE + pixel_idx] = (b - mean[2]) / std[2];   // channel 2: B
          }
        }
        resolve(tensor);
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    if (fileOrUrl instanceof File) {
      img.src = URL.createObjectURL(fileOrUrl);
    } else {
      img.src = fileOrUrl;
    }
  });
}

// ----- run inference -----
async function runInference(tensor) {
  if (!session) throw new Error("Model not loaded yet");
  const input = new ort.Tensor("float32", tensor, [1, 3, 64, 64]);
  const outputs = await session.run({ input: input });
  const logits = outputs.logits.data;
  return Array.from(logits);
}

// ----- UI helpers for viewer -----
function decisionClass(answered) {
  return answered ? "answered" : "abstained";
}
function decisionText(pred, conf, answered) {
  if (answered) return "Answered &mdash; " + pred;
  return "Abstained (route to review) &mdash; max " + formatPct(conf) + " < " + FROZEN_THRESHOLD;
}

// ----- build viewer grid -----
async function buildViewer(manifest) {
  await modelReady;
  const viewerStatusEl = el("viewerStatus");
  const examplesEl = el("examples");

  if (viewerStatusEl) viewerStatusEl.textContent = "Loading examples &hellip;";
  if (examplesEl) examplesEl.innerHTML = "";

  let matches = 0, mismatches = 0, answered = 0, abstained = 0;
  let total = manifest.examples.length;

  for (const ex of manifest.examples) {
    const card = document.createElement("div");
    card.className = "example-card";
    card.innerHTML = `
      <img class="example-thumb" src="examples/${ex.example_file}" alt="${ex.class}">
      <div class="example-label">${ex.class}</div>
      <div class="example-pred">--</div>
      <div class="example-prob">--</div>
      <div class="example-decision ${decisionClass(false)}">--</div>
      <div class="example-badge">--</div>
    `;
    if (examplesEl) examplesEl.appendChild(card);

    try {
      const tensor = await loadAndPreprocess("examples/" + ex.example_file);
      const logits = await runInference(tensor);
      const probs = softmax(logits);
      const best = top3(probs)[0];
      const pred = CLASS_NAMES[best.i];
      const conf = best.p;
      const isAnswered = conf >= FROZEN_THRESHOLD;
      const match = pred === ex.class;

      // Debug hook
      try { storeDebugResult(ex.example_file, probs, pred); } catch (e) {}

      if (isAnswered) answered++; else abstained++;
      if (match) matches++; else mismatches++;

      const predEl = card.querySelector(".example-pred");
      const probEl = card.querySelector(".example-prob");
      const decEl = card.querySelector(".example-decision");
      const badgeEl = card.querySelector(".example-badge");
      if (predEl) predEl.textContent = pred;
      if (probEl) probEl.textContent = formatPct(conf);
      if (decEl) {
        decEl.textContent = decisionText(pred, conf, isAnswered);
        decEl.className = "example-decision " + decisionClass(isAnswered);
      }
      if (badgeEl) {
        badgeEl.textContent = match ? "MATCH" : "MISMATCH";
        badgeEl.className = "example-badge " + (match ? "match" : "mismatch");
      }
      if (match) card.classList.add("match");
      else card.classList.add("mismatch");
      if (!isAnswered) card.classList.add("abstained");
    } catch (e) {
      console.error("Error on " + ex.example_file, e);
      const predEl = card.querySelector(".example-pred");
      if (predEl) predEl.textContent = "error";
      const decEl = card.querySelector(".example-decision");
      if (decEl) decEl.textContent = "error";
      const badgeEl = card.querySelector(".example-badge");
      if (badgeEl) badgeEl.textContent = "error";
    }

    if (viewerStatusEl) viewerStatusEl.textContent = "Loaded examples &hellip;";
  }

  if (viewerStatusEl) viewerStatusEl.textContent = "Loaded " + total + " examples.";
  if (examplesEl) {
    // Re-attach click handlers after all cards are rendered
    const cards = examplesEl.querySelectorAll(".example-card");
    cards.forEach((card, i) => {
      card.addEventListener("click", () => {
        const ex = manifest.examples[i];
        if (!ex) return;
        const imgEl = card.querySelector(".example-thumb");
        if (!imgEl) return;
        showResult(ex.example_file, imgEl.src, ex.class);
      });
    });
  }

  // Load exploratory abstention-by-class table (from the committed analysis, not from new evaluation).
  fetch("analysis/abstention_by_class.csv")
    .then((r) => r.text())
    .then((text) => {
      const lines = text.trim().split("\n");
      const header = lines[0].split(",");
      const rows = [];
      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i].split(",");
        const obj = {};
        header.forEach((h, j) => { obj[h] = vals[j]; });
        rows.push(obj);
      }
      const expl = el("explTable");
      const explStatus = el("explStatus");
      if (expl) {
        let html = "<table class=\"expl-table\"><thead><tr>";
        header.forEach((h) => { html += "<th>" + h + "</th>"; });
        html += "</tr></thead><tbody>";
        rows.forEach((row) => {
          html += "<tr>";
          header.forEach((h) => { html += "<td>" + row[h] + "</td>"; });
          html += "</tr>";
        });
        html += "</tbody></table>";
        expl.innerHTML = html;
      }
      if (explStatus) explStatus.textContent = "Exploratory table loaded (" + rows.length + " classes).";
    })
    .catch((e) => {
      console.warn("Could not load exploratory table:", e);
      const explStatus = el("explStatus");
      if (explStatus) explStatus.textContent = "Exploratory table unavailable.";
    });
}

// ----- show result for a given example -----
function showResult(fileName, imgSrc, trueLabel) {
  const thumb = el("resultThumb");
  const predEl = el("resultPred");
  const probEl = el("resultProb");
  const probFill = el("probFill");
  const probText = el("probText");
  const decEl = el("resultDecision");
  const badgeEl = el("resultBadge");
  const matchNote = el("resultMatchNote");
  const resultCard = el("resultCard");

  if (thumb) thumb.src = imgSrc;
  if (!resultCard) return;

  const dr = window.__debugResults && window.__debugResults[fileName];
  if (!dr) return;

  const probs = dr.probs;
  const best = top3(probs)[0];
  const pred = CLASS_NAMES[best.i];
  const conf = best.p;
  const isAnswered = conf >= FROZEN_THRESHOLD;
  const match = pred === trueLabel;

  if (predEl) predEl.textContent = pred;
  if (probEl) probEl.textContent = formatPct(conf);
  if (probFill) probFill.style.width = (conf * 100) + "%";
  if (probText) probText.textContent = formatPct(conf);
  if (decEl) {
    decEl.textContent = decisionText(pred, conf, isAnswered);
    decEl.className = "example-decision " + decisionClass(isAnswered);
  }
  if (badgeEl) {
    badgeEl.textContent = match ? "MATCH" : "MISMATCH";
    badgeEl.className = "example-badge " + (match ? "match" : "mismatch");
  }
  if (matchNote) {
    matchNote.textContent = match ? "Prediction matches true label." : "Prediction differs from true label.";
    matchNote.className = "muted " + (match ? "" : "");
  }

  resultCard.style.display = "block";
}

// ----- single upload -----
function displayPrediction(logits, srcLabel) {
  const probs = softmax(logits);
  const top = top3(probs)[0];
  const best = top[0];

  const tc = el("resultPred");
  const tfc = el("resultProb");
  if (tc) tc.textContent = CLASS_NAMES[best.i];
  if (tfc) tfc.textContent = formatPct(best.p) + " confidence";
  const answered = best.p >= FROZEN_THRESHOLD;
  const decEl = el("resultDecision");
  if (decEl) {
    decEl.textContent = answered
      ? "Answered &mdash; " + CLASS_NAMES[best.i]
      : "Abstained (route to review) &mdash; max " + formatPct(best.p) + " < " + FROZEN_THRESHOLD;
    decEl.className = "example-decision " + (answered ? "example-answered" : "example-abstained");
  }
  const ri = el("resultThumb");
  const pi = el("previewImg");
  if (ri && pi) { ri.src = pi.src; }

  const probFill = el("probFill");
  const probText = el("probText");
  if (probFill) probFill.style.width = (best.p * 100) + "%";
  if (probText) probText.textContent = formatPct(best.p);

  const resultCard = el("resultCard");
  if (resultCard) resultCard.style.display = "block";

  setStatus("Prediction complete for " + srcLabel + ". " + (answered ? "Answered" : "Abstained") + ".", answered ? "ok" : "");
}

// ----- model loading -----
async function loadModel() {
  const ls = el("loadStatus");
  const se = el("status");
  try {
    if (ls) { ls.textContent = "Loading ONNX runtime & model&hellip;"; }
    if (typeof ort === "undefined") {
      throw new Error("ort (onnxruntime-web) is not defined. Check vendor/ort.wasm.min.js loaded.");
    }
    session = await ort.InferenceSession.create(MODEL_PATH);
    if (ls) { ls.textContent = "Model loaded. Ready."; ls.className = "status ok"; }
    if (se) { se.textContent = ""; }
  } catch (e) {
    if (ls) { ls.textContent = "Failed to load model."; ls.className = "status err"; }
    console.error(e);
    if (se) {
      se.textContent = "Error: " + (e && e.message ? e.message : String(e));
      se.className = "status err";
    }
  } finally {
    if (modelResolve) modelResolve();
  }
}

loadModel();

// ----- load examples manifest and build viewer -----
fetch("examples_manifest.json")
  .then((r) => r.json())
  .then((manifest) => {
    buildViewer(manifest);
  })
  .catch((e) => {
    console.warn("Could not load examples manifest:", e);
    const vs = el("viewerStatus");
    if (vs) vs.textContent = "Could not load examples manifest.";
  });

// ----- file upload -----
const fileInput = el("file");
if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const pi = el("previewImg");
    if (pi) { pi.src = URL.createObjectURL(file); pi.alt = file.name; }
    const rn = el("resizeNotice");
    if (rn) rn.style.display = "none";
    setStatus("Uploaded " + file.name + ". Running inference&hellip;");
    const sl = el("singleLoading");
    if (sl) sl.style.display = "flex";
    try {
      const tensor = await loadAndPreprocess(file);
      const logits = await runInference(tensor);
      displayPrediction(logits, file.name);
    } catch (err) {
      console.error(err);
      setStatus("Error: " + (err.message || String(err)), "err");
      if (sl) sl.style.display = "none";
    }
  });
}
