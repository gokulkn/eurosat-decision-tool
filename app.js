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
        let idx = 0;
        for (let y = 0; y < 64; y++) {
          for (let x = 0; x < 64; x++) {
            const off = (y * 64 + x) * 4;
            const r = data[off] / 255;
            const g = data[off + 1] / 255;
            const b = data[off + 2] / 255;
            tensor[idx++] = (r - mean[0]) / std[0];
            tensor[idx++] = (g - mean[1]) / std[1];
            tensor[idx++] = (b - mean[2]) / std[2];
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
  const viewerGridEl = el("viewerGrid");
  const summaryEl = el("summary");
  const viewerNoteEl = el("viewerNote");

  if (viewerStatusEl) viewerStatusEl.textContent = "Loading examples &hellip;";
  if (viewerGridEl) viewerGridEl.innerHTML = "";
  if (summaryEl) summaryEl.textContent = "";
  if (viewerNoteEl) viewerNoteEl.textContent = "";

  let matches = 0, mismatches = 0, answered = 0, abstained = 0;
  let total = manifest.examples.length;

  for (const ex of manifest.examples) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <img class="thumb" src="examples/${ex.example_file}" alt="${ex.class}">
      <div class="true-label">True: ${ex.class}</div>
      <div class="pred-label">Predicted: <span id="pred-${ex.example_file}">--</span></div>
      <div class="prob">Confidence: <span id="conf-${ex.example_file}">--</span></div>
      <div class="decision ${decisionClass(false)}" id="decision-${ex.example_file}">--</div>
      <div class="badge" id="badge-${ex.example_file}">--</div>
    `;
    if (viewerGridEl) viewerGridEl.appendChild(card);

    try {
      const tensor = await loadAndPreprocess("examples/" + ex.example_file);
      const logits = await runInference(tensor);
      const probs = softmax(logits);
      const best = top3(probs)[0];
      const pred = CLASS_NAMES[best.i];
      const conf = best.p;
      const isAnswered = conf >= FROZEN_THRESHOLD;
      const match = pred === ex.class;

      if (isAnswered) answered++; else abstained++;
      if (match) matches++; else mismatches++;

      const predSpan = el("pred-" + ex.example_file);
      const confSpan = el("conf-" + ex.example_file);
      if (predSpan) predSpan.textContent = pred;
      if (confSpan) confSpan.textContent = formatPct(conf);
      const decEl = el("decision-" + ex.example_file);
      if (decEl) {
        decEl.textContent = decisionText(pred, conf, isAnswered);
        decEl.className = "decision " + decisionClass(isAnswered);
      }
      const badgeEl = card.querySelector("#badge-" + ex.example_file);
      if (badgeEl) {
        badgeEl.textContent = match ? "MATCH" : "MISMATCH";
        badgeEl.className = "badge " + (match ? "match" : "mismatch");
      }
      if (match) card.classList.add("match");
      else card.classList.add("mismatch");
      if (!isAnswered) card.classList.add("abstained");
    } catch (e) {
      console.error("Error on " + ex.example_file, e);
      const predSpan = card.querySelector(".pred-" + ex.example_file);
      if (predSpan) predSpan.textContent = "error";
      const decEl = card.querySelector("#decision-" + ex.example_file);
      if (decEl) decEl.textContent = "error";
      const badgeEl = card.querySelector("#badge-" + ex.example_file);
      if (badgeEl) badgeEl.textContent = "error";
    }

    if (viewerStatusEl) viewerStatusEl.textContent = "Loaded examples &hellip;";
  }

  if (viewerStatusEl) viewerStatusEl.textContent = "Loaded " + total + " examples.";
  if (summaryEl) {
    summaryEl.textContent =
      "Summary on these " + total + " examples only (not a test-set accuracy): " +
      matches + " match, " + mismatches + " mismatch, " +
      answered + " answered, " + abstained + " abstained (threshold " + FROZEN_THRESHOLD + ").";
  }
  let note = "";
  if (mismatches === 0) note += " No mismatches in this subset. ";
  if (abstained === 0) note += " No abstentions in this subset. ";
  if (viewerNoteEl) viewerNoteEl.textContent = note.trim();
}

// ----- single upload -----
function displayPrediction(logits, srcLabel) {
  const probs = softmax(logits);
  const top = top3(probs)[0];
  const best = top[0];

  const tc = el("topClass");
  const tfc = el("topConf");
  if (tc) tc.textContent = CLASS_NAMES[best.i];
  if (tfc) tfc.textContent = formatPct(best.p) + " confidence";
  const answered = best.p >= FROZEN_THRESHOLD;
  const sd = el("singleDecision");
  if (sd) {
    sd.textContent = answered
      ? "Answered &mdash; " + CLASS_NAMES[best.i]
      : "Abstained (route to review) &mdash; max " + formatPct(best.p) + " < " + FROZEN_THRESHOLD;
    sd.className = "decision-big " + (answered ? "answered-big" : "abstained-big");
  }
  const ri = el("resultImg");
  if (ri) { ri.src = el("previewImg").src; ri.alt = srcLabel; }

  const top3El = el("top3");
  if (top3El) {
    top3El.innerHTML = "";
    for (const t of top) {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = "<span>" + CLASS_NAMES[t.i] + "</span><span>" + formatPct(t.p) + "</span>";
      top3El.appendChild(row);
      const barWrap = document.createElement("div");
      barWrap.className = "prob-bar";
      const fill = document.createElement("div");
      fill.className = "prob-fill";
      fill.style.width = (t.p * 100) + "%";
      barWrap.appendChild(fill);
      top3El.appendChild(barWrap);
    }
  }

  const sr = el("singleResult");
  if (sr) sr.style.display = "block";
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
    const sr = el("singleResult");
    if (sr) sr.style.display = "none";
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
