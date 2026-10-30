/**
 * EuroSAT Decision Demo — app.js
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

const statusEl = document.getElementById("status");
const loadStatusEl = document.getElementById("loadStatus");
const resultEl = document.getElementById("result");
const resultLoadingEl = document.getElementById("resultLoading");
const resizeNoticeEl = document.getElementById("resizeNotice");
const previewImg = document.getElementById("previewImg");
const resultImg = document.getElementById("resultImg");
const topClassEl = document.getElementById("topClass");
const topConfEl = document.getElementById("topConf");
const decisionEl = document.getElementById("decision");
const top3El = document.getElementById("top3");
const examplesEl = document.getElementById("examples");
const fileInput = document.getElementById("file");

let session = null;

// ----- model loading -----
async function loadModel() {
  try {
    loadStatusEl.textContent = "Loading ONNX runtime & model&hellip;";
    // onnxruntime-web is loaded from vendor/ort.wasm.min.js (script tag in index.html)
    session = await ort.InferenceSession.create(MODEL_PATH);
    loadStatusEl.textContent = "Model loaded. Ready.";
    loadStatusEl.className = "status ok";
    statusEl.textContent = "";
  } catch (e) {
    loadStatusEl.textContent = "Failed to load model.";
    loadStatusEl.className = "status err";
    console.error(e);
    statusEl.textContent = "Error: " + (e && e.message ? e.message : String(e));
    statusEl.className = "status err";
  }
}

loadModel();

// ----- helpers -----
function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (kind || "");
}

function showResult() {
  resultEl.style.display = "block";
  resultLoadingEl.style.display = "none";
}

function hideResult() {
  resultEl.style.display = "none";
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
// Load an image file (File or URL) into an Image, then draw to a 64x64 canvas,
// returning an Float32Array tensor [1,3,64,64] normalized with ImageNet mean/std.
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
        if (resized) {
          resizeNoticeEl.style.display = "block";
        } else {
          resizeNoticeEl.style.display = "none";
        }
        const imageData = ctx.getImageData(0, 0, 64, 64);
        const data = imageData.data; // RGBA uint8 clamped
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
            tensor[idx++] = (r - mean[0]) / std[0];   // channel 0
            tensor[idx++] = (g - mean[1]) / std[1];   // channel 1
            tensor[idx++] = (b - mean[2]) / std[2];   // channel 2
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
  const logits = outputs.logits.data; // Float32Array length 10
  return Array.from(logits);
}

// ----- UI update -----
function displayPrediction(logits, srcLabel) {
  const probs = softmax(logits);
  const top = top3(probs);
  const best = top[0];

  topClassEl.textContent = CLASS_NAMES[best.i];
  topConfEl.textContent = formatPct(best.p) + " confidence";
  const answered = best.p >= FROZEN_THRESHOLD;
  decisionEl.textContent = answered
    ? `Answered — ${CLASS_NAMES[best.i]}`
    : `Abstained (route to review) — max confidence ${formatPct(best.p)} < ${FROZEN_THRESHOLD}`;
  decisionEl.className = "decision " + (answered ? "answered" : "abstained");

  resultImg.src = previewImg.src;
  resultImg.alt = srcLabel;

  top3El.innerHTML = "";
  for (const t of top) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `<span>${CLASS_NAMES[t.i]}</span><span>${formatPct(t.p)}</span>`;
    top3El.appendChild(row);
    const barWrap = document.createElement("div");
    barWrap.className = "prob-bar";
    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.width = (t.p * 100) + "%";
    barWrap.appendChild(fill);
    top3El.appendChild(barWrap);
  }

  setStatus(
    `Prediction complete for ${srcLabel}. ${answered ? "Answered" : "Abstained"}.`,
    answered ? "ok" : ""
  );
  showResult();
}

// ----- sample selection -----
function populateExamples(manifest) {
  examplesEl.innerHTML = "";
  for (const ex of manifest.examples) {
    const thumb = document.createElement("img");
    thumb.className = "thumb";
    thumb.src = "examples/" + ex.example_file;
    thumb.alt = ex.class + " (idx " + ex.test_index + ")";
    thumb.title = `${ex.class} — test index ${ex.test_index} — source: ${ex.source_file}`;
    thumb.onclick = () => {
      document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("selected"));
      thumb.classList.add("selected");
      previewImg.src = thumb.src;
      previewImg.alt = thumb.alt;
      resizeNoticeEl.style.display = "none";
      hideResult();
      setStatus(`Selected ${ex.class} (test index ${ex.test_index}). Click Predict if needed.`);
      autoPredictFromSrc(thumb.src, thumb.title);
    };
    examplesEl.appendChild(thumb);
  }
}

// Auto-predict when a sample is clicked
async function autoPredictFromSrc(src, title) {
  if (!session) return;
  try {
    resultLoadingEl.style.display = "flex";
    resultEl.style.display = "none";
    const tensor = await loadAndPreprocess(src);
    const logits = await runInference(tensor);
    displayPrediction(logits, title);
  } catch (e) {
    console.error(e);
    setStatus("Error running inference: " + (e.message || String(e)), "err");
    resultLoadingEl.style.display = "none";
  }
}

// ----- file upload -----
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  previewImg.src = URL.createObjectURL(file);
  previewImg.alt = file.name;
  resizeNoticeEl.style.display = "none";
  hideResult();
  setStatus(`Uploaded ${file.name}. Running inference&hellip;`);
  resultLoadingEl.style.display = "flex";
  try {
    const tensor = await loadAndPreprocess(file);
    const logits = await runInference(tensor);
    displayPrediction(logits, file.name);
  } catch (err) {
    console.error(err);
    setStatus("Error: " + (err.message || String(err)), "err");
    resultLoadingEl.style.display = "none";
  }
});

// Load examples manifest and pre-select the first one
fetch("examples_manifest.json")
  .then((r) => r.json())
  .then((manifest) => {
    populateExamples(manifest);
    if (manifest.examples.length > 0) {
      const first = manifest.examples[0];
      const firstThumb = examplesEl.children[0];
      if (firstThumb) {
        firstThumb.classList.add("selected");
        previewImg.src = firstThumb.src;
        previewImg.alt = firstThumb.alt;
        setStatus(`Loaded example: ${first.class} (test index ${first.test_index}).`);
      }
    }
  })
  .catch((e) => {
    console.warn("Could not load examples manifest:", e);
  });
