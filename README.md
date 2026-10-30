# EuroSAT Decision Tool — Browser Demo

A small, free, browser-based demo of the committed EuroSAT ResNet-18 decision model.
Pick a 64×64 image patch (from 12 included test-split examples, two per class, or upload your own)
and see the predicted class, the softmax-confidence abstention decision at the frozen threshold,
and the confidence. Everything runs locally — no network calls at runtime.

## What it does

- Loads the committed ResNet-18 model (exported to ONNX, opset 17) via onnxruntime-web from
  the local `vendor/` folder.
- Preprocesses a 64×64 RGB patch with the ImageNet mean/std used in training.
- Shows the top-3 class probabilities, the top class, and the abstention decision:
  - **answered** if max probability ≥ 0.50 (frozen threshold)
  - **abstained (route to review)** if max probability < 0.50
- Displays the committed test-set figures with their labels and denominators.

## How to run locally

From the `eurosat-decision-tool` folder:

```bash
python -m http.server 8765
```

Then open `http://localhost:8765/` in a browser. No build step, no framework, no CDN,
no external network calls at runtime.

## Files

| File | Purpose |
|------|---------|
| `index.html` | UI + inline CSS |
| `app.js` | Inference logic (onnxruntime-web) |
| `model.onnx` | Exported ResNet-18 model (ONNX opset 17) |
| `examples/` | 12 example patches (two per class) from the test split |
| `examples_manifest.json` | Source-file record for each example |
| `vendor/` | Local copy of onnxruntime-web dist (no CDN) |
| `onnx_verification.json` | ONNX vs PyTorch verification record |
| `README.md` | This file |

## Model & threshold

- Model: ResNet-18 (ImageNet-pretrained), final `fc` replaced for the 10 EuroSAT classes.
- Weights: exported from `eurosat-decision/models/best.pt` (sha256
  `C6C5EED4BE400498D8C43941E1115DA72DCDE94A37D73B358E6182923201716c`).
  The weights are NOT modified by this tool.
- Input size: 64×64 (native EuroSAT patch); no resize to 224.
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`.
- Frozen abstention threshold: **0.50**, chosen on validation only
  (see `eurosat-decision/reports/abstention_threshold.json`). Not retuned here.

## Test-set figures (from committed reports/TEST_RESULTS.md)

| Figure | Value | Label |
|--------|-------|-------|
| Overall test accuracy | 0.9627 (3,899 / 4,050) | Only unbiased figure |
| Validation accuracy | 0.9679 | Validation-selected, not unbiased |

The 0.9679 figure is validation-selected (LR=0.0001, epoch 4); the 0.9627 test figure is the
only unbiased result. The tool does not evaluate the model on any new data and reports no new
accuracy.

## Scope and limitations

- Benchmark on **EuroSAT only**; no accuracy is claimed on any other imagery.
- 64×64 Sentinel-2 patches at 10 m resolution — **not field imagery**.
- Single run with one seed (42); **no confidence interval computed**.
- Split is the project's own fixed stratified 70/15/15 split, not a published benchmark split.
- This demo is not claimed to be accurate on imagery outside EuroSAT.

## Source

Companion repository: `C:\Users\91974\OneDrive\Desktop\Gemini Antigravity\eurosat-decision`
(code, weights, splits, reports). This tool reads only from that repo; it does not modify it.
