"""
Step 1 — Export the committed ResNet-18 model to ONNX.

- Loads ResNet-18 (ImageNet-pretrained architecture), fc replaced for 10 classes.
- Loads state dict from source models/best.pt (CPU).
- Exports to ONNX opset 17, input [1,3,64,64], output logits.
- Verifies against PyTorch on 20 random test images from splits/test.csv.
- Normalizes inputs with ImageNet mean/std (same as training val/test transform).
"""

import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision import transforms, models
import onnx
import onnxruntime as ort

# --- Paths ---
SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eurosat-decision"))
TOOL = os.path.abspath(os.path.dirname(__file__))
SRC_MODEL = os.path.join(SOURCE, "models", "best.pt")
SRC_SPLITS = os.path.join(SOURCE, "splits")
SRC_DATA = os.path.join(SOURCE, "data")
OUT_MODEL = os.path.join(TOOL, "model.onnx")

assert os.path.exists(SRC_MODEL), f"Missing source model: {SRC_MODEL}"
assert os.path.exists(SRC_SPLITS), f"Missing source splits: {SRC_SPLITS}"

# --- Seeds ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print("=" * 60)
print("Step 1 — Export model to ONNX")
print("=" * 60)
print(f"Source model: {SRC_MODEL}")
print(f"Output model: {OUT_MODEL}")
print(f"Classes ({NUM_CLASSES}): {CLASSES}")
print(f"Input normalization: mean={IMAGENET_MEAN}, std={IMAGENET_STD} (ImageNet)")
print(f"Opset: 17 | Input shape: [1, 3, 64, 64] | Output: logits")

# --- Build model + load state dict ---
def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

model = build_model()
state = torch.load(SRC_MODEL, map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"Loaded state dict from {SRC_MODEL}")

# --- ONNX export ---
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model,
    dummy,
    OUT_MODEL,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes=None,
    dynamo=False,
)
print(f"Exported ONNX model to {OUT_MODEL} (opset 17)")

# Sanity-check the ONNX file loads
onnx_model = onnx.load(OUT_MODEL)
onnx.checker.check_model(onnx_model)
print("ONNX model checked OK")

# --- Verification against PyTorch on 20 random test images ---
def load_split_indices(name):
    path = os.path.join(SRC_SPLITS, f"{name}.csv")
    idxs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idxs.append(int(row["index"]))
    return sorted(idxs)

test_idx = load_split_indices("test")
print(f"Test split size: {len(test_idx)}")

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

full_test = EuroSAT(root=SRC_DATA, download=False, transform=val_tf)
test_subset = Subset(full_test, test_idx)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0)

# Pick 20 random test indices
sample_indices = random.sample(range(len(test_subset)), 20)
print(f"Verifying on 20 random test images: {sample_indices}")

session = ort.InferenceSession(OUT_MODEL, providers=["CPUExecutionProvider"])

mismatches = 0
checked = 0
for i, (x, y) in enumerate(test_loader):
    if i not in sample_indices:
        continue
    x_np = x.numpy().astype(np.float32)
    # PyTorch
    with torch.no_grad():
        pt_logits = model(x)
        pt_pred = int(pt_logits.argmax(1).item())
    # ONNX
    ort_out = session.run(["logits"], {"input": x_np})[0]
    ort_pred = int(np.argmax(ort_out, axis=1).item())
    match = pt_pred == ort_pred
    checked += 1
    if not match:
        mismatches += 1
    print(f"  img #{i}: true={CLASSES[y.item()]} pt={CLASSES[pt_pred]} ort={CLASSES[ort_pred]} match={match}")

print(f"Verified {checked} images, mismatches={mismatches}")
assert checked == 20, f"Expected 20 checks, got {checked}"
assert mismatches == 0, f"Found {mismatches} mismatches"
print("ONNX vs PyTorch: ALL 20 MATCH")

# --- Record verification record ---
record_path = os.path.join(TOOL, "onnx_verification.json")
record = {
    "source_model": SRC_MODEL,
    "source_model_sha256": "C6C5EED4BE400498D8C43941E1115DA72DCDE94A37D73B358E6182923201716C",
    "output_model": OUT_MODEL,
    "opset": 17,
    "input_shape": [1, 3, 64, 64],
    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "classes": CLASSES,
    "num_checked": checked,
    "mismatches": mismatches,
    "all_match": mismatches == 0,
    "test_indices_checked": sample_indices,
    "source_note": "Model weights and threshold are NOT modified; exported from committed eurosat-decision models/best.pt",
}
import json
with open(record_path, "w") as f:
    json.dump(record, f, indent=2)
print(f"Wrote verification record to {record_path}")
print("Step 1 complete — model exported and verified.")
