"""
Extract 12 example patches (2 per class, across 6 classes) from the EuroSAT test split
into tool/examples/, recording each source file.
"""

import csv
import os
import shutil

SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eurosat-decision"))
TOOL = os.path.abspath(os.path.dirname(__file__))
SRC_DATA = os.path.join(SOURCE, "data")
SRC_SPLITS = os.path.join(SOURCE, "splits")
EXAMPLES_DIR = os.path.join(TOOL, "examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

def load_split_indices(name):
    path = os.path.join(SRC_SPLITS, f"{name}.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

test_rows = load_split_indices("test")
print(f"Test split rows: {len(test_rows)}")

# Group test rows by class
by_class = {c: [] for c in CLASSES}
for row in test_rows:
    by_class[row["class"]].append(row)

# Pick 2 per class from first 6 classes = 12 total
picks = []
for c in CLASSES[:6]:
    chosen = by_class[c][:2]
    for row in chosen:
        picks.append((c, row))

print(f"Picked {len(picks)} examples:")
manifest = []
for c, row in picks:
    rel = row["filepath"]
    src_path = os.path.join(SRC_DATA, rel)
    assert os.path.exists(src_path), f"Missing source: {src_path}"
    # Destination name: class_index_sourcefilename
    src_name = os.path.basename(src_path)
    dst_name = f"{c}_{row['index']}_{src_name}"
    dst_path = os.path.join(EXAMPLES_DIR, dst_name)
    shutil.copy2(src_path, dst_path)
    manifest.append({
        "class": c,
        "test_index": int(row["index"]),
        "source_file": rel,
        "example_file": dst_name,
    })
    print(f"  {c} idx={row['index']}: {rel} -> {dst_name}")

# Write manifest
import json
manifest_path = os.path.join(TOOL, "examples_manifest.json")
with open(manifest_path, "w") as f:
    json.dump({
        "total": len(manifest),
        "classes_covered": [m["class"] for m in manifest],
        "examples": manifest,
    }, f, indent=2)
print(f"\nWrote manifest to {manifest_path} ({len(manifest)} examples)")
