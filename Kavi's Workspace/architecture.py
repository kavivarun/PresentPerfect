#!/usr/bin/env python3
"""
Script to load a YOLO .pt checkpoint and generate a Graphviz block-diagram of its DetectionModel
architecture **wrapped into multiple rows** so the image is neither excessively wide nor tall.

Features
--------
* Safe checkpoint loading (handles PyTorch ≥2.6 `weights_only` change)
* Automatic allow-listing of `DetectionModel` for unpickling
* Configurable number of layers per row (default = 5)
* SVG diagram written to OUTPUT_DIR

Prerequisites
-------------
```bash
# system renderer
sudo apt install graphviz        # or: brew install graphviz / choco install graphviz
# python wrapper
pip install graphviz torch ultralytics
```

Usage
-----
Just run the script — paths are hard-coded for your project layout, but you can tweak
`CHECKPOINT_PATH`, `OUTPUT_DIR`, or `LAYERS_PER_ROW` below.
"""

import os
import torch
from ultralytics.nn.tasks import DetectionModel
import torch.serialization
from graphviz import Digraph

# ───────────────────────────────────────── CONFIG ─────────────────────────────────────────
CHECKPOINT_PATH = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11\affectnet_yolov11.pt"
OUTPUT_DIR      = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11"
LAYERS_PER_ROW  = 5   # how many Sequential modules to show per row
# -----------------------------------------------------------------------------------------

# Allowlist DetectionModel for unpickling
torch.serialization.add_safe_globals([DetectionModel])

# ─────────────────────────────────── 1. Checkpoint Loader ─────────────────────────────────

def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """Load a YOLO `.pt` checkpoint, retrying with full unpickle if weights-only fails."""
    try:
        return torch.load(path, map_location=map_location)
    except RuntimeError as e:
        if "Weights only load failed" in str(e):
            print("[warn] weights-only load failed → retrying with weights_only=False …")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise

# ─────────────────────────────────── 2. Diagram Builder ───────────────────────────────────

def build_diagram(model: DetectionModel, layers_per_row: int = LAYERS_PER_ROW) -> Digraph:
    """Generate a Graphviz `Digraph` with the Sequential blocks wrapped across rows."""

    seq = getattr(model, "model", None)
    if seq is None:
        raise ValueError("DetectionModel has no .model Sequential attribute")

    dot = Digraph("YOLO_Arch", format="svg")
    dot.attr(rankdir="TB", fontsize="10", nodesep="0.4", ranksep="0.6", labelloc="t")

    modules = list(seq)
    rows = [modules[i:i + layers_per_row] for i in range(0, len(modules), layers_per_row)]

    # create one subgraph (same rank) per row so nodes line up horizontally
    for r, row in enumerate(rows):
        with dot.subgraph(name=f"cluster_row{r}") as sg:
            sg.attr(rank="same", style="invis")  # invisible border
            for c, module in enumerate(row):
                idx = r * layers_per_row + c
                params = sum(p.numel() for p in module.parameters())
                sg.node(f"n{idx}", f"{idx}: {module.__class__.__name__}\nParams: {params}")

    # sequential edges across all nodes
    for idx in range(1, len(modules)):
        dot.edge(f"n{idx-1}", f"n{idx}")

    return dot

# ─────────────────────────────────────── 3. Main ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"[info] Loading checkpoint → {CHECKPOINT_PATH}")
    try:
        ckpt = load_checkpoint(CHECKPOINT_PATH)
    except Exception as err:
        print(f"[error] Could not load checkpoint: {err}")
        raise SystemExit(1)

    model = ckpt.get("model")
    if model is None:
        print("[error] 'model' key missing in checkpoint!")
        raise SystemExit(1)

    # Build diagram
    diagram = build_diagram(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target = os.path.join(OUTPUT_DIR, "architecture_diagram")
    svg_path = diagram.render(filename=target)
    print(f"[done] Diagram saved ➜ {svg_path}")
