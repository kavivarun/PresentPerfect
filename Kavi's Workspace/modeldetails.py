#!/usr/bin/env python3
"""
Script to load a YOLO .pt checkpoint and print all stored details:
  - Top-level keys
  - Model architecture
  - YAML configuration
  - Training hyper-parameters
  - Optimizer state (first param group)
  - Validation metrics history
  - Exports hyper-parameters to CSV and JSON

Usage:
  Just run the script and follow prompts.
"""

import torch
import pprint
import pandas as pd
import json
# Allowlist DetectionModel so torch can unpickle it
from ultralytics.nn.tasks import DetectionModel
import torch.serialization
# Persistently add safe global for DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

def load_checkpoint(path):
    """
    Attempt to load a checkpoint. If a weights-only load fails,
    retry with weights_only=False (requires trusted source).
    """
    try:
        return torch.load(path, map_location="cpu")
    except RuntimeError as e:
        msg = str(e)
        if "Weights only load failed" in msg:
            print("Warning: weights-only load failed. Retrying with weights_only=False...")
            # Unsafe load: executes arbitrary code, trust your source
            return torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise

def main():
    # Prompt user for checkpoint path
    checkpoint_path = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11.pt"
    if not checkpoint_path:
        print("No checkpoint path provided. Exiting.")
        return

    # Default export filenames
    csv_file = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11\besthyperparametersbase.csv"
    json_file = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11\hyperparametersbase.json"

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    pp = pprint.PrettyPrinter(indent=2)

    # Top-level keys
    print("\n== Top-level keys ==")
    pp.pprint(list(ckpt.keys()))

    # Model architecture
    if "model" in ckpt:
        print("\n== Model architecture ==")
        print(ckpt["model"])
    else:
        print("\nNo 'model' key found in checkpoint.")

    # YAML configuration (if present)
    yaml_cfg = getattr(ckpt.get("model", {}), "yaml", None)
    if yaml_cfg:
        print("\n== Model YAML config ==")
        print(yaml_cfg)

    # Training hyper-parameters
    hyp = ckpt.get("train_args", ckpt.get("hyp", {}))
    if hyp:
        print("\n== Training hyper-parameters ==")
        pp.pprint(hyp)
    else:
        print("\nNo training hyper-parameters found.")

    # Optimizer state
    opt = ckpt.get("optimizer", {})
    if opt and "param_groups" in opt and opt["param_groups"]:
        print("\n== Optimizer state (first param group) ==")
        pp.pprint(opt["param_groups"][0])
    else:
        print("\nNo optimizer state found or 'param_groups' missing.")

    # Validation metrics
    metrics = ckpt.get("metrics", None)
    if metrics:
        print("\n== Validation metrics history ==")
        pp.pprint(metrics)
    else:
        print("\nNo validation metrics found.")

    # Export hyper-parameters to CSV
    if hyp:
        try:
            df = pd.DataFrame([hyp]).T
            df.to_csv(csv_file)
            print(f"\nSaved hyper-parameters CSV to {csv_file}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    # Export hyper-parameters to JSON
    if hyp:
        try:
            with open(json_file, "w") as f:
                json.dump(hyp, f, indent=2)
            print(f"Saved hyper-parameters JSON to {json_file}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")


if __name__ == "__main__":
    main()