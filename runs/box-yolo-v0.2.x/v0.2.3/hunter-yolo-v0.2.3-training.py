"""
===========================================================
MODEL: Microplastics YOLO V4 — Bounding Box Detector
PURPOSE: High-fidelity detection in dense, overlapping scenes
STATUS: ARCHIVED 
===========================================================

Key characteristics:
- Bounding box detection only
- Heavy augmentation to simulate dense overlaps
- Tuned to reduce false negatives at cost of duplicate boxes within boxes 
- Ultimately limited by box-only supervision
- Even at the perfect confidence and IOU, the model draws duplicate boxes while also misses some microplastics entirely 
"""

import os
import sys
import glob
import yaml
from ultralytics import YOLO


def find_and_fix_config(root_dir="."):
    """
    Finds dataset.yaml or data.yaml and rewrites the `path`
    field to an absolute path. Required for CHTC execution.
    """
    print(f"🔎 Searching for data config in {os.path.abspath(root_dir)}...")

    matches = glob.glob(os.path.join(root_dir, "**", "dataset.yaml"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)

    if not matches:
        print("❌ CRITICAL ERROR: Could not find 'dataset.yaml' or 'data.yaml'!")
        sys.exit(1)

    yaml_path = os.path.abspath(matches[0])
    yaml_dir = os.path.dirname(yaml_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Force absolute root path
    data["path"] = yaml_dir

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"✅ YAML patched. New root path: {data['path']}")
    return yaml_path


def main():
    print("--- 🔍 PYTHON MICROPLASTIC TRAINING (V4 HIGH-FIDELITY BBOX) ---")

    # Dataset (unzipped into data_root/)
    data_config_path = find_and_fix_config("data_root")

    # ------------------------------------------------------------
    # RUN IDENTITY
    # ------------------------------------------------------------
    project_name = "yolo_results"
    run_name = "microplastic_v4_high_fidelity"
    checkpoint_path = os.path.join(
        project_name, run_name, "weights", "last.pt"
    )

    # ------------------------------------------------------------
    # RESUME LOGIC
    # ------------------------------------------------------------
    if os.path.exists(checkpoint_path):
        print(f"🔄 RESUMING from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        resume_flag = True
    else:
        print("🚀 STARTING FRESH TRAINING RUN")
        model = YOLO("yolo11x.pt")  # YOLO11 extra-large (DETECTION)
        resume_flag = False

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------
    print("🔥 Starting YOLO V4 training...")

    model.train(
        data=data_config_path,
        project=project_name,
        name=run_name,
        resume=resume_flag,

        # ---------------- RESOURCES ----------------
        epochs=300,
        patience=50,
        imgsz=1280,     # High resolution for dense separation
        batch=6,        # Lower batch to allow heavy augmentations
        device=0,
        workers=8,

        # ---------------- OVERLAP / CROWD LOGIC ----------------
        # Mixup simulates transparency and partial overlap
        mixup=0.1,

        # Copy-paste aggressively simulates dense crowds
        copy_paste=0.3,

        # ---------------- SENSITIVITY ----------------
        # Penalize missed detections heavily
        cls=2.0,

        # ---------------- BOX STABILITY ----------------
        # Reduced from earlier extreme values to avoid jitter
        box=7.5,
        dfl=1.5,

        # Training-side NMS IoU (validation)
        # High so the model does NOT learn to suppress overlaps
        iou=0.7,

        # ---------------- COLOR / TEXTURE ----------------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ---------------- REFINEMENT PHASE ----------------
        # Turn off heavy mosaic near the end
        close_mosaic=50,

        # ---------------- GEOMETRY ----------------
        degrees=90.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,              # Disabled (shear distorts circular plastics)
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,

        # ---------------- CONTEXT ----------------
        mosaic=1.0,

        # ---------------- LOGGING ----------------
        save=True,
        val=True,
        plots=True,
        exist_ok=True
    )

    print("✅ Training complete for YOLO V4 (Bounding Boxes).")


if __name__ == "__main__":
    main()
