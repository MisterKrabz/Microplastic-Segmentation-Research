import os
import sys
import glob
import yaml
from ultralytics import YOLO

def find_and_fix_config(root_dir="."):
    print(f"🔎 Searching for data config in {os.path.abspath(root_dir)}...")
    matches = glob.glob(os.path.join(root_dir, "**", "dataset.yaml"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)
    if not matches:
        print("❌ Could not find 'dataset.yaml' or 'data.yaml'!")
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
    print("--- 🧼 MICROPLASTICS: YOLOv11-L INSTANCE SEG (CLEAN BASELINE / OVERFIT SANITY CHECK) ---")

    # --------------------------------------------
    # IMPORTANT FOR CHTC/CONTAINERS:
    # Redirect Ultralytics settings/cache/runs to a writable directory
    # (current working directory is writable inside the job sandbox)
    # --------------------------------------------
    wd = os.getcwd()
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", os.path.join(wd, ".ultralytics"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(wd, ".ultralytics"))
    os.environ.setdefault("YOLO_RUNS_DIR", os.path.join(wd, "runs"))

    os.makedirs(os.environ["ULTRALYTICS_SETTINGS_DIR"], exist_ok=True)
    os.makedirs(os.environ["YOLO_RUNS_DIR"], exist_ok=True)

    # Keep the same dataset root and config behavior
    data_config_path = find_and_fix_config("data_root")

    # Keep same project/name pattern as your old script
    project_name = "yolo_results"
    run_name = "microplastics_yolo11l_seg_CLEAN_BASELINE"
    checkpoint_path = os.path.join(project_name, run_name, "weights", "last.pt")

    # Baseline = start fresh (do NOT resume). But keep your structure + prints.
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        print(f"ℹ️ Found existing checkpoint at: {checkpoint_path}")
        print("🧼 Clean baseline should start fresh. Delete the run folder if you want a totally clean rerun.")
    print("🚀 STARTING FRESH from pretrained YOLOv11-L seg")
    model = YOLO("yolo11l-seg.pt")
    resume_flag = False
    model_init = "yolo11l-seg.pt"

    # -------------------------
    # CLEAN BASELINE TRAINING
    # -------------------------
    model.train(
        data=data_config_path,
        project=project_name,   # ensures outputs go to ./yolo_results (writable)
        name=run_name,
        resume=resume_flag,

        # -------------------------
        # RESOURCES / TRAIN LENGTH
        # -------------------------
        # With 38 images, you WANT lots of epochs to overfit.
        epochs=600,
        patience=200,
        imgsz=1280,
        batch=2,
        device=0,
        workers=8,

        # -------------------------
        # OPTIMIZATION / STABILITY
        # -------------------------
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.01,
        warmup_epochs=5.0,
        label_smoothing=0.0,
        multi_scale=False,

        # -------------------------
        # AUGMENTATION (CLEAN / NON-DESTRUCTIVE)
        # -------------------------
        # Disable scene-mixing and anything likely to create hallucinations
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,

        # Allow rotation (microscopy OK), but avoid translate/scale cropping risk
        degrees=360.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,

        # Mild photometric only (texture-safe)
        hsv_h=0.005,
        hsv_s=0.15,
        hsv_v=0.10,

        # -------------------------
        # CHTC SAFETY: disable AMP to avoid amp-check predict writing to /ultralytics/runs
        # -------------------------
        amp=False,

        # -------------------------
        # LOGGING
        # -------------------------
        val=True,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("\n--- RUN DOCUMENTATION (paste into README) ---")
    print(f"run_name={run_name}")
    print(f"model_init={model_init}")
    print(f"data={data_config_path}")
    print("epochs=600 patience=200 imgsz=1280 batch=2 optimizer=AdamW lr0=0.001 lrf=0.01 cos_lr=True")
    print("weight_decay=0.01 warmup_epochs=5.0 label_smoothing=0.0 multi_scale=False amp=False")
    print("mosaic=0.0 close_mosaic=0 mixup=0.0 copy_paste=0.0")
    print("degrees=360 translate=0.0 scale=0.0 shear=0 perspective=0 flipud=0.5 fliplr=0.5")
    print("hsv_h=0.005 hsv_s=0.15 hsv_v=0.10")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
