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
    data["path"] = yaml_dir
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"✅ YAML patched. New root path: {data['path']}")
    return yaml_path

def main():
    print("--- 🧪 MICROPLASTICS: YOLOv11-L INSTANCE SEG (HEAVY AUG, SMALL DATASET) ---")

    data_config_path = find_and_fix_config("data_root")

    project_name = "yolo_results"
    run_name = "microplastics_yolo11l_seg_v3_heavyaug"
    checkpoint_path = os.path.join(project_name, run_name, "weights", "last.pt")

    if os.path.exists(checkpoint_path):
        print(f"🔄 RESUMING from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        resume_flag = True
        model_init = checkpoint_path
    else:
        print("🚀 STARTING FRESH from pretrained YOLOv11-L seg")
        model = YOLO("yolo11l-seg.pt")
        resume_flag = False
        model_init = "yolo11l-seg.pt"

    model.train(
        data=data_config_path,
        project=project_name,
        name=run_name,
        resume=resume_flag,

        # RESOURCES
        epochs=600,          # more epochs => more unique augmented views
        patience=120,
        imgsz=1280,
        batch=2,
        device=0,
        workers=8,

        # OPTIMIZATION / REGULARIZATION (critical for L/X on tiny datasets)
        optimizer="AdamW",
        lr0=0.0015,           # stable for finetuning pretrained seg
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.015,  # stronger anti-memorization
        warmup_epochs=3.0,
        label_smoothing=0.05,

        # MULTI-SCALE increases robustness to zoom
        multi_scale=True,

        # AUGMENTATION: aggressive but microscopy-realistic
        mosaic=1.0,
        close_mosaic=75,

        # For segmentation: keep mixup off (blurs boundaries)
        mixup=0.0,

        # Copy-paste adds crowding but can cause fragmentation if too high
        copy_paste=0.10,

        # HSV moderate (you requested)
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,

        # Geometry: very strong rotations + flips OK in microscopy
        degrees=360.0,
        translate=0.12,
        scale=0.6,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,

        # LOGGING
        val=True,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("\n--- RUN DOCUMENTATION (paste into README) ---")
    print(f"run_name={run_name}")
    print(f"model_init={model_init}")
    print(f"data={data_config_path}")
    print("epochs=600 patience=120 imgsz=1280 batch=6 optimizer=AdamW lr0=0.002 lrf=0.01 cos_lr=True")
    print("weight_decay=0.015 warmup_epochs=3.0 label_smoothing=0.05 multi_scale=True")
    print("mosaic=1.0 close_mosaic=75 mixup=0.0 copy_paste=0.10")
    print("hsv_h=0.01 hsv_s=0.35 hsv_v=0.25")
    print("degrees=360 translate=0.12 scale=0.6 shear=0 perspective=0 flipud=0.5 fliplr=0.5")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    main()
