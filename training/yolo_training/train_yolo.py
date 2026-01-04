import os
import sys
import glob
import yaml 
from ultralytics import YOLO

def find_and_fix_config(root_dir="."):
    """
    Finds and fixes the dataset.yaml to use absolute paths.
    Essential for CHTC execution.
    """
    print(f"🔎 Searching for data config in {os.path.abspath(root_dir)}...")
    matches = glob.glob(os.path.join(root_dir, "**", "dataset.yaml"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)
    
    if not matches:
        print("CRITICAL ERROR: Could not find 'dataset.yaml' or 'data.yaml'!")
        sys.exit(1)

    yaml_path = os.path.abspath(matches[0])
    yaml_dir = os.path.dirname(yaml_path)
    
    # Read the current YAML
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Force the 'path' to be the directory containing the yaml
    data['path'] = yaml_dir 
    
    # Write it back
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    print(f"YAML patched. New root path: {data['path']}")
    return yaml_path

def main():
    print("--- 🔍 PYTHON MICROPLASTIC TRAINING (PRECISION TUNING V2) ---")

    # Data source
    data_config_path = find_and_fix_config("data_root")

    # AUTO-RESUME LOGIC 
    project_name = "yolo_results"
    run_name = "microplastic_v2_precision"
    checkpoint_path = os.path.join(project_name, run_name, "weights", "last.pt")

    resume_flag = False

    if os.path.exists(checkpoint_path):
        print(f"🔄 EVICTION RECOVERY: Found checkpoint at {checkpoint_path}")
        print("⚡ Resuming training from where it left off...")
        model = YOLO(checkpoint_path)
        resume_flag = True
    else:
        print("NO CHECKPOINT FOUND: Starting fresh training...")
        print("Loading YOLOv11-Extra-Large (x)...")
        model = YOLO('yolo11x.pt') 
        resume_flag = False

    print(f"Starting Training...")
    
    results = model.train(
        data=data_config_path,
        project=project_name,
        name=run_name,
        resume=resume_flag,
        
        # --- RESOURCES ---
        epochs=300,
        patience=50,
        imgsz=1280,
        batch=8,
        device=0,
        workers=8,
        
        # --- BOUNDARY & SEPARATION LOGIC ---
        box=12.0,
        dfl=3.0,

        # --- HALLUCINATION & OVERLAP LOGIC ---
        iou=0.5,
        mixup=0.0,
        copy_paste=0.15,

        # --- SENSITIVITY TUNING ---
        hsv_h=0.015, 
        hsv_s=0.7,   
        hsv_v=0.4,   

        # --- FALSE POSITIVE REDUCTION ---
        close_mosaic=50,

        # --- GEOMETRY ---
        degrees=180.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        
        # --- CONTEXT ---
        mosaic=1.0, 
        
        exist_ok=True,
        save=True,
        val=True,
        plots=True
    )

if __name__ == '__main__':
    main()