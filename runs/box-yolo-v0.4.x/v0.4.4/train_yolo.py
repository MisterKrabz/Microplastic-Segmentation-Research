import glob
import os
import sys
import yaml


def log_epoch_stats(trainer):
    """Callback to log metrics to a custom text file and print at the end of each epoch."""
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    # Extract bounding box metrics (since we are back to detection)
    metrics = trainer.metrics or {}
    p = metrics.get('metrics/precision(B)', 0.0)
    r = metrics.get('metrics/recall(B)', 0.0)
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    # Safely extract training loss if available
    try:
        train_loss = getattr(trainer, 'loss', 0.0)
        if hasattr(train_loss, 'item'):
            train_loss = train_loss.item()
        elif isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.sum().item()
    except Exception:
        train_loss = 0.0
    
    log_string = (
        f"Epoch {epoch:03d}/{total_epochs:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Precision: {p:.4f} | Recall: {r:.4f} | mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}\n"
    )
    
    # Print to standard output and flush immediately for real-time HTCondor viewing
    print(f"[STATUS UPDATE] {log_string.strip()}")
    sys.stdout.flush()
    
    # Append to a custom log file (transferred back via HTCondor)
    with open("epoch_training_progress.log", "a") as f:
        f.write(log_string)


def rewrite_yaml_for_chtc(root_dir="data_root"):
    root_dir = os.path.abspath(root_dir)
    
    # Find all yamls, use the first one as the base for class names
    yamls = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)
    if not yamls:
        yamls = glob.glob(os.path.join(root_dir, "**", "dataset.yaml"), recursive=True)
    if not yamls:
        raise FileNotFoundError("Could not find data.yaml or dataset.yaml under data_root/")
        
    base_yaml = yamls[0]
    with open(base_yaml, "r") as f:
        data = yaml.safe_load(f) or {}

    # Find ALL train/val/test image directories across all datasets
    train_images = [os.path.abspath(p) for p in glob.glob(os.path.join(root_dir, "**", "train", "images"), recursive=True)]
    val_images = [os.path.abspath(p) for p in glob.glob(os.path.join(root_dir, "**", "valid", "images"), recursive=True)]
    val_images += [os.path.abspath(p) for p in glob.glob(os.path.join(root_dir, "**", "val", "images"), recursive=True)]
    test_images = [os.path.abspath(p) for p in glob.glob(os.path.join(root_dir, "**", "test", "images"), recursive=True)]

    # Deduplicate lists
    train_images = list(dict.fromkeys(train_images))
    val_images = list(dict.fromkeys(val_images))
    test_images = list(dict.fromkeys(test_images))

    if not train_images:
        raise FileNotFoundError("Could not find train/images anywhere under data_root/")

    if not val_images:
        print("⚠️ No valid/ or val/ split found. Falling back to val=train.")
        val_images = train_images

    # YOLO allows lists of directories for training on multiple datasets
    data["path"] = root_dir
    data["train"] = train_images
    data["val"] = val_images
    
    if test_images:
        data["test"] = test_images
    elif "test" in data:
        del data["test"]

    # Save to a new master yaml
    master_yaml_path = os.path.join(root_dir, "master_data.yaml")
    with open(master_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return master_yaml_path


def main():
    print("--- 📦 YOLO BBOX TRAINING ON CHTC ---")

    wd = os.getcwd()
    os.environ["USER"] = "condor"
    os.environ["LOGNAME"] = "condor"
    os.environ["HOME"] = wd
    os.environ["XDG_CACHE_HOME"] = os.path.join(wd, ".cache")
    os.environ["TORCH_HOME"] = os.path.join(wd, ".cache", "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(wd, ".cache", "torchinductor")
    os.environ["MPLCONFIGDIR"] = os.path.join(wd, ".cache", "matplotlib")
    os.environ["ULTRALYTICS_SETTINGS_DIR"] = os.path.join(wd, ".ultralytics")
    os.environ["YOLO_CONFIG_DIR"] = os.path.join(wd, ".ultralytics")

    for key in ["XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "MPLCONFIGDIR", "ULTRALYTICS_SETTINGS_DIR"]:
        os.makedirs(os.environ[key], exist_ok=True)

    global torch
    import torch
    from ultralytics import YOLO

    data_config_path = rewrite_yaml_for_chtc("data_root")
    project_dir = os.path.join(wd, "yolo_results")
    run_name = "lake_mendota_yolo11l_bbox_combined"
    os.makedirs(project_dir, exist_ok=True)

    # Pure Object Detection model for your SAM2 pipeline
    model = YOLO("yolo11l.pt")
    model.add_callback("on_fit_epoch_end", log_epoch_stats)

    workers = min(4, os.cpu_count() or 1)

    model.train(
        data=data_config_path,
        project=project_dir,
        name=run_name,
        resume=False,

        epochs=2000,
        patience=200,
        imgsz=1280,
        batch=8,          # Try changing this to 8 later if your GPU memory allows!
        device=0,
        workers=workers,

        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.01,
        warmup_epochs=5.0,

        # Microscopy-Safe Augmentations for Blob Detection
        mosaic=0.0,       # OFF: Preserves slide context and relative sizes
        mixup=0.05,
        copy_paste=0.0,   # OFF: Irrelevant for bbox models

        degrees=360.0,    # Safe: Microscopy samples have no up/down
        scale=0.25,        # Slight variance helps it recognize large blobs
        translate=0.1,    # Shifts image so blobs aren't always centered
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
	overlap_mask=True,

        hsv_h=0.0,        # OFF: Color sensitive data
        hsv_s=0.0,        # OFF: Color sensitive data
        hsv_v=0.0,        # OFF: Color sensitive data

        amp=True,
        val=True,
        save=True,
        plots=True,
        exist_ok=True,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ train_yolo.py failed: {e}", file=sys.stderr)
        raise