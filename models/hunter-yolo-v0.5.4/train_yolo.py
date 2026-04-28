import glob
import os
import sys
import yaml
import gc
import torch
from ultralytics import YOLO

def log_epoch_stats(trainer):
    """Safely log metrics using Ultralytics internal dicts."""
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    losses = trainer.label_loss_items(trainer.tloss, prefix="train") if hasattr(trainer, 'tloss') else {}
    train_loss = sum(losses.values()) if losses else 0.0
    
    metrics = trainer.metrics or {}
    p = metrics.get('metrics/precision(B)', 0.0)
    r = metrics.get('metrics/recall(B)', 0.0)
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    save_dir_obj = getattr(trainer, 'save_dir', 'unknown_run')
    run_name = os.path.basename(str(save_dir_obj))
    
    log_string = (
        f"[{run_name}] Epoch {epoch:03d}/{total_epochs:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Precision: {p:.4f} | Recall: {r:.4f} | mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}\n"
    )
    
    print(f"[STATUS UPDATE] {log_string.strip()}")
    sys.stdout.flush()
    
    with open("epoch_training_progress.log", "a") as f:
        f.write(log_string)

def rewrite_yaml_for_chtc(root_dir="data_root"):
    root_dir = os.path.abspath(root_dir)
    
    yamls = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)
    if not yamls:
        yamls = glob.glob(os.path.join(root_dir, "**", "dataset.yaml"), recursive=True)
    if not yamls:
        raise FileNotFoundError("Could not find data.yaml or dataset.yaml under data_root/")
        
    base_yaml = yamls[0]
    with open(base_yaml, "r") as f:
        data = yaml.safe_load(f) or {}

    train_dirs = glob.glob(os.path.join(root_dir, "**", "train", "images"), recursive=True)
    val_dirs = glob.glob(os.path.join(root_dir, "**", "valid", "images"), recursive=True)
    if not val_dirs:
        val_dirs = glob.glob(os.path.join(root_dir, "**", "val", "images"), recursive=True)

    data["path"] = root_dir
    data["train"] = os.path.relpath(train_dirs[0], root_dir) if train_dirs else "."
    data["val"] = os.path.relpath(val_dirs[0], root_dir) if val_dirs else data["train"]
    
    if "test" in data:
        del data["test"]

    master_yaml_path = os.path.join(root_dir, "master_data.yaml")
    with open(master_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return master_yaml_path

def verify_and_patch_checkpoint(filepath, current_data_path, current_project_dir):
    if not os.path.exists(filepath): return None, False
    if os.path.getsize(filepath) < 10_000_000: return None, False 

    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
        run_name = ckpt.get('train_args', {}).get('name', '')
        
        is_resumable = 'epoch' in ckpt and 'optimizer' in ckpt and ckpt.get('epoch', -1) >= 0
        
        if is_resumable and run_name:
            print(f"🔧 Patching checkpoint absolute paths to match current HTCondor node...")
            ckpt['train_args']['data'] = current_data_path
            ckpt['train_args']['project'] = current_project_dir
            ckpt['train_args']['save_dir'] = os.path.join(current_project_dir, run_name)
            torch.save(ckpt, filepath)
            
        return run_name, is_resumable
    except Exception as e:
        print(f"⚠️ Corrupted checkpoint found at {filepath}: {e}")
        return None, False

def main():
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

    data_config_path = rewrite_yaml_for_chtc("data_root")
    project_dir = os.path.join(wd, "yolo_results")
    os.makedirs(project_dir, exist_ok=True)
    workers = min(4, os.cpu_count() or 1)

    # ==========================================
    # BULLETPROOF RECOVERY ALGORITHM
    # ==========================================
    print(f"\n--- 🔄 INITIATING BULLETPROOF RECOVERY SCAN ---")
    start_phase = 1
    resume_target_path = None
    can_resume = False
    
    potential_checkpoints = [
        os.path.join(wd, "checkpoint.pt"),
        os.path.join(project_dir, "phase2_cooldown", "weights", "last.pt"),
        os.path.join(project_dir, "phase1_mosaic", "weights", "last.pt")
    ]

    for path in potential_checkpoints:
        run_name, is_resumable = verify_and_patch_checkpoint(path, data_config_path, project_dir)
        if run_name:
            if is_resumable:
                print(f"✅ Found FULL resumable weights at: {path}")
                resume_target_path = path
                can_resume = True
                if run_name == 'phase2_cooldown':
                    print("➡️ Status: Died during Phase 2. Resuming Phase 2.")
                    start_phase = 2
                else:
                    print("➡️ Status: Died during Phase 1. Resuming Phase 1.")
                    start_phase = 1
                break 
            else:
                print(f"⚠️ Found stripped weights at {path}. Missing optimizer memory. Cannot resume.")
            
    if not resume_target_path:
        print("⚠️ No valid resumable checkpoints found. Starting Phase 1 entirely from scratch.")

    # ==========================================
    # PHASE 1: THE MOSAIC BOOTCAMP
    # ==========================================
    if start_phase == 1:
        print("\n--- 📦 STARTING PHASE 1: MOSAIC BOOTCAMP ---")
        if can_resume:
            model_phase1 = YOLO(resume_target_path)
            model_phase1.add_callback("on_fit_epoch_end", log_epoch_stats)
            model_phase1.train(
                resume=True,
                data=data_config_path,
                project=project_dir,
                name="phase1_mosaic",
                epochs=4000
            )
        else:
            model_phase1 = YOLO("yolo11x.pt")
            model_phase1.add_callback("on_fit_epoch_end", log_epoch_stats)
            
            # --- YOUR ORIGINAL PARAMETERS RESTORED ---
            model_phase1.train(
                data=data_config_path,
                project=project_dir,
                name="phase1_mosaic",
                resume=False,
                epochs=4000,
                patience=300,
                imgsz=1536,
                batch=4,
                device=0,
                workers=workers,
                optimizer="AdamW",
                lr0=0.0015,
                lrf=0.01,
                cos_lr=True,
                weight_decay=0.05,
                warmup_epochs=5.0,
                warmup_momentum=0.8,
                max_det=2000,
                iou=0.6,
                box=5.0,
                dfl=3.0,
                cls=3.0,
                mosaic=1.0,
                close_mosaic=0,
                mixup=0.0,
                copy_paste=0.0,
                erasing=0.0,
                degrees=360.0,
                scale=0.5,
                translate=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.5,
                fliplr=0.5,
                hsv_h=0.07,
                hsv_s=0.7,
                hsv_v=0.6,
                amp=True,
                val=True,
                save=True,
                save_period=-1,
                plots=False,
                exist_ok=True
            )

    # ==========================================
    # PHASE 2: REAL-WORLD COOLDOWN
    # ==========================================
    if start_phase <= 2:
        if 'model_phase1' in locals():
            print("\n🧹 Purging Phase 1 model from CUDA memory to prevent OOM...")
            del model_phase1
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n--- 🌍 STARTING PHASE 2: REAL-WORLD COOLDOWN ---")

        if start_phase == 2 and can_resume:
            model_phase2 = YOLO(resume_target_path)
            model_phase2.add_callback("on_fit_epoch_end", log_epoch_stats)
            model_phase2.train(
                resume=True,
                data=data_config_path,
                project=project_dir,
                name="phase2_cooldown",
                epochs=4000
            )
        else:
            phase1_best_weights = os.path.join(project_dir, "phase1_mosaic", "weights", "best.pt")
            if not os.path.exists(phase1_best_weights):
                print("❌ Phase 1 failed to produce best.pt. Aborting Phase 2.")
                return

            print(f"Loading perfect shape-weights from: {phase1_best_weights}")
            model_phase2 = YOLO(phase1_best_weights)
            model_phase2.add_callback("on_fit_epoch_end", log_epoch_stats)

            # --- YOUR ORIGINAL PARAMETERS RESTORED ---
            model_phase2.train(
                data=data_config_path,
                project=project_dir,
                name="phase2_cooldown",
                resume=False,
                epochs=4000,
                patience=300,
                imgsz=1536,
                batch=4,
                device=0,
                workers=workers,
                optimizer="AdamW",
                lr0=0.0005,
                lrf=0.01,
                cos_lr=True,
                weight_decay=0.05,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                max_det=2000,
                iou=0.6,
                box=5.0,
                dfl=3.0,
                cls=3.0,
                mosaic=0.0,
                close_mosaic=0,
                mixup=0.0,
                copy_paste=0.0,
                erasing=0.0,
                degrees=360.0,
                scale=0.15,
                translate=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.5,
                fliplr=0.5,
                hsv_h=0.07,
                hsv_s=0.7,
                hsv_v=0.4,
                amp=True,
                val=True,
                save=True,
                save_period=-1,
                plots=True,
                exist_ok=True
            )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ train_yolo.py failed: {e}", file=sys.stderr)
        raise