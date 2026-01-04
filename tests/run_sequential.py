import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- NATIVE SAM2 IMPORTS ---
# This uses the standard installed library structure
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
# 1. PATHS
SOURCE_PATH = "./../datasets/raw_data/LMMP"   # Your images folder
YOLO_MODEL_PATH = "../models/hunter-yolo-v0.2.2.pt" # Your YOLO weights (downloaded from CHTC)

# 2. SAM2 CONFIGURATION (NATIVE)
# You must have the .pt model file. 
# The config name is internal to the native library (e.g., "sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"     # Path to your local SAM2 weights file
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml" # Standard config path in the repo

# 3. TUNING
CONFIDENCE = 0.25      # YOLO Confidence
IOU_THRESH = 0.45      # NMS Threshold (removes YOLO double boxes before SAM2 sees them)
IMG_SIZE   = 1280      # Inference size
DEVICE     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# ==========================================

class NativeSAM2Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Native SAM2 + YOLO Viewer | Device: {DEVICE}")
        self.root.geometry("1200x900")
        
        # State
        self.image_files = self.get_image_list(SOURCE_PATH)
        self.processed_results = [] 
        self.current_idx = 0
        self.is_processing = True
        
        # GUI Setup
        self.setup_gui()

        # Validation
        if not os.path.exists(YOLO_MODEL_PATH):
            self.lbl_status.config(text=f"❌ MISSING YOLO WEIGHTS: {YOLO_MODEL_PATH}")
            return
        if not os.path.exists(SAM2_CHECKPOINT):
            self.lbl_status.config(text=f"❌ MISSING SAM2 WEIGHTS: {SAM2_CHECKPOINT}")
            return

        # Start Pipeline
        self.thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.thread.start()
        self.check_updates()

    def setup_gui(self):
        # Top Bar
        frame_top = tk.Frame(self.root, pady=5)
        frame_top.pack(side=tk.TOP, fill=tk.X)
        self.lbl_status = tk.Label(frame_top, text="Loading Models...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()
        self.progress = ttk.Progressbar(frame_top, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # Canvas
        self.lbl_image = tk.Label(self.root, bg="#202020", text="Waiting for Pipeline...", fg="white")
        self.lbl_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom Bar
        frame_bot = tk.Frame(self.root, pady=15, bg="#f0f0f0")
        frame_bot.pack(side=tk.BOTTOM, fill=tk.X)
        self.btn_prev = tk.Button(frame_bot, text="<< Prev", command=self.prev_img, state=tk.DISABLED, width=15)
        self.btn_prev.pack(side=tk.LEFT, padx=20)
        self.lbl_count = tk.Label(frame_bot, text="0 / 0", font=("Arial", 10))
        self.lbl_count.pack(side=tk.LEFT, padx=20)
        self.btn_next = tk.Button(frame_bot, text="Next >>", command=self.next_img, state=tk.DISABLED, width=15)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

    def get_image_list(self, path):
        exts = ('.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg')
        paths = []
        if os.path.isfile(path): return [path]
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(exts): paths.append(os.path.join(root, f))
        return sorted(paths)

    def run_pipeline(self):
        try:
            # 1. Load YOLO
            print(f"🔹 Loading YOLO: {YOLO_MODEL_PATH}")
            yolo = YOLO(YOLO_MODEL_PATH)

            # 2. Load Native SAM2
            print(f"🔹 Loading Native SAM2: {SAM2_CHECKPOINT}")
            # Native build_sam2 takes the config STRING and the checkpoint PATH
            sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
            predictor = SAM2ImagePredictor(sam2_model)

        except Exception as e:
            print(f"❌ MODEL LOAD ERROR: {e}")
            self.lbl_status.config(text=f"Error: {e}")
            return

        total = len(self.image_files)
        
        for i, img_path in enumerate(self.image_files):
            # A. Prepare Image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # B. YOLO Inference (Get Boxes)
            results = yolo.predict(img_rgb, conf=CONFIDENCE, iou=IOU_THRESH, imgsz=IMG_SIZE, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]

            # C. SAM2 Inference (Use Boxes as Prompts)
            final_overlay = img_bgr.copy()
            
            if len(boxes) > 0:
                predictor.set_image(img_rgb)
                
                # Native SAM2 can handle a list of boxes at once
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes, 
                    multimask_output=False
                )
                
                # D. Draw Results (Masks + Boxes)
                final_overlay = self.draw_results(final_overlay, boxes, masks)

            # E. Save for Display
            pil_img = Image.fromarray(cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB))
            self.processed_results.append({
                "image": pil_img, 
                "name": os.path.basename(img_path),
                "count": len(boxes)
            })
            
            self.progress_val = (i+1)/total * 100

        self.is_processing = False

    def draw_results(self, image, boxes, masks):
        overlay = image.copy()
        
        # Ensure masks dimension is correct (N, H, W)
        if masks.ndim == 4: masks = masks.squeeze(1)

        for box, mask in zip(boxes, masks):
            # Random color for each object
            color = np.random.randint(0, 255, (3,), dtype=int).tolist()
            
            # Draw Mask (Filled with transparency)
            mask_u8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, -1)
            cv2.drawContours(image, contours, -1, color, 2) # Border

            # Draw Box (White)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Blend
        return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    def check_updates(self):
        if hasattr(self, 'progress_val'): self.progress['value'] = self.progress_val
        
        if self.processed_results and self.lbl_image.cget("text") != "":
            self.update_display()
        
        self.update_buttons()
        
        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"Processing: {done}/{len(self.image_files)}...")
            self.root.after(100, self.check_updates)
        else:
            self.lbl_status.config(text=f"✅ Done! {len(self.image_files)} images processed.")

    def update_display(self):
        data = self.processed_results[self.current_idx]
        
        # Resize logic
        cw, ch = self.root.winfo_width(), self.root.winfo_height() - 150
        if cw < 100: cw, ch = 800, 600
        
        img = data["image"].copy()
        img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=self.tk_img, text="")
        self.lbl_count.config(text=f"{self.current_idx + 1} / {len(self.processed_results)}")
        self.root.title(f"Native SAM2 | {data['name']} | Objects: {data['count']}")

    def update_buttons(self):
        state_prev = tk.NORMAL if self.current_idx > 0 else tk.DISABLED
        state_next = tk.NORMAL if self.current_idx < len(self.processed_results) - 1 else tk.DISABLED
        self.btn_prev.config(state=state_prev)
        self.btn_next.config(state=state_next)

    def next_img(self):
        if self.current_idx < len(self.processed_results) - 1:
            self.current_idx += 1
            self.update_display()
            self.update_buttons()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            self.update_buttons()

if __name__ == "__main__":
    root = tk.Tk()
    app = NativeSAM2Viewer(root)
    root.mainloop()