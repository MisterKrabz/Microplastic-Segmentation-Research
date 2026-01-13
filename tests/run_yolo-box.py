import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from PIL import Image  # (already implied by Image usage, leaving minimal)

# ==========================================
# CONFIGURATION
# ==========================================
# Can be a single file OR a folder (scanned recursively)
SOURCE_PATH = "./../datasets/Microplastics-Bounding-Box"
MODEL_PATH = "../models/baseline_bbox_v3.pt"

# Tuning Parameters
CONFIDENCE = 0.3
IOU_THRESH = 0.25
IMG_SIZE   = 1280
# ==========================================


# =========================
# NEW: GT counting helpers
# =========================
def label_for_image(img_path: str) -> str:
    """
    Given .../images/<name>.<ext>  ->  .../labels/<name>.txt
    If 'images' is not in the path, fallback to same folder, .txt extension.
    """
    p = os.path.normpath(img_path)
    parts = p.split(os.sep)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_path = os.sep.join(parts)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        return label_path

    # fallback: same directory, same base name, .txt
    return os.path.splitext(p)[0] + ".txt"


def count_gt_instances(label_path: str) -> int:
    """GT count = number of non-empty lines in label file (each line = 1 instance)."""
    if not os.path.exists(label_path):
        return 0
    n = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def count_accuracy(gt: int, pred: int) -> float:
    """
    Accuracy based on count agreement:
    - If gt == 0: accuracy = 1 if pred == 0 else 0
    - Else: 1 - |pred-gt|/gt (clamped to [0,1])
    """
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    acc = 1.0 - (abs(pred - gt) / gt)
    if acc < 0:
        acc = 0.0
    if acc > 1:
        acc = 1.0
    return acc


class MicroplasticViewer:
    def __init__(self, root, image_source, model_path):
        self.root = root
        self.root.title("Microplastic Detection Viewer")
        self.root.geometry("1000x800")

        # Data State
        self.image_files = self.get_image_list(image_source)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.model_path = model_path

        # NEW: running totals for overall accuracy
        self.total_gt = 0
        self.total_pred = 0
        self.sum_img_acc = 0.0
        self.n_imgs = 0

        # --- GUI LAYOUT ---

        # 1. Top Bar: Status Info
        self.status_frame = tk.Frame(root, pady=5)
        self.status_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = tk.Label(self.status_frame, text="Initializing...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()

        # NEW: overall accuracy label (top bar)
        self.lbl_accuracy = tk.Label(self.status_frame, text="Accuracy: --", font=("Arial", 11))
        self.lbl_accuracy.pack()

        self.progress = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # 2. Main Area: Image Display
        self.canvas_frame = tk.Frame(root, bg="#2b2b2b")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lbl_image = tk.Label(self.canvas_frame, bg="#2b2b2b", text="Waiting for YOLO...", fg="white")
        self.lbl_image.pack(expand=True, fill=tk.BOTH)

        # 3. Bottom Bar: Navigation Controls
        self.nav_frame = tk.Frame(root, pady=15, bg="#f0f0f0")
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(self.nav_frame, text="<< Previous", command=self.show_prev, state=tk.DISABLED, width=15)
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.lbl_counter = tk.Label(self.nav_frame, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=20)

        # NEW: per-image accuracy + Pred/GT (bottom bar)
        self.lbl_img_metrics = tk.Label(self.nav_frame, text="Img: Acc -- | Pred/GT --/--", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_img_metrics.pack(side=tk.LEFT, padx=20)

        self.btn_next = tk.Button(self.nav_frame, text="Next >>", command=self.show_next, state=tk.DISABLED, width=15)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

        # --- START PROCESSING ---
        if not self.image_files:
            self.lbl_status.config(text=f"❌ Error: No images found at {image_source}")
            return

        # Start the background thread for YOLO
        self.thread = threading.Thread(target=self.run_inference_thread, daemon=True)
        self.thread.start()

        # Start the GUI update looper
        self.check_for_updates()

    def get_image_list(self, path):
        """
        MODIFIED: Now uses os.walk to recursively find images in
        folders of folders of folders.
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_paths = []

        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        full_path = os.path.join(root, file)
                        image_paths.append(full_path)
            return sorted(image_paths)
        return []

    def run_inference_thread(self):
        """Runs in background: Loads model, processes images one by one."""
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        total = len(self.image_files)

        for i, img_path in enumerate(self.image_files):
            results = model.predict(
                source=img_path,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                verbose=False,
                agnostic_nms=True
            )

            # Process Result for Display (UNCHANGED)
            res_plot = results[0].plot(line_width=2, font_size=2)
            res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(res_rgb)

            # Pred count (UNCHANGED logic)
            pred_count = len(results[0].boxes)

            # NEW: GT count from label
            lbl_path = label_for_image(img_path)
            gt_count = count_gt_instances(lbl_path)

            # NEW: per-image accuracy
            img_acc = count_accuracy(gt_count, pred_count)

            # NEW: update running totals
            self.total_gt += gt_count
            self.total_pred += pred_count
            self.sum_img_acc += img_acc
            self.n_imgs += 1

            filename = os.path.basename(img_path)

            self.processed_results.append({
                "image": pil_img,
                "pred_count": pred_count,
                "gt_count": gt_count,
                "img_acc": img_acc,
                "filename": filename,
                "label_path": lbl_path
            })

            self.progress_val = (i + 1) / total * 100

        self.is_processing = False

    def check_for_updates(self):
        """Polls to see if new results arrived from the thread."""

        if hasattr(self, 'progress_val'):
            self.progress['value'] = self.progress_val

        if len(self.processed_results) > 0 and self.lbl_image.cget("image") == "":
            self.update_display()

        self.update_buttons()

        # NEW: update overall accuracy label live
        if self.n_imgs > 0:
            mean_acc = (self.sum_img_acc / self.n_imgs) * 100.0
            self.lbl_accuracy.config(
                text=f"Accuracy (mean per-image): {mean_acc:.1f}% | Total Pred/GT: {self.total_pred}/{self.total_gt}"
            )
        else:
            self.lbl_accuracy.config(text="Accuracy: --")

        total = len(self.image_files)
        processed = len(self.processed_results)

        if self.is_processing:
            self.lbl_status.config(text=f"⚙️ Processing: {processed}/{total} images ready...")
            self.root.after(100, self.check_for_updates)
        else:
            self.lbl_status.config(text=f"✅ Complete! Processed {total} images.")
            self.update_buttons()

    def update_display(self):
        """Updates the main image area with the current index."""
        if not self.processed_results:
            return

        data = self.processed_results[self.current_idx]
        pil_image = data["image"]

        canvas_width = self.root.winfo_width()
        canvas_height = self.root.winfo_height() - 150

        if canvas_width < 10: canvas_width = 800
        if canvas_height < 10: canvas_height = 600

        img_copy = pil_image.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img_copy)

        self.lbl_image.config(image=self.tk_image, text="")
        self.lbl_counter.config(text=f"Image {self.current_idx + 1} of {len(self.image_files)}")

        # NEW: per-image metrics in bottom bar
        img_acc = data.get("img_acc", None)
        pred = data.get("pred_count", None)
        gt = data.get("gt_count", None)
        if img_acc is None or pred is None or gt is None:
            self.lbl_img_metrics.config(text="Img: Acc -- | Pred/GT --/--")
        else:
            self.lbl_img_metrics.config(text=f"Img: Acc {img_acc * 100:.1f}% | Pred/GT {pred}/{gt}")

        # Keep your existing title behavior, just add GT
        self.root.title(f"Viewer - {data['filename']} | Pred: {data['pred_count']}  GT: {data['gt_count']}")

    def update_buttons(self):
        """Enables/Disables buttons based on available data."""
        if self.current_idx > 0:
            self.btn_prev.config(state=tk.NORMAL)
        else:
            self.btn_prev.config(state=tk.DISABLED)

        if self.current_idx < len(self.processed_results) - 1:
            self.btn_next.config(state=tk.NORMAL)
        else:
            self.btn_next.config(state=tk.DISABLED)

    def show_next(self):
        if self.current_idx < len(self.processed_results) - 1:
            self.current_idx += 1
            self.update_display()
            self.update_buttons()

    def show_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            self.update_buttons()

def main():
    root = tk.Tk()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    app = MicroplasticViewer(root, SOURCE_PATH, MODEL_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()
