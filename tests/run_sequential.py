import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime

# --- NATIVE SAM2 IMPORTS ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_PATH = "./../datasets/ES&T2024_LakeMendota"   
YOLO_MODEL_PATH = "../models/yolo-v0.2.4.pt"

# SAM2
SAM2_CHECKPOINT = "../models/sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# YOLO Tuning
CONFIDENCE = 0.3
IOU_THRESH = 0.25
IMG_SIZE   = 1280

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

SPLITS = ["train", "valid", "test"]
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

BOX_SHRINK = 0.10
MASK_THRESH = 0.5
MASK_ALPHA = 0.45

# Export config
EXPORT_ROOT_NAME = "_exports"  # inside SOURCE_PATH
EXPORT_PREFIX = "yolo_sam2_overlay"  # folder name prefix


# ==========================================
# GT counting helpers
# ==========================================
def label_for_image(img_path: str) -> str:
    p = os.path.normpath(img_path)
    parts = p.split(os.sep)

    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_path = os.sep.join(parts)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        return label_path

    return os.path.splitext(p)[0] + ".txt"


def count_gt_instances(label_path: str) -> int:
    if not os.path.exists(label_path):
        return 0
    n = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def count_accuracy(gt: int, pred: int) -> float:
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    acc = 1.0 - (abs(pred - gt) / gt)
    return max(0.0, min(1.0, acc))


# ==========================================
# SAM2 + drawing helpers
# ==========================================
def shrink_box_xyxy(box_xyxy: np.ndarray, w: int, h: int, frac: float) -> np.ndarray:
    if frac <= 0:
        return box_xyxy

    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    dx = bw * frac
    dy = bh * frac

    x1n = np.clip(x1 + dx, 0, w - 1)
    y1n = np.clip(y1 + dy, 0, h - 1)
    x2n = np.clip(x2 - dx, 0, w - 1)
    y2n = np.clip(y2 - dy, 0, h - 1)

    if x2n <= x1n + 1:
        x1n, x2n = x1, x2
    if y2n <= y1n + 1:
        y1n, y2n = y1, y2

    return np.array([x1n, y1n, x2n, y2n], dtype=np.float32)


def draw_masks_on_top(base_bgr: np.ndarray, masks: list[np.ndarray], alpha: float = MASK_ALPHA) -> np.ndarray:
    out = base_bgr.copy()
    overlay = base_bgr.copy()

    for mask in masks:
        if mask is None:
            continue

        mask_bin = (mask > MASK_THRESH).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue

        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, -1)
        cv2.drawContours(out, contours, -1, color, 2)

    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


# ==========================================
# MAIN VIEWER
# ==========================================
class NativeSAM2YOLOViewer:
    def __init__(self, root):
        self.root = root
        self.root.title(f"YOLO (boxes+accuracy) + SAM2 (masks) | Device: {DEVICE}")
        self.root.geometry("1200x900")

        self.image_files = self.get_image_list(SOURCE_PATH)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.progress_val = 0.0

        self.total_gt = 0
        self.total_pred = 0
        self.sum_img_acc = 0.0
        self.n_imgs = 0

        self.export_dir = None
        self.export_thread = None
        self.is_exporting = False

        self.setup_gui()

        if not self.image_files:
            self.lbl_status.config(text=f"❌ Error: No images found at {SOURCE_PATH}")
            return

        if not os.path.exists(YOLO_MODEL_PATH):
            self.lbl_status.config(text=f"❌ MISSING YOLO WEIGHTS: {YOLO_MODEL_PATH}")
            return
        if not os.path.exists(SAM2_CHECKPOINT):
            self.lbl_status.config(text=f"❌ MISSING SAM2 WEIGHTS: {SAM2_CHECKPOINT}")
            return

        self.thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.thread.start()
        self.check_updates()

    # -------------------------
    # GUI
    # -------------------------
    def setup_gui(self):
        frame_top = tk.Frame(self.root, pady=5)
        frame_top.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = tk.Label(frame_top, text="Loading Models...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()

        self.lbl_accuracy = tk.Label(frame_top, text="Accuracy: --", font=("Arial", 11))
        self.lbl_accuracy.pack()

        self.progress = ttk.Progressbar(frame_top, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # Export status line
        self.lbl_export = tk.Label(frame_top, text="Export: --", font=("Arial", 10))
        self.lbl_export.pack()

        self.lbl_image = tk.Label(self.root, bg="#202020", text="Waiting for Pipeline...", fg="white")
        self.lbl_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_bot = tk.Frame(self.root, pady=15, bg="#f0f0f0")
        frame_bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(frame_bot, text="<< Prev", command=self.prev_img, state=tk.DISABLED, width=15)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.lbl_counter = tk.Label(frame_bot, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=10)

        self.lbl_img_metrics = tk.Label(frame_bot, text="Img: Acc -- | Pred/GT --/--", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_img_metrics.pack(side=tk.LEFT, padx=10)

        # NEW: Export button
        self.btn_export = tk.Button(
            frame_bot,
            text="Export All (YOLO+SAM2)",
            command=self.export_all_images,
            state=tk.DISABLED,
            width=22
        )
        self.btn_export.pack(side=tk.RIGHT, padx=10)

        self.btn_next = tk.Button(frame_bot, text="Next >>", command=self.next_img, state=tk.DISABLED, width=15)
        self.btn_next.pack(side=tk.RIGHT, padx=10)

    # -------------------------
    # Dataset scanning
    # -------------------------
    def get_image_list(self, dataset_root):
        image_paths = []

        # If user passed a single file, keep same behavior
        if os.path.isfile(dataset_root):
            if dataset_root.lower().endswith(VALID_EXTS):
                return [dataset_root]
            return []

        # NEW: Walk the entire folder recursively and pick only image files
        if os.path.isdir(dataset_root):
            for r, _, files in os.walk(dataset_root):
                for f in files:
                    if f.lower().endswith(VALID_EXTS):
                        image_paths.append(os.path.join(r, f))

        return sorted(image_paths)

    # -------------------------
    # Pipeline
    # -------------------------
    def run_pipeline(self):
        try:
            print(f"🔹 Loading YOLO: {YOLO_MODEL_PATH}")
            yolo = YOLO(YOLO_MODEL_PATH)

            print(f"🔹 Loading Native SAM2: {SAM2_CHECKPOINT}")
            sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
            predictor = SAM2ImagePredictor(sam2_model)

        except Exception as e:
            print(f"❌ MODEL LOAD ERROR: {e}")
            self.lbl_status.config(text=f"Error: {e}")
            self.is_processing = False
            return

        total = len(self.image_files)

        for i, img_path in enumerate(self.image_files):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_bgr.shape[:2]

            results = yolo.predict(
                source=img_path,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                verbose=False,
                agnostic_nms=True
            )
            r0 = results[0]

            pred_count = len(r0.boxes)

            lbl_path = label_for_image(img_path)
            gt_count = count_gt_instances(lbl_path)

            img_acc = count_accuracy(gt_count, pred_count)

            self.total_gt += gt_count
            self.total_pred += pred_count
            self.sum_img_acc += img_acc
            self.n_imgs += 1

            base_bgr = r0.plot(line_width=2, font_size=2)

            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)

            masks_for_draw = []
            if len(boxes) > 0:
                predictor.set_image(img_rgb)

                for b in boxes:
                    b = shrink_box_xyxy(b, w=w, h=h, frac=BOX_SHRINK)

                    m, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=b[None, :],
                        multimask_output=False
                    )

                    m0 = m[0]
                    if m0.ndim == 3:
                        m0 = m0.squeeze(0)
                    masks_for_draw.append(m0)

            final_bgr = draw_masks_on_top(base_bgr, masks_for_draw, alpha=MASK_ALPHA)

            pil_img = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

            self.processed_results.append({
                "image": pil_img,  # <- EXACTLY what you see will be exported
                "filename": os.path.basename(img_path),
                "pred_count": pred_count,
                "gt_count": gt_count,
                "img_acc": img_acc,
                "label_path": lbl_path,
                "src_path": img_path,  # keep original path for export naming if desired
            })

            self.progress_val = (i + 1) / total * 100.0

        self.is_processing = False

    # -------------------------
    # Export logic
    # -------------------------
    def make_export_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Put exports inside dataset root so it's easy to find
        export_root = os.path.join(SOURCE_PATH, EXPORT_ROOT_NAME)
        os.makedirs(export_root, exist_ok=True)

        export_dir = os.path.join(export_root, f"{EXPORT_PREFIX}_{ts}")
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_all_images(self):
        if self.is_exporting:
            return

        # If nothing is processed yet, do nothing
        if len(self.processed_results) == 0:
            self.lbl_export.config(text="Export: nothing ready yet.")
            return

        self.export_dir = self.make_export_dir()
        self.is_exporting = True
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_export.config(text=f"Export: writing to {self.export_dir}")

        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        self._poll_export_done()

    def _export_worker(self):
        # Export whatever is ready at click time (no re-run)
        results_snapshot = list(self.processed_results)

        for idx, item in enumerate(results_snapshot, start=1):
            img: Image.Image = item["image"]
            src_path = item.get("src_path", "")
            base = os.path.splitext(os.path.basename(src_path))[0] if src_path else os.path.splitext(item["filename"])[0]

            out_name = f"{base}_yolo_sam2.png"
            out_path = os.path.join(self.export_dir, out_name)

            # Always PNG to preserve overlays cleanly
            img.save(out_path, format="PNG")

            if idx % 10 == 0 or idx == len(results_snapshot):
                self._export_progress = (idx, len(results_snapshot))

        self._export_progress = (len(results_snapshot), len(results_snapshot))
        self._export_done = True

    def _poll_export_done(self):
        if not hasattr(self, "_export_done"):
            self._export_done = False
        if not hasattr(self, "_export_progress"):
            self._export_progress = (0, max(1, len(self.processed_results)))

        done, total = self._export_progress
        self.lbl_export.config(text=f"Export: {done}/{total} saved → {self.export_dir}")

        if self._export_done:
            self.is_exporting = False
            self._export_done = False
            self.btn_export.config(state=tk.NORMAL)  # allow exporting again
            return

        self.root.after(150, self._poll_export_done)

    # -------------------------
    # UI update loop
    # -------------------------
    def check_updates(self):
        self.progress["value"] = self.progress_val

        if self.processed_results and self.lbl_image.cget("text") != "":
            self.update_display()

        self.update_buttons()

        if self.n_imgs > 0:
            mean_acc = (self.sum_img_acc / self.n_imgs) * 100.0
            self.lbl_accuracy.config(
                text=f"Accuracy (mean per-image): {mean_acc:.1f}% | Total Pred/GT: {self.total_pred}/{self.total_gt}"
            )
        else:
            self.lbl_accuracy.config(text="Accuracy: --")

        # Enable export as soon as at least 1 image is ready
        if len(self.processed_results) > 0 and not self.is_exporting:
            self.btn_export.config(state=tk.NORMAL)

        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"⚙️ Processing: {done}/{len(self.image_files)} images ready...")
            self.root.after(100, self.check_updates)
        else:
            self.lbl_status.config(text=f"✅ Done! Processed {len(self.image_files)} images.")
            self.update_buttons()
            if len(self.processed_results) > 0 and not self.is_exporting:
                self.btn_export.config(state=tk.NORMAL)

    def update_display(self):
        data = self.processed_results[self.current_idx]

        cw, ch = self.root.winfo_width(), self.root.winfo_height() - 150
        if cw < 100:
            cw, ch = 800, 600

        img = data["image"].copy()
        img.thumbnail((cw, ch), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=self.tk_img, text="")
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.processed_results)}")

        pred = data.get("pred_count", None)
        gt = data.get("gt_count", None)
        img_acc = data.get("img_acc", None)
        if pred is None or gt is None or img_acc is None:
            self.lbl_img_metrics.config(text="Img: Acc -- | Pred/GT --/--")
        else:
            self.lbl_img_metrics.config(text=f"Img: Acc {img_acc * 100:.1f}% | Pred/GT {pred}/{gt}")

        self.root.title(
            f"YOLO+SAM2 | {data['filename']} | Pred: {pred}  GT: {gt}"
        )

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
    app = NativeSAM2YOLOViewer(root)
    root.mainloop()
