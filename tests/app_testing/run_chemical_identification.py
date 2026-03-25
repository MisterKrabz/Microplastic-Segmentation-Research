"""
Chemical Identification App
============================
Runs YOLO + SAM2 on images that have a paired CNN-predicted chemical map CSV.
For each segmented microplastic, the app looks up which grid cells the mask
overlaps and determines the material via majority vote.

Expected folder structure:
    <base>__NN_modified.jpg                            (image to analyse)
    <base>__NN_Raw_predictedMap_cnn_aug.csv            (grid chemical map)
    <base>__NN_Raw_aug_cnn_predicted_*_2nd.png         (visualisation, optional)
"""

import os
import csv
import re
import threading
import tkinter as tk
from tkinter import ttk
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "./../../datasets/ImagesWithPredictedChemicalMap"
YOLO_MODEL_PATH = "./../../models/hunter-yolo-v0.4.4.pt"

SAM2_CHECKPOINT = "./../../models/sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

CONFIDENCE = 0.1
IOU_THRESH = 0.25
IMG_SIZE = 1280

BOX_LINE_WIDTH = 2
LABEL_FONT_SCALE = 0.45
LABEL_FONT_THICKNESS = 1

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

BOX_SHRINK = 0.10
MASK_THRESH = 0.5
MASK_ALPHA = 0.45

ZOOM_STEP = 1.12
MIN_USER_ZOOM = 0.20
MAX_USER_ZOOM = 12.0

MATERIAL_COLORS = {
    "Polystyrene":              (0, 180, 180),
    "Polymethyl Methacrylate":  (0, 140, 255),
    "Polyethylene":             (180, 0, 180),
    "Cotton":                   (80, 200, 80),
    "NA":                       (128, 128, 128),
}

DEFAULT_COLOR = (200, 200, 200)


# ==========================================
# Chemical map helpers
# ==========================================
def load_chemical_map(csv_path: str) -> list[list[str]]:
    """Load a predicted chemical map CSV into a 2-D grid of material strings."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        grid = []
        for row in reader:
            grid.append(row)
    return grid


def classify_mask(
    mask: np.ndarray,
    chem_grid: list[list[str]],
    img_h: int,
    img_w: int,
) -> tuple[str, dict[str, int]]:
    """
    Given a binary mask and a chemical-map grid, determine the material
    of the segmented object by majority vote of the non-NA grid cells
    the mask overlaps.

    Returns (winner_material, vote_counts).
    """
    n_rows = len(chem_grid)
    n_cols = len(chem_grid[0]) if n_rows > 0 else 0
    if n_rows == 0 or n_cols == 0:
        return "Unknown", {}

    cell_h = img_h / n_rows
    cell_w = img_w / n_cols

    mask_bin = (mask > MASK_THRESH).astype(np.uint8) if mask.dtype != np.uint8 else mask
    ys, xs = np.where(mask_bin > 0)

    if len(ys) == 0:
        return "Unknown", {}

    grid_rows = np.clip((ys / cell_h).astype(int), 0, n_rows - 1)
    grid_cols = np.clip((xs / cell_w).astype(int), 0, n_cols - 1)

    cell_indices = set(zip(grid_rows.tolist(), grid_cols.tolist()))

    votes: Counter = Counter()
    for r, c in cell_indices:
        material = chem_grid[r][c].strip() if c < len(chem_grid[r]) else "NA"
        if material and material != "NA":
            votes[material] += 1

    if not votes:
        return "NA", {"NA": len(cell_indices)}

    winner = votes.most_common(1)[0][0]
    return winner, dict(votes)


def color_for_material(material: str) -> tuple[int, int, int]:
    for key, color in MATERIAL_COLORS.items():
        if key.lower() in material.lower():
            return color
    return DEFAULT_COLOR


# ==========================================
# Dataset pairing
# ==========================================
def find_image_csv_pairs(dataset_dir: str) -> list[dict]:
    """
    Scan a directory and pair each *_modified.jpg with its
    *_Raw_predictedMap_cnn_aug.csv by matching the __NN index.
    """
    files = os.listdir(dataset_dir)

    images = {}
    csvs = {}

    for f in files:
        full = os.path.join(dataset_dir, f)
        if f.endswith("_modified.jpg") or f.endswith("_modified.png"):
            m = re.search(r"__(\d+)_modified\.", f)
            if m:
                images[m.group(1)] = full
        elif f.endswith("_predictedMap_cnn_aug.csv"):
            m = re.search(r"__(\d+)_Raw_predictedMap", f)
            if m:
                csvs[m.group(1)] = full

    pairs = []
    for idx in sorted(images.keys()):
        if idx in csvs:
            pairs.append({
                "image_path": images[idx],
                "csv_path": csvs[idx],
                "index": idx,
            })
    return pairs


# ==========================================
# Drawing helpers
# ==========================================
def shrink_box(box_xyxy, w, h, frac):
    if frac <= 0:
        return box_xyxy
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    dx, dy = bw * frac, bh * frac
    x1n = np.clip(x1 + dx, 0, w - 1)
    y1n = np.clip(y1 + dy, 0, h - 1)
    x2n = np.clip(x2 - dx, 0, w - 1)
    y2n = np.clip(y2 - dy, 0, h - 1)
    if x2n <= x1n + 1:
        x1n, x2n = x1, x2
    if y2n <= y1n + 1:
        y1n, y2n = y1, y2
    return np.array([x1n, y1n, x2n, y2n], dtype=np.float32)


def draw_material_overlays(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    materials: list[str],
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """Draw color-coded masks and material labels on the image."""
    out = image_bgr.copy()
    overlay = image_bgr.copy()

    for i, (mask, material) in enumerate(zip(masks, materials)):
        if mask is None:
            continue
        mask_bin = (mask > MASK_THRESH).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue

        color = color_for_material(material)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, -1)
        cv2.drawContours(out, contours, -1, color, 2)

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for i, material in enumerate(materials):
        if i >= len(boxes_xyxy):
            continue
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        color = color_for_material(material)
        conf = confidences[i] if i < len(confidences) else 0.0

        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_LINE_WIDTH)

        label = f"{material} ({conf:.0%})"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
        )

        bg_top = y1 - th - baseline - 10
        bg_bottom = y1
        if bg_top < 0:
            bg_top = y2
            bg_bottom = y2 + th + baseline + 10
            text_y = y2 + th + 4
        else:
            text_y = y1 - 6

        cv2.rectangle(out, (x1, bg_top), (x1 + tw + 10, bg_bottom), color, -1)
        cv2.putText(
            out, label, (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
            (255, 255, 255), LABEL_FONT_THICKNESS, cv2.LINE_AA,
        )

    return out


# ==========================================
# VIEWER
# ==========================================
class ChemicalIdentificationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Chemical Identification | YOLO + SAM2 + CNN Map | {DEVICE}")
        self.root.geometry("1300x950")

        self.pairs = find_image_csv_pairs(DATASET_DIR)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.progress_val = 0.0
        self.show_overlays = True
        self.is_exporting = False
        self._export_done = False
        self._export_progress = (0, 1)

        self.tk_img = None
        self.canvas_img_id = None
        self.user_zoom = 1.0
        self.fit_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_last_x = None
        self._drag_last_y = None

        self.setup_gui()
        self.bind_events()

        if not self.pairs:
            self.lbl_status.config(text=f"No image/CSV pairs found in {DATASET_DIR}")
            self.is_processing = False
            return

        self.thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.thread.start()
        self.check_updates()

    # -----------------------------------------------
    # GUI setup
    # -----------------------------------------------
    def setup_gui(self):
        frame_top = tk.Frame(self.root, pady=5)
        frame_top.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = tk.Label(frame_top, text="Loading models...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()

        self.lbl_summary = tk.Label(frame_top, text="Materials: --", font=("Arial", 11))
        self.lbl_summary.pack()

        self.progress = ttk.Progressbar(frame_top, orient=tk.HORIZONTAL, length=500, mode="determinate")
        self.progress.pack(pady=5)

        self.canvas = tk.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_bot = tk.Frame(self.root, pady=12, bg="#f0f0f0")
        frame_bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_img_info = tk.Label(frame_top, text="", font=("Arial", 9), wraplength=800)
        self.lbl_img_info.pack()

        self.btn_prev = tk.Button(frame_bot, text="<< Prev", command=self.prev_img, state=tk.DISABLED, width=12)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.lbl_counter = tk.Label(frame_bot, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=10)

        self.btn_toggle = tk.Button(
            frame_bot, text="Hide Overlays", command=self.toggle_overlays, width=14
        )
        self.btn_toggle.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(frame_bot, text="Next >>", command=self.next_img, state=tk.DISABLED, width=12)
        self.btn_next.pack(side=tk.RIGHT, padx=10)

        self.btn_export = tk.Button(frame_bot, text="Export All", command=self.export_all, state=tk.DISABLED, width=12)
        self.btn_export.pack(side=tk.RIGHT, padx=10)

        self.lbl_export = tk.Label(frame_top, text="", font=("Arial", 9))
        self.lbl_export.pack()

    def bind_events(self):
        self.root.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)
        self.root.bind("<KeyPress-plus>", self.zoom_in_key)
        self.root.bind("<KeyPress-equal>", self.zoom_in_key)
        self.root.bind("<KeyPress-minus>", self.zoom_out_key)
        self.root.bind("<KeyPress-0>", self.reset_zoom_key)
        self.root.bind("<Command-plus>", self.zoom_in_key)
        self.root.bind("<Command-equal>", self.zoom_in_key)
        self.root.bind("<Command-minus>", self.zoom_out_key)
        self.root.bind("<Command-0>", self.reset_zoom_key)

    # -----------------------------------------------
    # Viewport
    # -----------------------------------------------
    def get_current_pil_image(self):
        if not self.processed_results:
            return None
        data = self.processed_results[self.current_idx]
        return data["image_overlay"] if self.show_overlays else data["image_original"]

    def reset_view(self):
        self.user_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def effective_scale(self, img_w, img_h, cw, ch):
        if img_w <= 0 or img_h <= 0 or cw <= 1 or ch <= 1:
            return 1.0
        fit = min(cw / img_w, ch / img_h)
        self.fit_scale = fit
        return fit * self.user_zoom

    def clamp_pan(self, sw, sh, cw, ch):
        if sw <= cw:
            self.pan_x = (cw - sw) / 2.0
        else:
            self.pan_x = min(max(self.pan_x, cw - sw), 0)
        if sh <= ch:
            self.pan_y = (ch - sh) / 2.0
        else:
            self.pan_y = min(max(self.pan_y, ch - sh), 0)

    # -----------------------------------------------
    # Events
    # -----------------------------------------------
    def on_resize(self, e):
        if e.widget == self.root and self.processed_results:
            self.update_display()

    def on_drag_start(self, e):
        self._drag_last_x, self._drag_last_y = e.x, e.y

    def on_drag_move(self, e):
        if self._drag_last_x is None:
            return
        self.pan_x += e.x - self._drag_last_x
        self.pan_y += e.y - self._drag_last_y
        self._drag_last_x, self._drag_last_y = e.x, e.y
        self.update_display()

    def on_drag_end(self, _):
        self._drag_last_x = self._drag_last_y = None

    def _zoom_modifier(self, e):
        return bool(getattr(e, "state", 0) & 0x001C)

    def zoom_about(self, zoom_in, cx, cy):
        pil = self.get_current_pil_image()
        if pil is None:
            return
        iw, ih = pil.size
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        old_s = self.effective_scale(iw, ih, cw, ch)
        old_uz = self.user_zoom
        self.user_zoom = min(MAX_USER_ZOOM, self.user_zoom * ZOOM_STEP) if zoom_in else max(MIN_USER_ZOOM, self.user_zoom / ZOOM_STEP)
        if abs(self.user_zoom - old_uz) < 1e-12:
            return
        new_s = self.effective_scale(iw, ih, cw, ch)
        ix = (cx - self.pan_x) / old_s
        iy = (cy - self.pan_y) / old_s
        self.pan_x = cx - ix * new_s
        self.pan_y = cy - iy * new_s
        self.update_display()

    def on_mousewheel(self, e):
        if not self.processed_results:
            return
        if self._zoom_modifier(e):
            self.zoom_about(e.delta > 0, e.x_root - self.canvas.winfo_rootx(), e.y_root - self.canvas.winfo_rooty())
        else:
            self.pan_y += 45 if e.delta > 0 else -45
            self.update_display()

    def on_mousewheel_linux(self, e):
        if not self.processed_results:
            return
        cx = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        cy = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        if self._zoom_modifier(e):
            self.zoom_about(e.num == 4, cx, cy)
        else:
            self.pan_y += 45 if e.num == 4 else -45
            self.update_display()

    def zoom_in_key(self, _=None):
        if self.processed_results:
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            self.zoom_about(True, cw / 2, ch / 2)

    def zoom_out_key(self, _=None):
        if self.processed_results:
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            self.zoom_about(False, cw / 2, ch / 2)

    def reset_zoom_key(self, _=None):
        if self.processed_results:
            self.reset_view()
            self.update_display()

    # -----------------------------------------------
    # Pipeline
    # -----------------------------------------------
    def run_pipeline(self):
        try:
            print(f"Loading YOLO: {YOLO_MODEL_PATH}")
            yolo = YOLO(YOLO_MODEL_PATH)
            print(f"Loading SAM2: {SAM2_CHECKPOINT}")
            sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:
            print(f"Model load error: {e}")
            self.is_processing = False
            return

        total = len(self.pairs)

        for idx, pair in enumerate(self.pairs):
            img_path = pair["image_path"]
            csv_path = pair["csv_path"]

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_bgr.shape[:2]

            chem_grid = load_chemical_map(csv_path)

            results = yolo.predict(
                source=img_path, conf=CONFIDENCE, iou=IOU_THRESH,
                imgsz=IMG_SIZE, verbose=False, agnostic_nms=True,
            )
            r0 = results[0]
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32) if len(r0.boxes) > 0 else np.empty((0, 4), dtype=np.float32)
            confs = r0.boxes.conf.cpu().numpy() if len(r0.boxes) > 0 else np.empty((0,), dtype=np.float32)

            pred_masks = []
            materials = []
            vote_details = []

            if len(boxes) > 0:
                predictor.set_image(img_rgb)
                for b in boxes:
                    b_shrunk = shrink_box(b, w, h, BOX_SHRINK)
                    m, _, _ = predictor.predict(
                        point_coords=None, point_labels=None,
                        box=b_shrunk[None, :], multimask_output=False,
                    )
                    m0 = m[0]
                    if m0.ndim == 3:
                        m0 = m0.squeeze(0)
                    pred_masks.append(m0)

                    material, votes = classify_mask(m0, chem_grid, h, w)
                    materials.append(material)
                    vote_details.append(votes)

            overlay_bgr = draw_material_overlays(
                img_bgr, pred_masks, materials, boxes, confs
            )

            pil_overlay = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
            pil_original = Image.fromarray(img_rgb)

            material_counts = Counter(materials)

            self.processed_results.append({
                "image_overlay": pil_overlay,
                "image_original": pil_original,
                "filename": os.path.basename(img_path),
                "csv_file": os.path.basename(csv_path),
                "n_detections": len(boxes),
                "materials": materials,
                "material_counts": material_counts,
                "vote_details": vote_details,
                "confidences": confs,
            })

            self.progress_val = (idx + 1) / total * 100.0
            print(
                f"[{idx + 1}/{total}] {os.path.basename(img_path)}: "
                f"{len(boxes)} detections -> {dict(material_counts)}"
            )

        self.is_processing = False

    # -----------------------------------------------
    # UI loop
    # -----------------------------------------------
    def check_updates(self):
        self.progress["value"] = self.progress_val
        if self.processed_results:
            self.update_display()
        self.update_buttons()

        if self.processed_results and not self.is_exporting:
            self.btn_export.config(state=tk.NORMAL)

        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"Processing: {done}/{len(self.pairs)} images...")
            self.root.after(100, self.check_updates)
        else:
            self.lbl_status.config(text=f"Done! Processed {len(self.processed_results)} images.")
            self.update_buttons()

    def update_display(self):
        pil = self.get_current_pil_image()
        if pil is None:
            return

        iw, ih = pil.size
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = self.effective_scale(iw, ih, cw, ch)
        sw = max(1, int(iw * scale))
        sh = max(1, int(ih * scale))
        self.clamp_pan(sw, sh, cw, ch)

        img = pil.resize((sw, sh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)

        if self.canvas_img_id is None:
            self.canvas.delete("all")
            self.canvas_img_id = self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_img)
            self.canvas.coords(self.canvas_img_id, self.pan_x, self.pan_y)

        data = self.processed_results[self.current_idx]
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.processed_results)}")

        counts = data["material_counts"]
        parts = [f"{mat}: {n}" for mat, n in sorted(counts.items())]
        self.lbl_summary.config(text=f"Detected {data['n_detections']} particles | {', '.join(parts) or 'none'}")

        self.lbl_img_info.config(text=f"{data['filename']}  |  CSV: {data['csv_file']}")
        self.root.title(
            f"Chemical ID | {data['filename']} | "
            f"{data['n_detections']} particles | "
            f"Overlays: {'ON' if self.show_overlays else 'OFF'} | "
            f"Zoom: {self.user_zoom:.2f}x"
        )

    def update_buttons(self):
        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.processed_results) - 1 else tk.DISABLED)

    def toggle_overlays(self):
        self.show_overlays = not self.show_overlays
        self.btn_toggle.config(text="Hide Overlays" if self.show_overlays else "Show Overlays")
        self.update_display()

    def next_img(self):
        if self.current_idx < len(self.processed_results) - 1:
            self.current_idx += 1
            self.reset_view()
            self.update_display()
            self.update_buttons()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.reset_view()
            self.update_display()
            self.update_buttons()

    # -----------------------------------------------
    # Export
    # -----------------------------------------------
    def export_all(self):
        if self.is_exporting or not self.processed_results:
            return
        self.is_exporting = True
        self._export_done = False
        self._export_progress = (0, len(self.processed_results))
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_export.config(text="Exporting...")

        self._export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self._export_thread.start()
        self._poll_export()

    def _export_worker(self):
        from datetime import datetime

        export_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "exports")
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(export_dir, f"chemical_id_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._export_dir = run_dir

        snapshot = list(self.processed_results)
        img_key = "image_overlay" if self.show_overlays else "image_original"

        for i, item in enumerate(snapshot, start=1):
            img: Image.Image = item[img_key]
            base = os.path.splitext(item["filename"])[0]
            suffix = "_overlay" if self.show_overlays else "_original"
            out_path = os.path.join(run_dir, f"{base}{suffix}.png")
            img.save(out_path, format="PNG")
            self._export_progress = (i, len(snapshot))

        self._export_done = True

    def _poll_export(self):
        done, total = self._export_progress
        self.lbl_export.config(text=f"Export: {done}/{total} saved")

        if self._export_done:
            self.is_exporting = False
            self.btn_export.config(state=tk.NORMAL)
            self.lbl_export.config(text=f"Exported {total} images to {self._export_dir}")
            return

        self.root.after(150, self._poll_export)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChemicalIdentificationViewer(root)
    root.mainloop()
