"""
Chemical Identification App
============================
Runs YOLO + SAM2 on images that have a paired CNN-predicted chemical map CSV.
For each segmented microplastic, the app looks up which grid cells the mask
overlaps and determines the material via majority vote.

Toggle controls for YOLO boxes, SAM2 masks, and labels independently.
Hovering over a detection reveals its hidden elements even when toggled off.
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
DATASET_DIR = "./../../datasets/testing_datasets/Overlay"
YOLO_MODEL_PATH = "./../../models/hunter-yolo-v0.5.4/hunter-yolo-v0.5.4.pt"

SAM2_CHECKPOINT = "./../../models/sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

CONFIDENCE = 0.2
IOU_THRESH = .7
IMG_SIZE = 1536

BOX_LINE_WIDTH = 1
LABEL_FONT_SCALE = 0.30
LABEL_FONT_THICKNESS = 1

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BOX_SHRINK = 0.0
MASK_THRESH = 0.5
MASK_ALPHA = 0.45

ZOOM_STEP = 1.12
MIN_USER_ZOOM = 0.20
MAX_USER_ZOOM = 12.0

MATERIAL_COLORS = {
    "Polystyrene":              (0, 192, 255),    # #ffc000
    "Polymethyl Methacrylate":  (141, 139, 15),   # #0f8b8d
    "Polyethylene":             (49, 125, 237),    # #ed7d31
    "Cotton":                   (80, 200, 80),
    "NA":                       (128, 128, 128),
}
DEFAULT_COLOR = (200, 200, 200)
RED_BGR  = (0, 0, 255)
BLUE_BGR = (255, 0, 0)


# ==========================================
# Helpers
# ==========================================
def load_chemical_map(csv_path: str) -> list[list[str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader]


_NA_VARIANTS = {"na", "n/a", "nan", "none", "unknown", ""}
_LOW_PRIORITY_VARIANTS = {"polycarbonate"}

HIGH_PRIORITY_MULTIPLIER = 3


def _is_na(material: str) -> bool:
    """True for any string that represents an undefined / missing chemical."""
    return material.strip().lower() in _NA_VARIANTS


def _is_low_priority(material: str) -> bool:
    """True for NA-like or substrate materials (e.g. Polycarbonate)."""
    s = material.strip().lower()
    return s in _NA_VARIANTS or s in _LOW_PRIORITY_VARIANTS


def classify_mask(mask, chem_grid, img_h, img_w):
    """Determine the single definitive material for a SAM2 mask by weighted
    pixel-area vote against the CNN chemical map grid.

    Priority scoring
    ────────────────
    1. Count how many mask pixels fall on each material in the chemical grid.
    2. Low-priority materials (NA, Polycarbonate) keep their raw pixel count
       as the score.
    3. High-priority materials (all other real chemicals) have their pixel
       count multiplied by HIGH_PRIORITY_MULTIPLIER (3×).
    4. The material with the highest final score wins.
    """
    n_rows = len(chem_grid)
    n_cols = len(chem_grid[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return "NA", {}

    cell_h, cell_w = img_h / n_rows, img_w / n_cols
    mask_bin = (mask > MASK_THRESH).astype(np.uint8) if mask.dtype != np.uint8 else mask
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return "NA", {}

    grid_rows = np.clip((ys / cell_h).astype(int), 0, n_rows - 1)
    grid_cols = np.clip((xs / cell_w).astype(int), 0, n_cols - 1)

    unique_cells, cell_pixel_counts = np.unique(
        np.stack([grid_rows, grid_cols], axis=1), axis=0, return_counts=True
    )

    pixel_votes: Counter = Counter()
    for (r, c), count in zip(unique_cells.tolist(), cell_pixel_counts.tolist()):
        raw = chem_grid[r][c].strip() if c < len(chem_grid[r]) else ""
        label = "NA" if _is_na(raw) else raw
        pixel_votes[label] += count

    if not pixel_votes:
        return "NA", {}

    weighted_scores = {}
    for material, area in pixel_votes.items():
        if _is_low_priority(material):
            weighted_scores[material] = area
        else:
            weighted_scores[material] = area * HIGH_PRIORITY_MULTIPLIER

    winner = max(weighted_scores, key=weighted_scores.get)
    return winner, dict(pixel_votes)


def color_for_material(material):
    for key, color in MATERIAL_COLORS.items():
        if key.lower() in material.lower():
            return color
    return DEFAULT_COLOR


def find_image_csv_pairs(dataset_dir):
    files = os.listdir(dataset_dir)
    images, csvs = {}, {}
    for f in files:
        full = os.path.join(dataset_dir, f)
        if f.endswith("_modified.jpg") or f.endswith("_modified.png"):
            # key = full stem before "_modified.*" — unique across different series
            key = re.sub(r"_modified\.(jpg|png)$", "", f, flags=re.IGNORECASE)
            images[key] = full
        elif f.endswith("_predictedMap_cnn_aug.csv"):
            key = re.sub(r"_Raw_predictedMap_cnn_aug\.csv$", "", f, flags=re.IGNORECASE)
            csvs[key] = full
    pairs = []
    for key in sorted(images):
        if key in csvs:
            pairs.append({"image_path": images[key], "csv_path": csvs[key], "index": key})
    return pairs


def render_chemical_map_image(chem_grid, img_h, img_w):
    """Render the CNN-predicted chemical map as a BGR image sized to the original."""
    n_rows = len(chem_grid)
    n_cols = len(chem_grid[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cell_h = img_h / n_rows
    cell_w = img_w / n_cols
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            mat = chem_grid[r][c].strip() if c < len(chem_grid[r]) else "NA"
            color = color_for_material(mat)
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            img[y1:y2, x1:x2] = color
    return img


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


# ==========================================
# Compositing
# ==========================================
def compose_image(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    materials: list[str],
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    *,
    show_yolo: bool = True,
    show_sam2: bool = True,
    show_labels: bool = True,
    use_material_colors: bool = True,
    hover_idx: int = -1,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """Compose the display image based on toggle states and hover."""
    out = image_bgr.copy()
    overlay = image_bgr.copy()

    for i, (mask, material) in enumerate(zip(masks, materials)):
        if mask is None:
            continue
        visible = show_sam2 or i == hover_idx
        if not visible:
            continue
        mask_bin = (mask > MASK_THRESH).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue
        color = color_for_material(material) if use_material_colors else BLUE_BGR
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, -1)
        cv2.drawContours(out, contours, -1, color, 2)

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for i in range(min(len(materials), len(boxes_xyxy))):
        material = materials[i]
        color = color_for_material(material) if use_material_colors else RED_BGR
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        conf = confidences[i] if i < len(confidences) else 0.0

        if show_yolo or i == hover_idx:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_LINE_WIDTH)

        if show_labels or i == hover_idx:
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

        self.show_yolo = True
        self.show_sam2 = True
        self.show_labels = True
        self.use_material_colors = True
        self.use_chem_bg = False
        self.hover_idx = -1

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
        self._is_dragging = False

        self._cache_key = None
        self._cached_pil = None

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
    # GUI
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

        self.lbl_img_info = tk.Label(frame_top, text="", font=("Arial", 9), wraplength=800)
        self.lbl_img_info.pack()

        self.lbl_export = tk.Label(frame_top, text="", font=("Arial", 9))
        self.lbl_export.pack()

        self.canvas = tk.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_bot = tk.Frame(self.root, pady=12, bg="#f0f0f0")
        frame_bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(frame_bot, text="<< Prev", command=self.prev_img, state=tk.DISABLED, width=10)
        self.btn_prev.pack(side=tk.LEFT, padx=6)

        self.lbl_counter = tk.Label(frame_bot, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=6)

        self.btn_yolo = tk.Button(frame_bot, text="Toggle YOLO", command=self.toggle_yolo, width=12)
        self.btn_yolo.pack(side=tk.LEFT, padx=4)

        self.btn_sam2 = tk.Button(frame_bot, text="Toggle SAM2", command=self.toggle_sam2, width=12)
        self.btn_sam2.pack(side=tk.LEFT, padx=4)

        self.btn_labels = tk.Button(frame_bot, text="Toggle Labels", command=self.toggle_labels, width=12)
        self.btn_labels.pack(side=tk.LEFT, padx=4)

        self.btn_colors = tk.Button(frame_bot, text="Toggle Colors", command=self.toggle_colors, width=12)
        self.btn_colors.pack(side=tk.LEFT, padx=4)

        self.btn_background = tk.Button(frame_bot, text="Toggle Background", command=self.toggle_background, width=16)
        self.btn_background.pack(side=tk.LEFT, padx=4)

        self.btn_next = tk.Button(frame_bot, text="Next >>", command=self.next_img, state=tk.DISABLED, width=10)
        self.btn_next.pack(side=tk.RIGHT, padx=6)

        self.btn_export = tk.Button(frame_bot, text="Export All", command=self.export_all, state=tk.DISABLED, width=10)
        self.btn_export.pack(side=tk.RIGHT, padx=6)

        self.lbl_hover = tk.Label(frame_bot, text="", font=("Arial", 9), bg="#f0f0f0", anchor="w")
        self.lbl_hover.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

    def _update_toggle_btn(self, btn, label, state):
        btn.config(text=f"Toggle {label}")

    def bind_events(self):
        self.root.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
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
    # Compositing
    # -----------------------------------------------
    def _cache_compose_key(self):
        return (self.current_idx, self.show_yolo, self.show_sam2, self.show_labels,
                self.use_material_colors, self.use_chem_bg, self.hover_idx)

    def get_current_pil_image(self):
        if not self.processed_results:
            return None

        key = self._cache_compose_key()
        if self._cache_key == key and self._cached_pil is not None:
            return self._cached_pil

        data = self.processed_results[self.current_idx]
        base_img = data["chem_map_bgr"] if self.use_chem_bg else data["image_bgr"]
        bgr = compose_image(
            base_img,
            data["masks"],
            data["materials"],
            data["boxes_xyxy"],
            data["confidences"],
            show_yolo=self.show_yolo,
            show_sam2=self.show_sam2,
            show_labels=self.show_labels,
            use_material_colors=self.use_material_colors,
            hover_idx=self.hover_idx,
        )
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._cache_key = key
        self._cached_pil = pil
        return pil

    def invalidate_cache(self):
        self._cache_key = None
        self._cached_pil = None

    # -----------------------------------------------
    # Hover detection
    # -----------------------------------------------
    def _canvas_to_image_coords(self, cx, cy):
        """Convert canvas pixel coords to original image pixel coords."""
        if not self.processed_results:
            return -1, -1
        data = self.processed_results[self.current_idx]
        ih, iw = data["image_bgr"].shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = self.effective_scale(iw, ih, cw, ch)
        if scale <= 0:
            return -1, -1
        img_x = (cx - self.pan_x) / scale
        img_y = (cy - self.pan_y) / scale
        return int(img_x), int(img_y)

    def _hit_test(self, img_x, img_y):
        """Return the index of the detection under (img_x, img_y), or -1."""
        if not self.processed_results:
            return -1
        data = self.processed_results[self.current_idx]
        ih, iw = data["image_bgr"].shape[:2]
        if img_x < 0 or img_y < 0 or img_x >= iw or img_y >= ih:
            return -1

        for i, mask in enumerate(data["masks"]):
            if mask is None:
                continue
            if mask.ndim == 2 and 0 <= img_y < mask.shape[0] and 0 <= img_x < mask.shape[1]:
                if mask[img_y, img_x] > MASK_THRESH:
                    return i

        for i, box in enumerate(data["boxes_xyxy"]):
            x1, y1, x2, y2 = box.astype(int)
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                return i

        return -1

    def on_mouse_move(self, e):
        if self._is_dragging or not self.processed_results:
            return
        if not self.show_yolo or not self.show_sam2 or not self.show_labels:
            img_x, img_y = self._canvas_to_image_coords(e.x, e.y)
            new_hover = self._hit_test(img_x, img_y)
        else:
            new_hover = -1

        if new_hover != self.hover_idx:
            self.hover_idx = new_hover
            self.invalidate_cache()
            self.update_display()

        if new_hover >= 0:
            data = self.processed_results[self.current_idx]
            mat = data["materials"][new_hover] if new_hover < len(data["materials"]) else "?"
            conf = data["confidences"][new_hover] if new_hover < len(data["confidences"]) else 0
            self.lbl_hover.config(text=f"Hovering: #{new_hover + 1} — {mat} ({conf:.0%})")
        else:
            self.lbl_hover.config(text="")

    def on_mouse_leave(self, _):
        if self.hover_idx != -1:
            self.hover_idx = -1
            self.invalidate_cache()
            self.update_display()
            self.lbl_hover.config(text="")

    # -----------------------------------------------
    # Viewport
    # -----------------------------------------------
    def reset_view(self):
        self.user_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.hover_idx = -1

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
        self._is_dragging = True

    def on_drag_move(self, e):
        if self._drag_last_x is None:
            return
        self.pan_x += e.x - self._drag_last_x
        self.pan_y += e.y - self._drag_last_y
        self._drag_last_x, self._drag_last_y = e.x, e.y
        self.update_display()

    def on_drag_end(self, _):
        self._drag_last_x = self._drag_last_y = None
        self._is_dragging = False

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
            self.invalidate_cache()
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
                imgsz=IMG_SIZE, verbose=False, 
                agnostic_nms=False,
            )
            r0 = results[0]
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32) if len(r0.boxes) > 0 else np.empty((0, 4), dtype=np.float32)
            confs = r0.boxes.conf.cpu().numpy() if len(r0.boxes) > 0 else np.empty((0,), dtype=np.float32)

            pred_masks = []
            materials = []

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
                    mat, _ = classify_mask(m0, chem_grid, h, w)
                    materials.append(mat)

            chem_map_bgr = render_chemical_map_image(chem_grid, h, w)

            self.processed_results.append({
                "image_bgr": img_bgr,
                "chem_map_bgr": chem_map_bgr,
                "masks": pred_masks,
                "boxes_xyxy": boxes,
                "confidences": confs,
                "materials": materials,
                "material_counts": Counter(materials),
                "filename": os.path.basename(img_path),
                "csv_file": os.path.basename(csv_path),
            })

            self.progress_val = (idx + 1) / total * 100.0
            print(
                f"[{idx + 1}/{total}] {os.path.basename(img_path)}: "
                f"{len(boxes)} detections -> {dict(Counter(materials))}"
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
        n = len(data["boxes_xyxy"])
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.processed_results)}")

        counts = data["material_counts"]
        parts = [f"{mat}: {cnt}" for mat, cnt in sorted(counts.items())]
        self.lbl_summary.config(text=f"Detected {n} particles | {', '.join(parts) or 'none'}")
        self.lbl_img_info.config(text=f"{data['filename']}  |  CSV: {data['csv_file']}")

    def update_buttons(self):
        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.processed_results) - 1 else tk.DISABLED)

    # -----------------------------------------------
    # Toggles
    # -----------------------------------------------
    def toggle_yolo(self):
        self.show_yolo = not self.show_yolo
        self._update_toggle_btn(self.btn_yolo, "YOLO", self.show_yolo)
        self.invalidate_cache()
        self.update_display()

    def toggle_sam2(self):
        self.show_sam2 = not self.show_sam2
        self._update_toggle_btn(self.btn_sam2, "SAM2", self.show_sam2)
        self.invalidate_cache()
        self.update_display()

    def toggle_labels(self):
        self.show_labels = not self.show_labels
        self._update_toggle_btn(self.btn_labels, "Labels", self.show_labels)
        self.invalidate_cache()
        self.update_display()

    def toggle_colors(self):
        self.use_material_colors = not self.use_material_colors
        self._update_toggle_btn(self.btn_colors, "Colors", self.use_material_colors)
        self.invalidate_cache()
        self.update_display()

    def toggle_background(self):
        self.use_chem_bg = not self.use_chem_bg
        self._update_toggle_btn(self.btn_background, "Background", self.use_chem_bg)
        self.invalidate_cache()
        self.update_display()

    # -----------------------------------------------
    # Navigation
    # -----------------------------------------------
    def next_img(self):
        if self.current_idx < len(self.processed_results) - 1:
            self.current_idx += 1
            self.reset_view()
            self.invalidate_cache()
            self.update_display()
            self.update_buttons()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.reset_view()
            self.invalidate_cache()
            self.update_display()
            self.update_buttons()

    # -----------------------------------------------
    # Export
    # -----------------------------------------------
    def export_all(self):
        if self.is_exporting or not self.processed_results:
            return
        self._show_export_dialog()

    def _show_export_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Options")
        dialog.geometry("320x320")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Select layers to include:",
            font=("Arial", 12, "bold")
        ).pack(pady=(18, 12))

        var_boxes  = tk.BooleanVar(value=self.show_yolo)
        var_masks  = tk.BooleanVar(value=self.show_sam2)
        var_labels = tk.BooleanVar(value=self.show_labels)
        var_colors = tk.BooleanVar(value=self.use_material_colors)
        var_chem   = tk.BooleanVar(value=self.use_chem_bg)

        opts_frame = tk.Frame(dialog)
        opts_frame.pack(anchor="w", padx=40)

        tk.Checkbutton(opts_frame, text="YOLO Boxes",          variable=var_boxes,  font=("Arial", 11)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="SAM2 Masks",          variable=var_masks,  font=("Arial", 11)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="Labels",              variable=var_labels, font=("Arial", 11)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="Material Colors",     variable=var_colors, font=("Arial", 11)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="Chemical Background", variable=var_chem,   font=("Arial", 11)).pack(anchor="w", pady=2)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)

        def on_export():
            dialog.destroy()
            self._run_export(
                var_boxes.get(), var_masks.get(), var_labels.get(),
                var_colors.get(), var_chem.get(),
            )

        def on_cancel():
            dialog.destroy()

        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Export", command=on_export, width=10).pack(side=tk.LEFT, padx=10)

    def make_export_dir(self, layer_parts: list[str]) -> str:
        first_path = os.path.normpath(DATASET_DIR)
        parent_dir = os.path.dirname(first_path)
        base_name = os.path.basename(first_path)

        suffix = "_".join(layer_parts) if layer_parts else "original"
        folder_name = f"{base_name}_{suffix}"

        export_dir = os.path.join(parent_dir, folder_name)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def _run_export(self, show_boxes, show_masks, show_labels, use_material_colors, use_chem_bg):
        parts = []
        if show_boxes:
            parts.append("yolo")
        if show_masks:
            parts.append("sam2")
        if show_labels:
            parts.append("labels")
        if use_material_colors:
            parts.append("colors")
        if use_chem_bg:
            parts.append("chemBG")

        self._export_options = (show_boxes, show_masks, show_labels, use_material_colors, use_chem_bg)
        self._export_dir = self.make_export_dir(parts)
        self.is_exporting = True
        self._export_done = False
        self._export_progress = (0, len(self.processed_results))
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_export.config(text=f"Export: writing to {self._export_dir}")
        self._export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self._export_thread.start()
        self._poll_export()

    def _export_worker(self):
        show_boxes, show_masks, show_labels, use_material_colors, use_chem_bg = self._export_options

        snapshot = list(self.processed_results)
        for i, item in enumerate(snapshot, start=1):
            base_img = item["chem_map_bgr"] if use_chem_bg else item["image_bgr"]
            bgr = compose_image(
                base_img, item["masks"], item["materials"],
                item["boxes_xyxy"], item["confidences"],
                show_yolo=show_boxes,
                show_sam2=show_masks,
                show_labels=show_labels,
                use_material_colors=use_material_colors,
                hover_idx=-1,
            )
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            base = os.path.splitext(item["filename"])[0]
            pil.save(os.path.join(self._export_dir, f"{base}.png"), format="PNG")
            self._export_progress = (i, len(snapshot))

        self._export_done = True

    def _poll_export(self):
        done, total = self._export_progress
        self.lbl_export.config(text=f"Export: {done}/{total} saved -> {self._export_dir}")
        if self._export_done:
            self.is_exporting = False
            self.btn_export.config(state=tk.NORMAL)
            return
        self.root.after(150, self._poll_export)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChemicalIdentificationViewer(root)
    root.mainloop()
