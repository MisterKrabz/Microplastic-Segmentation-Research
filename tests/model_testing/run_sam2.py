import os
import gc
import cv2
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from PIL import Image, ImageTk
import openpyxl

# --- NATIVE SAM2 IMPORTS ---
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ==========================================
# CONFIGURATION
# ==========================================
# Comma-separated list of folder paths containing images
SOURCE_PATHS = "./../../datasets/testing_datasets/Quantification"
COUNTING_XLSX = "./../../datasets/testing_datasets/Quantification/Counting.xlsx"

# Header text to locate columns inside Counting.xlsx. The header row is found
# dynamically (e.g. headers may sit in row 2 rather than row 1), and these
# header strings are then used to resolve the right columns.
COUNTING_IMAGE_NAME_HEADER = "Image name"
COUNTING_SAM2_HEADER = "Counted # by SAM2 only"

# SAM2
SAM2_CHECKPOINT = "./../../models/sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM2 automatic mask generator settings. These are the SAM2-only "baseline"
# parameters: SAM2 is run as a class-agnostic segmenter that proposes masks
# from a regular grid of point prompts. The count for each image is then
# `len(masks)` after the area filter below.
SAM2_POINTS_PER_SIDE = 32
SAM2_POINTS_PER_BATCH = 64      # decoder prompts per kernel launch. 128 doubles
                                 # the GPU work per launch vs SAM2's default of
                                 # 64, halving the number of CPU<->GPU sync
                                 # gaps per image. Safe in bf16 on 18 GB M3.
SAM2_PRED_IOU_THRESH = 0.7
SAM2_STABILITY_SCORE_THRESH = 0.85
SAM2_BOX_NMS_THRESH = 0.7
SAM2_MIN_MASK_REGION_AREA = 20  # px, removes tiny noise-only masks
SAM2_CROP_N_LAYERS = 0           # >0 makes it slower but better for tiny objects

# Post-generation area filter (applied after SAM2's own NMS). Any mask whose
# area is outside [POST_MIN_AREA_PX, POST_MAX_AREA_FRAC * image_area] is
# discarded. This removes the inevitable "background" / full-frame masks that
# the automatic generator produces.
POST_MIN_AREA_PX = 20
POST_MAX_AREA_FRAC = 0.01

# YOLO label parsing kept only to surface ground truth so the SAM2 baseline
# can be measured the same way as run_sequential.
IOU_THRESH = 0.5
MASK_THRESH = 0.5
MASK_ALPHA = 0.45

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Export config
EXPORT_ROOT_NAME = "_exports"
EXPORT_PREFIX = "sam2_only_overlay"

# Zoom / pan behavior
ZOOM_STEP = 1.12
MIN_USER_ZOOM = 0.20
MAX_USER_ZOOM = 12.0


# ==========================================
# GT segmentation helpers (same as run_sequential)
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


def parse_yolo_seg_labels(label_path: str, img_w: int, img_h: int) -> list[np.ndarray]:
    """Parse YOLO segmentation label file and return list of binary masks."""
    masks = []
    if not os.path.exists(label_path):
        return masks

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])

            if len(points) >= 3:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)

    return masks


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_masks_and_compute_accuracy(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5
) -> tuple[float, int, int, int, list[float]]:
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0, 0, 0, 0, []

    if len(gt_masks) == 0:
        return 0.0, 0, len(pred_masks), 0, []

    if len(pred_masks) == 0:
        return 0.0, 0, 0, len(gt_masks), [0.0] * len(gt_masks)

    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            pred_binary = (pred_mask > MASK_THRESH).astype(np.uint8) if pred_mask.dtype != np.uint8 else pred_mask
            iou_matrix[i, j] = compute_mask_iou(gt_mask, pred_binary)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    all_ious: list[float] = []

    while True:
        if len(matched_gt) == len(gt_masks) or len(matched_pred) == len(pred_masks):
            break

        best_iou = -1.0
        best_gt_idx = -1
        best_pred_idx = -1

        for i in range(len(gt_masks)):
            if i in matched_gt:
                continue
            for j in range(len(pred_masks)):
                if j in matched_pred:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = float(iou_matrix[i, j])
                    best_gt_idx = i
                    best_pred_idx = j

        if best_iou < 0:
            break

        matched_gt.add(best_gt_idx)
        matched_pred.add(best_pred_idx)
        all_ious.append(best_iou)

    for i in range(len(gt_masks)):
        if i not in matched_gt:
            all_ious.append(0.0)

    tp = sum(1 for iou in all_ious if iou >= iou_threshold)
    fp = len(pred_masks) - len(matched_pred)
    fn = len(gt_masks) - tp

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return mean_iou, tp, fp, fn, all_ious


def count_gt_instances(label_path: str) -> int:
    if not os.path.exists(label_path):
        return 0
    n = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ==========================================
# Drawing helpers
# ==========================================
RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)


def masks_to_contours(masks: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Convert a list of binary H x W masks into a list of contour polygons.

    We store contours instead of full-resolution masks so the total memory
    cost of `processed_results` stays in MB rather than GB across the full
    dataset. Visually identical: cv2.drawContours is what the renderer used
    on the masks anyway.
    """
    out: list[list[np.ndarray]] = []
    for m in masks:
        if m is None:
            continue
        m_bin = (m > MASK_THRESH).astype(np.uint8) if m.dtype != np.uint8 else m
        if m_bin.sum() == 0:
            continue
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            out.append(list(contours))
    return out


def compose_layers(
    pil_original: Image.Image,
    mask_contours: list[list[np.ndarray]],
    show_masks: bool = True,
    use_red: bool = False,
) -> Image.Image:
    """Render the SAM2-only overlay: filled translucent fill + crisp outline,
    drawn from the cached contours. No labels, no YOLO boxes.
    """
    bgr = cv2.cvtColor(np.array(pil_original), cv2.COLOR_RGB2BGR)

    if show_masks and mask_contours:
        color = RED_BGR if use_red else BLUE_BGR
        overlay = bgr.copy()
        for cnts in mask_contours:
            cv2.drawContours(overlay, cnts, -1, color, -1)
            cv2.drawContours(bgr, cnts, -1, color, 2)
        bgr = cv2.addWeighted(overlay, MASK_ALPHA, bgr, 1 - MASK_ALPHA, 0)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ==========================================
# MAIN VIEWER
# ==========================================
class NativeSAM2OnlyViewer:
    def __init__(self, root):
        self.root = root
        self.root.title(f"SAM2-only Baseline | Device: {DEVICE}")
        self.root.geometry("1200x900")

        self.image_files = self.get_image_list(SOURCE_PATHS)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.progress_val = 0.0
        self.show_sam2 = True
        self.use_red = False

        # Segmentation accuracy metrics
        self.total_gt = 0
        self.total_pred = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.sum_mean_iou = 0.0
        self.n_imgs = 0

        self.export_dir = None
        self.export_thread = None
        self.is_exporting = False
        self._export_done = False
        self._export_progress = (0, 1)

        # image viewport state
        self.tk_img = None
        self.canvas_img_id = None
        self.user_zoom = 1.0       # zoom relative to fit-to-window size
        self.fit_scale = 1.0       # auto-computed to fit the canvas
        self.pan_x = 0.0           # in canvas coordinates
        self.pan_y = 0.0
        self._drag_last_x = None
        self._drag_last_y = None

        self.setup_gui()
        self.bind_navigation_events()

        if not self.image_files:
            self.lbl_status.config(text=f"❌ Error: No images found at {SOURCE_PATHS}")
            self.is_processing = False
            return

        if not os.path.exists(SAM2_CHECKPOINT):
            self.lbl_status.config(text=f"❌ MISSING SAM2 WEIGHTS: {SAM2_CHECKPOINT}")
            self.is_processing = False
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

        self.lbl_status = tk.Label(frame_top, text="Loading SAM2...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()

        self.lbl_accuracy = tk.Label(frame_top, text="Accuracy: --", font=("Arial", 11))
        self.lbl_accuracy.pack()

        self.progress = ttk.Progressbar(frame_top, orient=tk.HORIZONTAL, length=400, mode="determinate")
        self.progress.pack(pady=5)

        self.lbl_export = tk.Label(frame_top, text="Export: --", font=("Arial", 10))
        self.lbl_export.pack()

        self.canvas = tk.Canvas(self.root, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_bot = tk.Frame(self.root, pady=15, bg="#f0f0f0")
        frame_bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(frame_bot, text="<< Prev", command=self.prev_img, state=tk.DISABLED, width=12)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.lbl_counter = tk.Label(frame_bot, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=10)

        self.lbl_img_metrics = tk.Label(
            frame_bot,
            text="Img: Acc -- | Pred/GT --/--",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        self.lbl_img_metrics.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(frame_bot, text="Next >>", command=self.next_img, state=tk.DISABLED, width=12)
        self.btn_next.pack(side=tk.RIGHT, padx=10)

        self.btn_export = tk.Button(
            frame_bot,
            text="Export All",
            command=self.export_all_images,
            state=tk.DISABLED,
            width=16
        )
        self.btn_export.pack(side=tk.RIGHT, padx=10)

        self.btn_export_counts = tk.Button(
            frame_bot,
            text="Export Counts",
            command=self.export_counts,
            state=tk.DISABLED,
            width=14
        )
        self.btn_export_counts.pack(side=tk.RIGHT, padx=10)

        self.btn_toggle_color = tk.Button(
            frame_bot,
            text="Color: Default",
            command=self.toggle_color,
            width=14
        )
        self.btn_toggle_color.pack(side=tk.RIGHT, padx=10)

        self.btn_toggle_sam2 = tk.Button(
            frame_bot, text="Toggle SAM2",
            command=self.toggle_sam2, width=14,
        )
        self.btn_toggle_sam2.pack(side=tk.RIGHT, padx=4)

    def bind_navigation_events(self):
        self.root.bind("<Configure>", self.on_window_resize)

        # drag to pan
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        # mac/windows wheel
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)

        # linux fallback
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-4>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self.on_mousewheel_linux)

        # reliable keyboard zoom fallback
        self.root.bind("<KeyPress-plus>", self.zoom_in_key)
        self.root.bind("<KeyPress-equal>", self.zoom_in_key)
        self.root.bind("<KeyPress-minus>", self.zoom_out_key)
        self.root.bind("<KeyPress-underscore>", self.zoom_out_key)
        self.root.bind("<KeyPress-0>", self.reset_zoom_key)

        # mac command shortcuts
        self.root.bind("<Command-plus>", self.zoom_in_key)
        self.root.bind("<Command-equal>", self.zoom_in_key)
        self.root.bind("<Command-minus>", self.zoom_out_key)
        self.root.bind("<Command-0>", self.reset_zoom_key)

        self.canvas.focus_set()

    # -------------------------
    # Dataset scanning
    # -------------------------
    def get_image_list(self, source_paths_str):
        image_paths = []

        paths = [p.strip() for p in source_paths_str.split(",") if p.strip()]

        for dataset_root in paths:
            if os.path.isfile(dataset_root):
                if dataset_root.lower().endswith(VALID_EXTS):
                    image_paths.append(dataset_root)
                continue

            if os.path.isdir(dataset_root):
                for r, dirs, files in os.walk(dataset_root):
                    dirs[:] = [d for d in dirs if d != EXPORT_ROOT_NAME]
                    for f in files:
                        if f.lower().endswith(VALID_EXTS):
                            image_paths.append(os.path.join(r, f))

        return sorted(image_paths)

    # -------------------------
    # Viewport helpers
    # -------------------------
    def get_current_pil_image(self):
        if not self.processed_results:
            return None
        data = self.processed_results[self.current_idx]
        if not self.show_sam2:
            return data["image_original"]
        return compose_layers(
            data["image_original"],
            data["mask_contours"],
            show_masks=self.show_sam2,
            use_red=self.use_red,
        )

    def reset_view(self):
        self.user_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def get_effective_scale(self, img_w, img_h, canvas_w, canvas_h):
        if img_w <= 0 or img_h <= 0 or canvas_w <= 1 or canvas_h <= 1:
            return 1.0
        fit = min(canvas_w / img_w, canvas_h / img_h)
        self.fit_scale = fit
        return fit * self.user_zoom

    def clamp_pan(self, scaled_w, scaled_h, canvas_w, canvas_h):
        if scaled_w <= canvas_w:
            self.pan_x = (canvas_w - scaled_w) / 2.0
        else:
            min_x = canvas_w - scaled_w
            max_x = 0
            self.pan_x = min(max(self.pan_x, min_x), max_x)

        if scaled_h <= canvas_h:
            self.pan_y = (canvas_h - scaled_h) / 2.0
        else:
            min_y = canvas_h - scaled_h
            max_y = 0
            self.pan_y = min(max(self.pan_y, min_y), max_y)

    # -------------------------
    # Events
    # -------------------------
    def on_window_resize(self, event):
        if event.widget == self.root and self.processed_results:
            self.update_display()

    def on_drag_start(self, event):
        self._drag_last_x = event.x
        self._drag_last_y = event.y

    def on_drag_move(self, event):
        if self._drag_last_x is None or self._drag_last_y is None:
            return

        dx = event.x - self._drag_last_x
        dy = event.y - self._drag_last_y
        self._drag_last_x = event.x
        self._drag_last_y = event.y

        self.pan_x += dx
        self.pan_y += dy
        self.update_display()

    def on_drag_end(self, event):
        self._drag_last_x = None
        self._drag_last_y = None

    def is_zoom_modifier_active(self, event) -> bool:
        state = getattr(event, "state", 0)
        return bool(state & 0x0004) or bool(state & 0x0008) or bool(state & 0x0010)

    def zoom_about_point(self, zoom_in: bool, canvas_x: float, canvas_y: float):
        pil_img = self.get_current_pil_image()
        if pil_img is None:
            return

        img_w, img_h = pil_img.size
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        old_scale = self.get_effective_scale(img_w, img_h, canvas_w, canvas_h)
        old_user_zoom = self.user_zoom

        if zoom_in:
            self.user_zoom = min(MAX_USER_ZOOM, self.user_zoom * ZOOM_STEP)
        else:
            self.user_zoom = max(MIN_USER_ZOOM, self.user_zoom / ZOOM_STEP)

        if abs(self.user_zoom - old_user_zoom) < 1e-12:
            return

        new_scale = self.get_effective_scale(img_w, img_h, canvas_w, canvas_h)

        img_x = (canvas_x - self.pan_x) / old_scale
        img_y = (canvas_y - self.pan_y) / old_scale

        self.pan_x = canvas_x - img_x * new_scale
        self.pan_y = canvas_y - img_y * new_scale

        self.update_display()

    def on_mousewheel(self, event):
        if not self.processed_results:
            return

        if self.is_zoom_modifier_active(event):
            self.zoom_about_point(
                event.delta > 0,
                event.x_root - self.canvas.winfo_rootx(),
                event.y_root - self.canvas.winfo_rooty(),
            )
        else:
            step = 45 if event.delta > 0 else -45
            self.pan_y += step
            self.update_display()

    def on_mousewheel_linux(self, event):
        if not self.processed_results:
            return

        canvas_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        canvas_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

        if self.is_zoom_modifier_active(event):
            if event.num == 4:
                self.zoom_about_point(True, canvas_x, canvas_y)
            elif event.num == 5:
                self.zoom_about_point(False, canvas_x, canvas_y)
        else:
            if event.num == 4:
                self.pan_y += 45
            elif event.num == 5:
                self.pan_y -= 45
            self.update_display()

    def zoom_in_key(self, event=None):
        if not self.processed_results:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self.zoom_about_point(True, canvas_w / 2, canvas_h / 2)

    def zoom_out_key(self, event=None):
        if not self.processed_results:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self.zoom_about_point(False, canvas_w / 2, canvas_h / 2)

    def reset_zoom_key(self, event=None):
        if not self.processed_results:
            return
        self.user_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_display()

    # -------------------------
    # Pipeline
    # -------------------------
    def run_pipeline(self):
        try:
            print(f"Loading Native SAM2: {SAM2_CHECKPOINT}")
            sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=SAM2_POINTS_PER_SIDE,
                points_per_batch=SAM2_POINTS_PER_BATCH,
                pred_iou_thresh=SAM2_PRED_IOU_THRESH,
                stability_score_thresh=SAM2_STABILITY_SCORE_THRESH,
                box_nms_thresh=SAM2_BOX_NMS_THRESH,
                crop_n_layers=SAM2_CROP_N_LAYERS,
                min_mask_region_area=SAM2_MIN_MASK_REGION_AREA,
            )
        except Exception as e:
            print(f"MODEL LOAD ERROR: {e}")
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
            img_area = h * w
            max_area = POST_MAX_AREA_FRAC * img_area

            lbl_path = label_for_image(img_path)
            gt_count = count_gt_instances(lbl_path)
            gt_masks = parse_yolo_seg_labels(lbl_path, w, h)

            try:
                # inference_mode disables autograd bookkeeping (no grad
                # tensors allocated on the MPS pool). Autocast runs the
                # encoder + decoder in bfloat16 on Apple Silicon / CUDA,
                # which roughly halves activation memory and is ~30-50%
                # faster than fp32 with no measurable quality drop on
                # SAM2's mask outputs (this is what Meta's own SAM2
                # example notebook uses).
                amp_enabled = DEVICE in ("cuda", "mps")
                with torch.inference_mode(), torch.autocast(
                    device_type=DEVICE if amp_enabled else "cpu",
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    raw = mask_generator.generate(img_rgb)
            except Exception as e:
                print(f"  SAM2 generate error on {os.path.basename(img_path)}: {e}")
                raw = []

            kept_masks: list[np.ndarray] = []
            for ann in raw:
                m = ann.get("segmentation")
                if m is None:
                    continue
                m_bin = m.astype(np.uint8) if m.dtype != np.uint8 else m
                area = int(m_bin.sum())
                if area < POST_MIN_AREA_PX:
                    continue
                if area > max_area:
                    continue
                kept_masks.append(m_bin)

            # Free the raw SAM2 output as soon as we've extracted what we need.
            # Each entry holds a full-resolution boolean mask + scores; on a
            # busy image these add up to hundreds of MB.
            raw = None

            pred_count = len(kept_masks)

            mean_iou, tp, fp, fn, _ = match_masks_and_compute_accuracy(
                kept_masks, gt_masks, iou_threshold=IOU_THRESH
            )

            # Convert to contours immediately so we don't keep N x H x W
            # uint8 arrays per image alive in `processed_results`. Contours
            # take roughly 100-1000x less memory and render identically.
            mask_contours = masks_to_contours(kept_masks)
            kept_masks = None
            gt_masks = None

            self.total_gt += gt_count
            self.total_pred += pred_count
            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.sum_mean_iou += mean_iou
            self.n_imgs += 1

            pil_img_original = Image.fromarray(img_rgb)

            self.processed_results.append({
                "image_original": pil_img_original,
                "filename": os.path.basename(img_path),
                "pred_count": pred_count,
                "gt_count": gt_count,
                "mean_iou": mean_iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "label_path": lbl_path,
                "src_path": img_path,
                "mask_contours": mask_contours,
            })

            # Hand memory back to the OS / MPS pool between images so the
            # encoder activations from image i don't sit around while
            # image i+1 is being prepared.
            del img_bgr, img_rgb
            gc.collect()
            if DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif DEVICE == "cuda":
                torch.cuda.empty_cache()

            self.progress_val = (i + 1) / total * 100.0
            print(
                f"[{i + 1}/{total}] {os.path.basename(img_path)}: "
                f"SAM2 kept {pred_count} masks (GT: {gt_count})"
            )

        self.is_processing = False

    # -------------------------
    # Export counts to Counting.xlsx
    # -------------------------
    def export_counts(self):
        if not self.processed_results:
            self.lbl_export.config(text="Export Counts: nothing to export yet.")
            return
        self.btn_export_counts.config(state=tk.DISABLED)
        self.write_to_xlsx()
        self.btn_export_counts.config(state=tk.NORMAL)

    def write_to_xlsx(self):
        if not self.processed_results:
            return

        xlsx_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), COUNTING_XLSX
        )
        xlsx_path = os.path.normpath(xlsx_path)

        is_new_file = not os.path.exists(xlsx_path)
        if is_new_file:
            wb = openpyxl.Workbook()
            ws = wb.active
            # Match the layout of the existing Counting.xlsx: header row at
            # row 2, image names in column B, "Counted # by SAM2 only" in H.
            ws["B2"] = COUNTING_IMAGE_NAME_HEADER
            ws["C2"] = "PS size (um)"
            ws["D2"] = "Conc. (mgL)"
            ws["E2"] = "Counted # by algorithm"
            ws["F2"] = "Counted # by human"
            ws["G2"] = "Counted # by v3"
            ws["H2"] = COUNTING_SAM2_HEADER
            header_row = 2
            name_col = 2  # B
            sam2_col = 8  # H
        else:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb.active
            header_row = self._find_header_row(ws, COUNTING_IMAGE_NAME_HEADER)
            if header_row < 0:
                msg = (
                    f"Export Counts: could not find '{COUNTING_IMAGE_NAME_HEADER}' "
                    f"header in {os.path.basename(xlsx_path)}"
                )
                print(msg)
                self.lbl_export.config(text=msg)
                return

            name_col = self._find_column_for_header(
                ws, header_row, COUNTING_IMAGE_NAME_HEADER
            )
            sam2_col = self._find_column_for_header(
                ws, header_row, COUNTING_SAM2_HEADER
            )
            if name_col < 0:
                msg = f"Export Counts: '{COUNTING_IMAGE_NAME_HEADER}' column not found in row {header_row}."
                print(msg)
                self.lbl_export.config(text=msg)
                return

            if sam2_col < 0:
                # Header for the SAM2-only column doesn't exist yet -- create
                # it in column H (preferred) or, if H is occupied by another
                # header, in the first empty header cell after the image-name
                # column.
                preferred = 8  # H
                target = preferred
                existing = ws.cell(row=header_row, column=preferred).value
                if existing is not None and str(existing).strip() != "":
                    target = ws.max_column + 1
                ws.cell(row=header_row, column=target, value=COUNTING_SAM2_HEADER)
                sam2_col = target

        # Build a strict basename -> row map.
        name_to_row: dict[str, int] = {}
        last_data_row = header_row
        for r in range(header_row + 1, ws.max_row + 1):
            v = ws.cell(row=r, column=name_col).value
            if v is None:
                continue
            key = str(v).strip()
            if not key:
                continue
            if key not in name_to_row:
                name_to_row[key] = r
            last_data_row = max(last_data_row, r)

        next_row = last_data_row + 1

        written = 0
        appended = 0
        for data in self.processed_results:
            image_name = os.path.splitext(data["filename"])[0].strip()
            target_row = name_to_row.get(image_name)
            if target_row is None:
                ws.cell(row=next_row, column=name_col, value=image_name)
                ws.cell(row=next_row, column=sam2_col, value=data["pred_count"])
                name_to_row[image_name] = next_row
                next_row += 1
                appended += 1
            else:
                ws.cell(row=target_row, column=sam2_col, value=data["pred_count"])
                written += 1

        wb.save(xlsx_path)

        col_letter = openpyxl.utils.get_column_letter(sam2_col)
        msg = (
            f"Exported {written} count(s) to '{COUNTING_SAM2_HEADER}' "
            f"(col {col_letter})"
        )
        if appended:
            msg += f", appended {appended} new row(s)"
        print(msg)
        self.lbl_export.config(text=msg)

    @staticmethod
    def _find_header_row(ws, image_name_header: str) -> int:
        for row in ws.iter_rows(
            min_row=1, max_row=min(ws.max_row, 20), values_only=False
        ):
            for c in row:
                if c.value is not None and str(c.value).strip() == image_name_header:
                    return c.row
        return -1

    @staticmethod
    def _find_column_for_header(ws, header_row: int, header_text: str) -> int:
        for c in ws[header_row]:
            if c.value is not None and str(c.value).strip() == header_text:
                return c.column
        return -1

    # -------------------------
    # Export logic
    # -------------------------
    def make_export_dir(self, layer_parts: list[str]) -> str:
        first_path = os.path.normpath(SOURCE_PATHS.split(",")[0].strip())
        parent_dir = os.path.dirname(first_path)
        base_name = os.path.basename(first_path)

        suffix = "_".join(layer_parts) if layer_parts else "original"
        folder_name = f"{base_name}_sam2only_{suffix}"

        export_dir = os.path.join(parent_dir, folder_name)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_all_images(self):
        if self.is_exporting:
            return

        if len(self.processed_results) == 0:
            self.lbl_export.config(text="Export: nothing ready yet.")
            return

        self._show_export_dialog()

    def _show_export_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Options")
        dialog.geometry("320x260")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Select layers to include:",
            font=("Arial", 12, "bold")
        ).pack(pady=(18, 12))

        var_masks = tk.BooleanVar(value=True)
        var_red = tk.BooleanVar(value=self.use_red)

        opts_frame = tk.Frame(dialog)
        opts_frame.pack(anchor="w", padx=40)

        tk.Checkbutton(opts_frame, text="SAM2 Masks", variable=var_masks,
                        font=("Arial", 11)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="Red Color", variable=var_red,
                        font=("Arial", 11)).pack(anchor="w", pady=2)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)

        def on_export():
            dialog.destroy()
            self._run_export(var_masks.get(), var_red.get())

        def on_cancel():
            dialog.destroy()

        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Export", command=on_export, width=10).pack(side=tk.LEFT, padx=10)

    def _run_export(self, show_masks, use_red):
        parts = []
        if show_masks:
            parts.append("sam2")
        if use_red:
            parts.append("red")

        self.export_dir = self.make_export_dir(parts)
        self.is_exporting = True
        self._export_done = False
        self._export_progress = (0, max(1, len(self.processed_results)))
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_export.config(text=f"Export: writing to {self.export_dir}")

        self._export_options = (show_masks, use_red)
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        self._poll_export_done()

    def _export_worker(self):
        show_masks, use_red = self._export_options
        results_snapshot = list(self.processed_results)

        for idx, item in enumerate(results_snapshot, start=1):
            img = compose_layers(
                item["image_original"],
                item["mask_contours"],
                show_masks=show_masks,
                use_red=use_red,
            )

            src_path = item.get("src_path", "")
            base = os.path.splitext(os.path.basename(src_path))[0] if src_path else os.path.splitext(item["filename"])[0]

            out_name = f"{base}.png"
            out_path = os.path.join(self.export_dir, out_name)

            img.save(out_path, format="PNG")
            self._export_progress = (idx, len(results_snapshot))

        self._export_done = True

    def _poll_export_done(self):
        done, total = self._export_progress
        self.lbl_export.config(text=f"Export: {done}/{total} saved -> {self.export_dir}")

        if self._export_done:
            self.is_exporting = False
            self.btn_export.config(state=tk.NORMAL)
            return

        self.root.after(150, self._poll_export_done)

    # -------------------------
    # UI update loop
    # -------------------------
    def check_updates(self):
        self.progress["value"] = self.progress_val

        if self.processed_results:
            self.update_display()

        self.update_buttons()

        if self.n_imgs > 0:
            mean_iou = (self.sum_mean_iou / self.n_imgs) * 100.0
            precision = self.total_tp / max(1, self.total_tp + self.total_fp) * 100.0
            recall = self.total_tp / max(1, self.total_tp + self.total_fn) * 100.0
            self.lbl_accuracy.config(
                text=f"Mean IoU: {mean_iou:.1f}% | Precision: {precision:.1f}% | Recall: {recall:.1f}% | TP/FP/FN: {self.total_tp}/{self.total_fp}/{self.total_fn}"
            )
        else:
            self.lbl_accuracy.config(text="Segmentation Accuracy: --")

        if len(self.processed_results) > 0 and not self.is_exporting:
            self.btn_export.config(state=tk.NORMAL)
            self.btn_export_counts.config(state=tk.NORMAL)

        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"⚙️ Processing: {done}/{len(self.image_files)} images ready...")
            self.root.after(100, self.check_updates)
        else:
            self.lbl_status.config(text=f"Processed {len(self.processed_results)} images.")
            self.update_buttons()
            if len(self.processed_results) > 0 and not self.is_exporting:
                self.btn_export.config(state=tk.NORMAL)
                self.btn_export_counts.config(state=tk.NORMAL)

    def update_display(self):
        pil_img = self.get_current_pil_image()
        if pil_img is None:
            return

        img_w, img_h = pil_img.size
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        effective_scale = self.get_effective_scale(img_w, img_h, canvas_w, canvas_h)
        scaled_w = max(1, int(img_w * effective_scale))
        scaled_h = max(1, int(img_h * effective_scale))

        self.clamp_pan(scaled_w, scaled_h, canvas_w, canvas_h)

        img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)

        if self.canvas_img_id is None:
            self.canvas.delete("all")
            self.canvas_img_id = self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_img)
            self.canvas.coords(self.canvas_img_id, self.pan_x, self.pan_y)

        self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))

        data = self.processed_results[self.current_idx]
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.processed_results)}")

        pred = data.get("pred_count", None)
        gt = data.get("gt_count", None)
        mean_iou = data.get("mean_iou", None)
        tp = data.get("tp", 0)
        fp = data.get("fp", 0)
        fn = data.get("fn", 0)

        if pred is None or gt is None or mean_iou is None:
            self.lbl_img_metrics.config(text="Img: IoU -- | Pred/GT --/--")
        else:
            self.lbl_img_metrics.config(text=f"Img: IoU {mean_iou * 100:.1f}% | TP/FP/FN {tp}/{fp}/{fn} | Pred/GT {pred}/{gt}")

        iou_str = f"{mean_iou * 100:.1f}%" if mean_iou is not None else "--"
        self.root.title(
            f"SAM2-only Seg | {data['filename']} | IoU: {iou_str} | Pred/GT: {pred}/{gt} "
            f"| SAM2: {'ON' if self.show_sam2 else 'OFF'} | Zoom: {self.user_zoom:.2f}x"
        )

    def update_buttons(self):
        state_prev = tk.NORMAL if self.current_idx > 0 else tk.DISABLED
        state_next = tk.NORMAL if self.current_idx < len(self.processed_results) - 1 else tk.DISABLED
        self.btn_prev.config(state=state_prev)
        self.btn_next.config(state=state_next)

    def toggle_sam2(self):
        self.show_sam2 = not self.show_sam2
        self.update_display()

    def toggle_color(self):
        self.use_red = not self.use_red
        self.btn_toggle_color.config(text="Color: Red" if self.use_red else "Color: Default")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = NativeSAM2OnlyViewer(root)
    root.mainloop()
