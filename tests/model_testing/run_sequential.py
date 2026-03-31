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
# Comma-separated list of folder paths containing images
SOURCE_PATHS = "./../../datasets/NewImagesForSegmentationTesting"
YOLO_MODEL_PATH = "./../../models/hunter-yolo-v0.4.4.pt"

# SAM2
SAM2_CHECKPOINT = "./../../models/sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# YOLO inference
CONFIDENCE = 0.1
IOU_THRESH = 0.25
IMG_SIZE = 1280

# YOLO drawing
BOX_LINE_WIDTH = 1
LABEL_FONT_SCALE = 0.30
LABEL_FONT_THICKNESS = 1
SHOW_CLASS_NAME = False  # False = show only confidence

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

BOX_SHRINK = 0.10
MASK_THRESH = 0.5
MASK_ALPHA = 0.45

# Export config
EXPORT_ROOT_NAME = "_exports"
EXPORT_PREFIX = "yolo_sam2_overlay"

# Zoom / pan behavior
ZOOM_STEP = 1.12
MIN_USER_ZOOM = 0.20
MAX_USER_ZOOM = 12.0


# ==========================================
# GT segmentation helpers
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
            if len(parts) < 7:  # class + at least 3 points (6 coords)
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
    """Compute IoU between two binary masks."""
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
    """
    Match predicted masks to GT masks and compute accuracy metrics.
    
    Returns:
        mean_iou: Average IoU of matched pairs
        tp: True positives (matched pairs with IoU >= threshold)
        fp: False positives (unmatched predictions)
        fn: False negatives (unmatched GT)
        all_ious: List of best IoU for each GT mask
    """
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
    
    matched_gt = set()
    matched_pred = set()
    all_ious = []
    
    while True:
        if len(matched_gt) == len(gt_masks) or len(matched_pred) == len(pred_masks):
            break
        
        best_iou = -1
        best_gt_idx = -1
        best_pred_idx = -1
        
        for i in range(len(gt_masks)):
            if i in matched_gt:
                continue
            for j in range(len(pred_masks)):
                if j in matched_pred:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
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
    
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    return float(mean_iou), tp, fp, fn, all_ious


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


def draw_yolo_boxes_custom(
    image_bgr: np.ndarray,
    result,
    show_labels: bool = True,
    line_width: int = BOX_LINE_WIDTH,
    font_scale: float = LABEL_FONT_SCALE,
    font_thickness: int = LABEL_FONT_THICKNESS,
) -> np.ndarray:
    out = image_bgr.copy()

    boxes = result.boxes
    names = result.names if hasattr(result, "names") else {}

    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

    for box, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = box.tolist()
        color = (255, 0, 0)  # BGR blue

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        if show_labels:
            cls_name = names.get(cls_id, str(cls_id))
            label = f"{cls_name} {conf:.2f}" if SHOW_CLASS_NAME else f"{conf:.2f}"

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            text_x = x1 + 4
            text_y = y1 - 6
            bg_left = x1
            bg_top = y1 - th - baseline - 8
            bg_right = x1 + tw + 8
            bg_bottom = y1

            if bg_top < 0:
                bg_top = y1
                bg_bottom = y1 + th + baseline + 8
                text_y = y1 + th + 4

            cv2.rectangle(out, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
            cv2.putText(
                out,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return out


def draw_masks_on_top(base_bgr: np.ndarray, masks: list[np.ndarray], alpha: float = MASK_ALPHA) -> np.ndarray:
    out = base_bgr.copy()
    overlay = base_bgr.copy()

    for mask in masks:
        if mask is None:
            continue

        mask_bin = (mask > MASK_THRESH).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue

            # random contour color
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
        self.root.title(f"YOLO + SAM2 Segmentation Accuracy | Device: {DEVICE}")
        self.root.geometry("1200x900")

        self.image_files = self.get_image_list(SOURCE_PATHS)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.progress_val = 0.0
        self.show_overlays = True

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

        if not os.path.exists(YOLO_MODEL_PATH):
            self.lbl_status.config(text=f"❌ MISSING YOLO WEIGHTS: {YOLO_MODEL_PATH}")
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

        self.lbl_status = tk.Label(frame_top, text="Loading Models...", font=("Arial", 12, "bold"))
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

        self.btn_toggle_overlays = tk.Button(
            frame_bot,
            text="Toggle Overlays",
            command=self.toggle_overlays,
            width=14
        )
        self.btn_toggle_overlays.pack(side=tk.RIGHT, padx=10)

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
        self.root.bind("<KeyPress-equal>", self.zoom_in_key)      # handles '+' on many keyboards
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
                for r, _, files in os.walk(dataset_root):
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
        key = "image_with_overlays" if self.show_overlays else "image_original"
        return data[key]

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
        # command / control depending on platform + tkinter bit behavior
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

        # keep the pixel under the cursor fixed during zoom
        img_x = (canvas_x - self.pan_x) / old_scale
        img_y = (canvas_y - self.pan_y) / old_scale

        self.pan_x = canvas_x - img_x * new_scale
        self.pan_y = canvas_y - img_y * new_scale

        self.update_display()

    def on_mousewheel(self, event):
        if not self.processed_results:
            return

        # zoom with Command/Ctrl + scroll, otherwise pan vertically
        if self.is_zoom_modifier_active(event):
            self.zoom_about_point(event.delta > 0, event.x_root - self.canvas.winfo_rootx(), event.y_root - self.canvas.winfo_rooty())
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
            print(f"Loading YOLO: {YOLO_MODEL_PATH}")
            yolo = YOLO(YOLO_MODEL_PATH)

            print(f"Loading Native SAM2: {SAM2_CHECKPOINT}")
            sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
            predictor = SAM2ImagePredictor(sam2_model)

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
            
            # Parse GT segmentation masks
            gt_masks = parse_yolo_seg_labels(lbl_path, w, h)

            base_bgr_with_boxes = draw_yolo_boxes_custom(
                img_bgr, r0,
                show_labels=True,
                line_width=BOX_LINE_WIDTH,
                font_scale=LABEL_FONT_SCALE,
                font_thickness=LABEL_FONT_THICKNESS,
            )

            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32) if len(r0.boxes) > 0 else np.empty((0, 4), dtype=np.float32)
            pred_masks = []

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
                    pred_masks.append(m0)

            # Compute segmentation accuracy (mask IoU)
            mean_iou, tp, fp, fn, _ = match_masks_and_compute_accuracy(
                pred_masks, gt_masks, iou_threshold=IOU_THRESH
            )
            
            # Update totals
            self.total_gt += gt_count
            self.total_pred += pred_count
            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.sum_mean_iou += mean_iou
            self.n_imgs += 1

            final_bgr_with_overlays = draw_masks_on_top(base_bgr_with_boxes, pred_masks, alpha=MASK_ALPHA)

            pil_img_with_overlays = Image.fromarray(cv2.cvtColor(final_bgr_with_overlays, cv2.COLOR_BGR2RGB))
            pil_img_original = Image.fromarray(img_rgb)

            self.processed_results.append({
                "image_with_overlays": pil_img_with_overlays,
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
            })

            self.progress_val = (i + 1) / total * 100.0

        self.is_processing = False

    # -------------------------
    # Export logic
    # -------------------------
    def make_export_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        first_path = SOURCE_PATHS.split(",")[0].strip()
        export_root = os.path.join(first_path, EXPORT_ROOT_NAME)
        os.makedirs(export_root, exist_ok=True)

        export_dir = os.path.join(export_root, f"{EXPORT_PREFIX}_{ts}")
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_all_images(self):
        if self.is_exporting:
            return

        if len(self.processed_results) == 0:
            self.lbl_export.config(text="Export: nothing ready yet.")
            return

        self.export_dir = self.make_export_dir()
        self.is_exporting = True
        self._export_done = False
        self._export_progress = (0, max(1, len(self.processed_results)))
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_export.config(text=f"Export: writing to {self.export_dir}")

        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        self._poll_export_done()

    def _export_worker(self):
        results_snapshot = list(self.processed_results)
        img_key = "image_with_overlays" if self.show_overlays else "image_original"

        for idx, item in enumerate(results_snapshot, start=1):
            img: Image.Image = item[img_key]
            src_path = item.get("src_path", "")
            base = os.path.splitext(os.path.basename(src_path))[0] if src_path else os.path.splitext(item["filename"])[0]

            suffix = "_with_overlays" if self.show_overlays else "_original"
            out_name = f"{base}_yolo_sam2{suffix}.png"
            out_path = os.path.join(self.export_dir, out_name)

            img.save(out_path, format="PNG")
            self._export_progress = (idx, len(results_snapshot))

        self._export_done = True

    def _poll_export_done(self):
        done, total = self._export_progress
        self.lbl_export.config(text=f"Export: {done}/{total} saved → {self.export_dir}")

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

        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"⚙️ Processing: {done}/{len(self.image_files)} images ready...")
            self.root.after(100, self.check_updates)
        else:
            self.lbl_status.config(text=f"Processed {len(self.processed_results)} images.")
            self.update_buttons()
            if len(self.processed_results) > 0 and not self.is_exporting:
                self.btn_export.config(state=tk.NORMAL)

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
            f"YOLO+SAM2 Seg | {data['filename']} | IoU: {iou_str} | Pred/GT: {pred}/{gt} | Overlays: {'ON' if self.show_overlays else 'OFF'} | Zoom: {self.user_zoom:.2f}x"
        )

    def update_buttons(self):
        state_prev = tk.NORMAL if self.current_idx > 0 else tk.DISABLED
        state_next = tk.NORMAL if self.current_idx < len(self.processed_results) - 1 else tk.DISABLED
        self.btn_prev.config(state=state_prev)
        self.btn_next.config(state=state_next)

    def toggle_overlays(self):
        self.show_overlays = not self.show_overlays
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
    app = NativeSAM2YOLOViewer(root)
    root.mainloop()
    