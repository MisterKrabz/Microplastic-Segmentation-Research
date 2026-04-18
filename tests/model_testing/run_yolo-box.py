import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_PATHS = "./../../datasets/Quantification"
YOLO_MODEL_PATH = "./../../models/hunter-yolo-v0.4.4.pt"

# YOLO inference
CONFIDENCE = 0.1
IOU_THRESH = 0.25
IMG_SIZE = 1280

# YOLO drawing
BOX_LINE_WIDTH = 1
LABEL_FONT_SCALE = 0.30
LABEL_FONT_THICKNESS = 1
SHOW_CLASS_NAME = False

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Export config
EXPORT_ROOT_NAME = "_exports"
EXPORT_PREFIX = "yolo_box_overlay"

# Zoom / pan behavior
ZOOM_STEP = 1.12
MIN_USER_ZOOM = 0.20
MAX_USER_ZOOM = 12.0


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
# Drawing helpers
# ==========================================
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
        color = (255, 0, 0)

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


# ==========================================
# MAIN VIEWER
# ==========================================
class YOLOBoxViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Box Detection Viewer")
        self.root.geometry("1200x900")

        self.image_files = self.get_image_list(SOURCE_PATHS)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.progress_val = 0.0
        self.show_overlays = True

        self.total_gt = 0
        self.total_pred = 0
        self.sum_img_acc = 0.0
        self.n_imgs = 0

        self.export_dir = None
        self.export_thread = None
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
        self.bind_navigation_events()

        if not self.image_files:
            self.lbl_status.config(text=f"No images found at {SOURCE_PATHS}")
            self.is_processing = False
            return

        if not os.path.exists(YOLO_MODEL_PATH):
            self.lbl_status.config(text=f"MISSING YOLO WEIGHTS: {YOLO_MODEL_PATH}")
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

        self.lbl_status = tk.Label(frame_top, text="Loading Model...", font=("Arial", 12, "bold"))
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

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)

        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-4>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self.on_mousewheel_linux)

        self.root.bind("<KeyPress-plus>", self.zoom_in_key)
        self.root.bind("<KeyPress-equal>", self.zoom_in_key)
        self.root.bind("<KeyPress-minus>", self.zoom_out_key)
        self.root.bind("<KeyPress-underscore>", self.zoom_out_key)
        self.root.bind("<KeyPress-0>", self.reset_zoom_key)

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
            self.pan_x = min(max(self.pan_x, min_x), 0)

        if scaled_h <= canvas_h:
            self.pan_y = (canvas_h - scaled_h) / 2.0
        else:
            min_y = canvas_h - scaled_h
            self.pan_y = min(max(self.pan_y, min_y), 0)

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
    # Pipeline (YOLO only)
    # -------------------------
    def run_pipeline(self):
        try:
            print(f"Loading YOLO: {YOLO_MODEL_PATH}")
            yolo = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            print(f"MODEL LOAD ERROR: {e}")
            self.is_processing = False
            return

        total = len(self.image_files)

        for i, img_path in enumerate(self.image_files):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

            boxes_bgr = draw_yolo_boxes_custom(
                img_bgr, r0,
                show_labels=True,
                line_width=BOX_LINE_WIDTH,
                font_scale=LABEL_FONT_SCALE,
                font_thickness=LABEL_FONT_THICKNESS,
            )

            pil_with_overlays = Image.fromarray(cv2.cvtColor(boxes_bgr, cv2.COLOR_BGR2RGB))
            pil_original = Image.fromarray(img_rgb)

            self.processed_results.append({
                "image_with_overlays": pil_with_overlays,
                "image_original": pil_original,
                "filename": os.path.basename(img_path),
                "pred_count": pred_count,
                "gt_count": gt_count,
                "img_acc": img_acc,
                "label_path": lbl_path,
                "src_path": img_path,
            })

            self.progress_val = (i + 1) / total * 100.0
            print(
                f"[{i + 1}/{total}] {os.path.basename(img_path)}: "
                f"YOLO detected {pred_count} microplastics (GT: {gt_count})"
            )

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
            out_name = f"{base}_yolo_box{suffix}.png"
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
            mean_acc = (self.sum_img_acc / self.n_imgs) * 100.0
            self.lbl_accuracy.config(
                text=f"Accuracy (mean per-image): {mean_acc:.1f}% | Total Pred/GT: {self.total_pred}/{self.total_gt}"
            )
        else:
            self.lbl_accuracy.config(text="Accuracy: --")

        if len(self.processed_results) > 0 and not self.is_exporting:
            self.btn_export.config(state=tk.NORMAL)

        if self.is_processing:
            done = len(self.processed_results)
            self.lbl_status.config(text=f"Processing: {done}/{len(self.image_files)} images ready...")
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
        img_acc = data.get("img_acc", None)

        if pred is None or gt is None or img_acc is None:
            self.lbl_img_metrics.config(text="Img: Acc -- | Pred/GT --/--")
        else:
            self.lbl_img_metrics.config(text=f"Img: Acc {img_acc * 100:.1f}% | Pred/GT {pred}/{gt}")

        acc_str = f"{img_acc * 100:.1f}%" if img_acc is not None else "--"
        self.root.title(
            f"YOLO Box | {data['filename']} | Acc: {acc_str} | Pred/GT: {pred}/{gt} | Overlays: {'ON' if self.show_overlays else 'OFF'} | Zoom: {self.user_zoom:.2f}x"
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
    app = YOLOBoxViewer(root)
    root.mainloop()
