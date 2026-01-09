import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_PATH = "./../datasets/raw_data/LMMP"
MODEL_PATH = "../models/hunter-yolo-v0.3.0.pt"

CONFIDENCE = 0.25
IOU_THRESH = 0.5
IMG_SIZE   = 1280
# ==========================================

def overlay_masks_on_bgr(img_bgr, result, alpha=0.35, draw_contours=True):
    """
    Robust segmentation visualization for Ultralytics:
    - Primary: uses result.masks.data (bitmask) when available
    - Fallback: uses result.masks.xy (polygons) when data is not available
    """
    out = img_bgr.copy()

    masks_obj = getattr(result, "masks", None)
    if masks_obj is None:
        return out

    rng = np.random.default_rng(0)
    h, w = out.shape[:2]

    # -------------------------
    # Case A: Bitmask available
    # -------------------------
    if getattr(masks_obj, "data", None) is not None and masks_obj.data is not None:
        masks = masks_obj.data.detach().cpu().numpy()  # (N,H,W)

        for i in range(masks.shape[0]):
            mask = masks[i]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            mask_bin = (mask > 0.5).astype(np.uint8)
            if mask_bin.sum() == 0:
                continue

            color = rng.integers(0, 255, size=3, dtype=np.uint8)  # BGR
            colored = np.zeros_like(out, dtype=np.uint8)
            colored[:, :] = color

            idx = mask_bin.astype(bool)
            out[idx] = (out[idx] * (1 - alpha) + colored[idx] * alpha).astype(np.uint8)

            if draw_contours:
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color=tuple(int(c) for c in color), thickness=2)

        return out

    # -------------------------
    # Case B: Polygon fallback
    # -------------------------
    # Ultralytics often provides masks as polygons: masks.xy is a list of Nx2 float arrays (pixel coords)
    if hasattr(masks_obj, "xy") and masks_obj.xy is not None:
        polys = masks_obj.xy  # list of (Ni,2) arrays in pixel coords

        for i, poly in enumerate(polys):
            if poly is None or len(poly) < 3:
                continue

            color = rng.integers(0, 255, size=3, dtype=np.uint8)  # BGR
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

            # Fill with transparency by drawing on an overlay
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color=tuple(int(c) for c in color))
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

            if draw_contours:
                cv2.polylines(out, [pts], isClosed=True, color=tuple(int(c) for c in color), thickness=2)

        return out

    return out


def count_instances(result):
    """Robust instance counting for seg + detect outputs."""
    masks_obj = getattr(result, "masks", None)
    if masks_obj is not None:
        # Prefer polygon count if present
        if hasattr(masks_obj, "xy") and masks_obj.xy is not None:
            try:
                return len(masks_obj.xy)
            except Exception:
                pass
        # Else try bitmask count
        if getattr(masks_obj, "data", None) is not None and masks_obj.data is not None:
            try:
                return int(masks_obj.data.shape[0])
            except Exception:
                pass

    # Fallback to boxes
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is not None:
        try:
            return len(boxes_obj)
        except Exception:
            pass

    return 0


class MicroplasticViewer:
    def __init__(self, root, image_source, model_path):
        self.root = root
        self.root.title("Microplastic Detection Viewer")
        self.root.geometry("1000x800")

        self.image_files = self.get_image_list(image_source)
        self.processed_results = []
        self.current_idx = 0
        self.is_processing = True
        self.model_path = model_path
        self.progress_val = 0.0

        self.status_frame = tk.Frame(root, pady=5)
        self.status_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = tk.Label(self.status_frame, text="Initializing...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()

        self.progress = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        self.canvas_frame = tk.Frame(root, bg="#2b2b2b")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lbl_image = tk.Label(self.canvas_frame, bg="#2b2b2b", text="Waiting for YOLO...", fg="white")
        self.lbl_image.pack(expand=True, fill=tk.BOTH)

        self.nav_frame = tk.Frame(root, pady=15, bg="#f0f0f0")
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(self.nav_frame, text="<< Previous", command=self.show_prev, state=tk.DISABLED, width=15)
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.lbl_counter = tk.Label(self.nav_frame, text="0 / 0", font=("Arial", 10))
        self.lbl_counter.pack(side=tk.LEFT, padx=20)

        self.btn_next = tk.Button(self.nav_frame, text="Next >>", command=self.show_next, state=tk.DISABLED, width=15)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

        if not self.image_files:
            self.lbl_status.config(text=f"❌ Error: No images found at {image_source}")
            return

        self.thread = threading.Thread(target=self.run_inference_thread, daemon=True)
        self.thread.start()

        self.check_for_updates()

    def get_image_list(self, path):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_paths = []
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        image_paths.append(os.path.join(root, file))
            return sorted(image_paths)
        return []

    def run_inference_thread(self):
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_processing = False
            return

        total = len(self.image_files)

        for i, img_path in enumerate(self.image_files):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            results = model.predict(
                source=img_path,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                verbose=False,
                agnostic_nms=False  # helps reduce weird duplicates
            )

            r0 = results[0]

            # draw masks robustly
            vis_bgr = overlay_masks_on_bgr(img_bgr, r0, alpha=0.35, draw_contours=True)

            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_rgb)

            count = count_instances(r0)
            filename = os.path.basename(img_path)

            self.processed_results.append({
                "image": pil_img,
                "count": count,
                "filename": filename
            })

            self.progress_val = (i + 1) / total * 100

        self.is_processing = False

    def check_for_updates(self):
        self.progress['value'] = self.progress_val

        if len(self.processed_results) > 0 and self.lbl_image.cget("image") == "":
            self.update_display()

        self.update_buttons()

        total = len(self.image_files)
        processed = len(self.processed_results)

        if self.is_processing:
            self.lbl_status.config(text=f"⚙️ Processing: {processed}/{total} images ready...")
            self.root.after(100, self.check_for_updates)
        else:
            self.lbl_status.config(text=f"✅ Complete! Processed {total} images.")
            self.update_buttons()

    def update_display(self):
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
        self.root.title(f"Viewer - {data['filename']} | Count: {data['count']}")

    def update_buttons(self):
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
