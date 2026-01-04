import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# Can be a single file OR a folder (scanned recursively)
SOURCE_PATH = "./../datasets/raw_data/LMMP" 
MODEL_PATH = "../models/hunter-yolo-v0.2.2.pt"

# Tuning Parameters
CONFIDENCE = 0.25
IOU_THRESH = 0.5
IMG_SIZE   = 1280
# ==========================================

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
        
        # --- GUI LAYOUT ---
        
        # 1. Top Bar: Status Info
        self.status_frame = tk.Frame(root, pady=5)
        self.status_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_status = tk.Label(self.status_frame, text="Initializing...", font=("Arial", 12, "bold"))
        self.lbl_status.pack()
        
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

        self.lbl_counter = tk.Label(self.nav_frame, text="0 / 0", font=("Arial", 10))
        self.lbl_counter.pack(side=tk.LEFT, padx=20)

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
            # os.walk traverses the tree top-down
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
            # Run YOLO
            results = model.predict(
                source=img_path,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                verbose=False,
                agnostic_nms=True
            )
            
            # Process Result for Display
            res_plot = results[0].plot(line_width=2, font_size=2)
            res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image immediately to be ready for GUI
            pil_img = Image.fromarray(res_rgb)
            
            # Store result safely
            count = len(results[0].boxes)
            filename = os.path.basename(img_path)
            
            self.processed_results.append({
                "image": pil_img,
                "count": count,
                "filename": filename
            })
            
            # Update progress value for the GUI to read
            self.progress_val = (i + 1) / total * 100

        self.is_processing = False

    def check_for_updates(self):
        """Polls to see if new results arrived from the thread."""
        
        # Update Progress Bar
        if hasattr(self, 'progress_val'):
            self.progress['value'] = self.progress_val

        # If we have results and haven't shown anything yet, show the first one
        if len(self.processed_results) > 0 and self.lbl_image.cget("image") == "":
            self.update_display()

        # Update Buttons State
        self.update_buttons()
        
        # Update Status Text
        total = len(self.image_files)
        processed = len(self.processed_results)
        
        if self.is_processing:
            self.lbl_status.config(text=f"⚙️ Processing: {processed}/{total} images ready...")
            # Check again in 100ms
            self.root.after(100, self.check_for_updates)
        else:
            self.lbl_status.config(text=f"✅ Complete! Processed {total} images.")
            self.update_buttons() # Final check

    def update_display(self):
        """Updates the main image area with the current index."""
        if not self.processed_results:
            return

        data = self.processed_results[self.current_idx]
        pil_image = data["image"]
        
        # Resize logic to fit window (keep aspect ratio)
        canvas_width = self.root.winfo_width()
        canvas_height = self.root.winfo_height() - 150 # subtract UI space
        
        # Handle case where window isn't fully drawn yet
        if canvas_width < 10: canvas_width = 800
        if canvas_height < 10: canvas_height = 600

        # Create a copy to resize
        img_copy = pil_image.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img_copy)
        
        self.lbl_image.config(image=self.tk_image, text="") # Clear waiting text
        self.lbl_counter.config(text=f"Image {self.current_idx + 1} of {len(self.image_files)}")
        self.root.title(f"Viewer - {data['filename']} | Count: {data['count']}")

    def update_buttons(self):
        """Enables/Disables buttons based on available data."""
        # PREV button
        if self.current_idx > 0:
            self.btn_prev.config(state=tk.NORMAL)
        else:
            self.btn_prev.config(state=tk.DISABLED)

        # NEXT button
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
    
    # 1. Validate Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Launch App
    app = MicroplasticViewer(root, SOURCE_PATH, MODEL_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()