import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

# --- HELPER FUNCTIONS (From your original script) ---
def get_polygon_from_yolo(line, img_w, img_h):
    parts = line.strip().split()
    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    points = [(int(coords[i] * img_w), int(coords[i+1] * img_h)) for i in range(0, len(coords), 2)]
    return class_id, points

def is_interior(points, w, h, margin=5):
    if not points: return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) > margin and min(ys) > margin and
            max(xs) < (w - margin) and max(ys) < (h - margin))

# --- MAIN PREVIEW SCRIPT ---
def main():
    print("🔍 Scanning for images to preview...")
    data_root = "./../../datasets/training_datasets/MicroplasticsAggregate.v4-negativecropsincluded.yolov11"
    
    # Grab images from all possible splits
    image_paths = glob.glob(os.path.join(data_root, "**", "images", "*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_paths:
        print("❌ No images found in data_root!")
        return

    # Shuffle so you aren't just looking at the exact same sequence every time
    random.shuffle(image_paths)
    print(f"✅ Found {len(image_paths)} images. Launching live preview...")
    print("🖼️ Close the image window to proceed to the next one. Press Ctrl+C in terminal to quit.")

    for img_path in image_paths:
        lbl_path = img_path.replace("images", "labels")
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
        
        if not os.path.exists(lbl_path): 
            continue
        
        # Load Original Image
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            continue
            
        # Create a copy that we will augment in memory
        img_aug = img_orig.copy()
        h, w = img_aug.shape[:2]
        img_bbox = box(0, 0, w, h)
        
        # Parse existing objects
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        current_objects = []
        interior_objects = []
        
        for line in lines:
            class_id, points = get_polygon_from_yolo(line, w, h)
            if len(points) >= 3:
                obj_data = {"class_id": class_id, "points": points}
                current_objects.append(obj_data)
                if is_interior(points, w, h):
                    interior_objects.append(obj_data)
        
        # ==========================================
        # 1. APPLY COPY-PASTE (If safe particles exist)
        # ==========================================
        paste_count = 0
        if interior_objects:
            existing_polys = [Polygon(obj["points"]).buffer(0) for obj in current_objects if not Polygon(obj["points"]).buffer(0).is_empty]
            
            num_pastes = random.randint(3, 8)
            is_cluster_mode = random.random() < 0.70
            cluster_cx = random.randint(int(w*0.2), int(w*0.8))
            cluster_cy = random.randint(int(h*0.2), int(h*0.8))
            
            for _ in range(num_pastes):
                src_obj = random.choice(interior_objects)
                src_points = np.array(src_obj["points"], np.int32)
                
                x, y, obj_w, obj_h = cv2.boundingRect(src_points)
                if obj_w <= 1 or obj_h <= 1: continue
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [src_points], 255)
                roi = img_orig[y:y+obj_h, x:x+obj_w] # Pull from untouched original
                roi_mask = mask[y:y+obj_h, x:x+obj_w]
                
                radius = max(obj_w, obj_h) * 1.0
                
                for attempt in range(50):
                    if is_cluster_mode and attempt < 25:
                        new_x = int(random.gauss(cluster_cx, radius)) - obj_w // 2
                        new_y = int(random.gauss(cluster_cy, radius)) - obj_h // 2
                        radius += 3
                    else:
                        new_x = random.randint(-int(obj_w*0.8), w - int(obj_w*0.2))
                        new_y = random.randint(-int(obj_h*0.8), h - int(obj_h*0.2))
                    
                    tx1, ty1 = max(0, new_x), max(0, new_y)
                    tx2, ty2 = min(w, new_x + obj_w), min(h, new_y + obj_h)
                    
                    if tx1 >= tx2 or ty1 >= ty2: continue
                        
                    shift_x, shift_y = new_x - x, new_y - y
                    raw_candidate_points = [(px + shift_x, py + shift_y) for px, py in src_obj["points"]]
                    raw_poly = Polygon(raw_candidate_points).buffer(0)
                    clipped_poly = raw_poly.intersection(img_bbox)
                    
                    if clipped_poly.area < (raw_poly.area * 0.20): continue
                        
                    overlap = any((clipped_poly.intersects(ex_poly) and clipped_poly.intersection(ex_poly).area > 0) for ex_poly in existing_polys)
                    
                    if not overlap:
                        sx1, sy1 = int(x + (tx1 - new_x)), int(y + (ty1 - new_y))
                        sx2, sy2 = int(x + (tx2 - new_x)), int(y + (ty2 - new_y))
                        
                        roi_cropped = roi[sy1-y:sy2-y, sx1-x:sx2-x]
                        roi_mask_cropped = roi_mask[sy1-y:sy2-y, sx1-x:sx2-x]
                        
                        actual_h, actual_w = roi_cropped.shape[:2]
                        tx2, ty2 = tx1 + actual_w, ty1 + actual_h
                        img_target = img_aug[ty1:ty2, tx1:tx2]
                        
                        if img_target.shape[:2] != roi_cropped.shape[:2] or actual_h == 0 or actual_w == 0:
                            continue
                            
                        roi_inv_mask = cv2.bitwise_not(roi_mask_cropped)
                        img_bg = cv2.bitwise_and(img_target, img_target, mask=roi_inv_mask)
                        img_fg = cv2.bitwise_and(roi_cropped, roi_cropped, mask=roi_mask_cropped)
                        img_aug[ty1:ty2, tx1:tx2] = cv2.add(img_bg, img_fg)
                        
                        existing_polys.append(clipped_poly)
                        paste_count += 1
                        break 

        # ==========================================
        # 2. APPLY ROTATION
        # ==========================================
        angle = random.choice([90, 180, 270])
        if angle == 90:
            img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img_aug = cv2.rotate(img_aug, cv2.ROTATE_180)
        else: 
            img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ==========================================
        # 3. APPLY BLUR
        # ==========================================
        is_blurred = False
        if random.random() < 0.50:
            ksize = random.choice([3, 5, 7])
            img_aug = cv2.GaussianBlur(img_aug, (ksize, ksize), 0)
            is_blurred = True

        # ==========================================
        # 4. DISPLAY SIDE-BY-SIDE
        # ==========================================
        # Convert BGR to RGB for matplotlib
        img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_aug_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.canvas.manager.set_window_title(f"Live Preview - {os.path.basename(img_path)}")
        
        axes[0].imshow(img_orig_rgb)
        axes[0].set_title(f"Original\n({len(current_objects)} existing particles)", fontsize=14, pad=10)
        axes[0].axis("off")
        
        blur_text = "Yes" if is_blurred else "No"
        mode_text = "Cluster" if ('is_cluster_mode' in locals() and is_cluster_mode) else "Scatter"
        
        axes[1].imshow(img_aug_rgb)
        axes[1].set_title(f"Live Augmented (NOT SAVED)\nRotated {angle}° | Blurred: {blur_text} | Pasted: {paste_count} ({mode_text})", fontsize=14, pad=10)
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show(block=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPreview terminated by user.")