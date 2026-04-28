import os
import glob
import random
import cv2
import numpy as np
from shapely.geometry import Polygon, box

def get_polygon_from_yolo(line, img_w, img_h):
    """Parses YOLO instance segmentation multi-point coordinates."""
    parts = line.strip().split()
    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    points = [(int(coords[i] * img_w), int(coords[i+1] * img_h)) for i in range(0, len(coords), 2)]
    return class_id, points

def yolo_format(class_id, points, new_w, new_h):
    """Converts pixel coordinates back to YOLO normalized floats."""
    coords = []
    for x, y in points:
        coords.append(f"{max(0, min(x, new_w))/new_w:.6f}")
        coords.append(f"{max(0, min(y, new_h))/new_h:.6f}")
    return f"{class_id} " + " ".join(coords)

def rotate_points(points, angle, w, h):
    """Mathematically rotates polygon coordinates to match the rotated image matrix."""
    new_points = []
    for x, y in points:
        if angle == 90:
            new_points.append((h - y, x))
        elif angle == 180:
            new_points.append((w - x, h - y))
        elif angle == 270:
            new_points.append((y, w - x))
    return new_points

def extract_valid_coords(geom):
    """Safely extracts coordinates if clipping or .buffer(0) breaks the polygon into pieces."""
    if geom.is_empty: return []
    
    # If the fix resulted in a single polygon, return its coordinates
    if geom.geom_type == 'Polygon':
        return list(geom.exterior.coords)
    
    # If the fix caused the shape to pinch into multiple polygons, keep the biggest one
    elif geom.geom_type == 'MultiPolygon':
        largest = max(geom.geoms, key=lambda p: p.area)
        return list(largest.exterior.coords)
    
    # If it created a collection of polygons and random lines/dots, filter and keep the biggest polygon
    elif geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        if polys:
            largest = max(polys, key=lambda p: p.area)
            return extract_valid_coords(largest) # Recursively extract
            
    return []

def is_interior(points, w, h, margin=5):
    """Checks if a particle is completely away from the image edges to avoid cloning semicircles."""
    if not points: return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) > margin and min(ys) > margin and
            max(xs) < (w - margin) and max(ys) < (h - margin))

def main():
    print("--- 🧬 STARTING INSTANCE-MASK CLUSTER AUGMENTATION ---")
    data_root = "data_root"
    
    # Grab images from all possible splits
    image_paths = glob.glob(os.path.join(data_root, "**", "train", "images", "*.*"), recursive=True)
    image_paths += glob.glob(os.path.join(data_root, "**", "valid", "images", "*.*"), recursive=True)
    image_paths += glob.glob(os.path.join(data_root, "**", "val", "images", "*.*"), recursive=True)
    
    # Filter strictly for images
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    total_existing = len(image_paths)
    TARGET_TOTAL = 300  # Will generate 182 new images if you have 118 base images
    num_to_generate = TARGET_TOTAL - total_existing
    TARGET_VALID_NEW = int(num_to_generate * 0.10) # Send ~10% of new data to validation
    
    if num_to_generate <= 0:
        print(f"Dataset already has {total_existing} images. No generation needed.")
        return

    print(f"Found {total_existing} base images. Generating {num_to_generate} new synthetic images...")
    images_to_augment = random.choices(image_paths, k=num_to_generate)
    valid_count, train_count = 0, 0
    blur_count = 0
    
    for i, img_path in enumerate(images_to_augment):
        # Find the paired YOLO text label file
        lbl_path = img_path.replace("images", "labels")
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
        
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img_bbox = box(0, 0, w, h)
        
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        current_objects = []
        interior_objects = []  # ONLY whole, un-cut particles
        
        for line in lines:
            class_id, points = get_polygon_from_yolo(line, w, h)
            if len(points) >= 3:
                obj_data = {"class_id": class_id, "points": points}
                current_objects.append(obj_data)
                
                # Flag it as a safe source object only if it's strictly inside the image bounds
                if is_interior(points, w, h):
                    interior_objects.append(obj_data)
        
        # Only attempt to paste new particles if there is at least one safe, whole source particle!
        if interior_objects:
            # Apply .buffer(0) magic fix to all existing polygons in the scene to prevent math crashes
            existing_polys = []
            for obj in current_objects:
                poly = Polygon(obj["points"]).buffer(0)
                if not poly.is_empty:
                    existing_polys.append(poly)
            
            num_pastes = random.randint(3, 8) # Add 3 to 8 new microplastics per image
            
            # --- ALGORITHM: 70% Cluster Mode, 30% Scatter Mode ---
            is_cluster_mode = random.random() < 0.70
            cluster_cx = random.randint(int(w*0.2), int(w*0.8))
            cluster_cy = random.randint(int(h*0.2), int(h*0.8))
            
            for _ in range(num_pastes):
                # SECURE: Only copy from perfect, whole particles!
                src_obj = random.choice(interior_objects)
                src_points = np.array(src_obj["points"], np.int32)
                
                x, y, obj_w, obj_h = cv2.boundingRect(src_points)
                if obj_w <= 1 or obj_h <= 1: continue
                
                # Cut out the EXACT instance mask polygon, not the bounding box
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [src_points], 255)
                roi = img[y:y+obj_h, x:x+obj_w]
                roi_mask = mask[y:y+obj_h, x:x+obj_w]
                
                pasted = False
                radius = max(obj_w, obj_h) * 1.0 # Start with a very tight cluster radius
                
                for attempt in range(50):
                    # If cluster is too dense after 25 tries, scatter to guarantee placement
                    if is_cluster_mode and attempt < 25:
                        new_x = int(random.gauss(cluster_cx, radius)) - obj_w // 2
                        new_y = int(random.gauss(cluster_cy, radius)) - obj_h // 2
                        radius += 3 # Slowly expand the drop zone if it's too crowded
                    else:
                        new_x = random.randint(-int(obj_w*0.8), w - int(obj_w*0.2))
                        new_y = random.randint(-int(obj_h*0.8), h - int(obj_h*0.2))
                    
                    tx1, ty1 = max(0, new_x), max(0, new_y)
                    tx2, ty2 = min(w, new_x + obj_w), min(h, new_y + obj_h)
                    
                    if tx1 >= tx2 or ty1 >= ty2: continue
                        
                    shift_x = new_x - x
                    shift_y = new_y - y
                    raw_candidate_points = [(px + shift_x, py + shift_y) for px, py in src_obj["points"]]
                    
                    # Apply .buffer(0) magic fix to the newly generated clone
                    raw_poly = Polygon(raw_candidate_points).buffer(0)
                    clipped_poly = raw_poly.intersection(img_bbox)
                    
                    if clipped_poly.area < (raw_poly.area * 0.20): continue
                        
                    # STRICT 0-OVERLAP CHECK ON THE POLYGON MASK
                    overlap = False
                    for ex_poly in existing_polys:
                        if clipped_poly.intersects(ex_poly) and clipped_poly.intersection(ex_poly).area > 0:
                            overlap = True
                            break
                    
                    if not overlap:
                        sx1, sy1 = int(x + (tx1 - new_x)), int(y + (ty1 - new_y))
                        sx2, sy2 = int(x + (tx2 - new_x)), int(y + (ty2 - new_y))
                        
                        # Cut the source arrays
                        roi_cropped = img[sy1:sy2, sx1:sx2]
                        roi_mask_cropped = mask[sy1:sy2, sx1:sx2]
                        
                        # --- SAFEGUARD: Prevent OpenCV Shape Mismatch ---
                        actual_h, actual_w = roi_cropped.shape[:2]
                        tx2, ty2 = tx1 + actual_w, ty1 + actual_h
                        img_target = img[ty1:ty2, tx1:tx2]
                        
                        if img_target.shape[:2] != roi_cropped.shape[:2] or actual_h == 0 or actual_w == 0:
                            continue
                        # ------------------------------------------------
                        
                        roi_inv_mask = cv2.bitwise_not(roi_mask_cropped)
                        img_bg = cv2.bitwise_and(img_target, img_target, mask=roi_inv_mask)
                        img_fg = cv2.bitwise_and(roi_cropped, roi_cropped, mask=roi_mask_cropped)
                        img[ty1:ty2, tx1:tx2] = cv2.add(img_bg, img_fg)
                        
                        existing_polys.append(clipped_poly)
                        valid_coords = extract_valid_coords(clipped_poly)
                        if valid_coords:
                            current_objects.append({"class_id": src_obj["class_id"], "points": valid_coords})
                        pasted = True
                        break 
        
        # 1. Rotate the image to break spatial memory
        angle = random.choice([90, 180, 270])
        if angle == 90:
            rotated_img, new_w, new_h = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), h, w
        elif angle == 180:
            rotated_img, new_w, new_h = cv2.rotate(img, cv2.ROTATE_180), w, h
        else: 
            rotated_img, new_w, new_h = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), h, w
            
        # 2. Add Gaussian Blur to 50% of the synthetic images
        if random.random() < 0.50:
            ksize = random.choice([3, 5, 7]) # Mild, Medium, or Heavy blur
            rotated_img = cv2.GaussianBlur(rotated_img, (ksize, ksize), 0)
            blur_count += 1
            
        # 3. Determine output directory
        dest_split = "valid" if valid_count < TARGET_VALID_NEW else "train"
        if dest_split == "valid": valid_count += 1
        else: train_count += 1
            
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        new_img_dir = os.path.join(base_dir, dest_split, "images")
        new_lbl_dir = os.path.join(base_dir, dest_split, "labels")
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_lbl_dir, exist_ok=True)
        
        # 4. Save the synthetic image and updated labels
        base_name = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
        new_img_name = f"{base_name}_aug_rot{angle}_{i}.jpg"
        new_lbl_name = f"{base_name}_aug_rot{angle}_{i}.txt"
        
        cv2.imwrite(os.path.join(new_img_dir, new_img_name), rotated_img)
        with open(os.path.join(new_lbl_dir, new_lbl_name), "w") as f:
            for obj in current_objects:
                rotated_pts = rotate_points(obj["points"], angle, w, h)
                f.write(yolo_format(obj["class_id"], rotated_pts, new_w, new_h) + "\n")
                
    print(f"✅ Augmentation Complete!")
    print(f"   -> Added {train_count} new images to Train, {valid_count} to Valid.")
    print(f"   -> {blur_count} of these new images were blurred for edge-case training.")

if __name__ == "__main__":
    main()