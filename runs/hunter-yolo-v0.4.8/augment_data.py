import os
import glob
import random
import cv2
import numpy as np
from shapely.geometry import Polygon, box

def get_polygon_from_yolo(line, img_w, img_h):
    parts = line.strip().split()
    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    points = [(int(coords[i] * img_w), int(coords[i+1] * img_h)) for i in range(0, len(coords), 2)]
    return class_id, points

def yolo_format(class_id, points, new_w, new_h):
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
    """Safely extracts coordinates if clipping an object off the screen breaks it into pieces."""
    if geom.is_empty: return []
    if geom.geom_type == 'Polygon':
        return list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        largest = max(geom.geoms, key=lambda p: p.area)
        return list(largest.exterior.coords)
    return []

def main():
    print("--- 🧬 STARTING ROTATIONAL WITHIN-IMAGE AUGMENTATION ---")
    data_root = "data_root"
    
    # Grab both train and valid images to get the true total dataset size
    image_paths = glob.glob(os.path.join(data_root, "**", "train", "images", "*.*"), recursive=True)
    image_paths += glob.glob(os.path.join(data_root, "**", "valid", "images", "*.*"), recursive=True)
    image_paths += glob.glob(os.path.join(data_root, "**", "val", "images", "*.*"), recursive=True)
    
    # Filter for only images
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.png'))]
    
    total_existing = len(image_paths)
    TARGET_TOTAL = 110
    num_to_generate = TARGET_TOTAL - total_existing
    TARGET_VALID_NEW = 5  # How many of the generated images go to the validation set
    
    if num_to_generate <= 0:
        print(f"Dataset already has {total_existing} images. No generation needed.")
        return

    print(f"Found {total_existing} base images. Generating {num_to_generate} new mixed/rotated images...")
    
    images_to_augment = random.choices(image_paths, k=num_to_generate)
    valid_count = 0
    train_count = 0
    
    for i, img_path in enumerate(images_to_augment):
        lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img_bbox = box(0, 0, w, h)
        
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        current_objects = []
        for line in lines:
            class_id, points = get_polygon_from_yolo(line, w, h)
            if len(points) >= 3:
                current_objects.append({"class_id": class_id, "points": points})
        
        if current_objects:
            existing_polys = [Polygon(obj["points"]) for obj in current_objects]
            num_pastes = random.randint(2, 6)
            
            for _ in range(num_pastes):
                src_obj = random.choice(current_objects)
                src_points = np.array(src_obj["points"], np.int32)
                
                x, y, obj_w, obj_h = cv2.boundingRect(src_points)
                if obj_w <= 1 or obj_h <= 1: continue
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [src_points], 255)
                roi = img[y:y+obj_h, x:x+obj_w]
                roi_mask = mask[y:y+obj_h, x:x+obj_w]
                
                for attempt in range(50):
                    new_x = random.randint(-int(obj_w*0.8), w - int(obj_w*0.2))
                    new_y = random.randint(-int(obj_h*0.8), h - int(obj_h*0.2))
                    
                    tx1, ty1 = max(0, new_x), max(0, new_y)
                    tx2, ty2 = min(w, new_x + obj_w), min(h, new_y + obj_h)
                    
                    if tx1 >= tx2 or ty1 >= ty2: continue
                        
                    shift_x = new_x - x
                    shift_y = new_y - y
                    raw_candidate_points = [(px + shift_x, py + shift_y) for px, py in src_obj["points"]]
                    raw_poly = Polygon(raw_candidate_points)
                    
                    clipped_poly = raw_poly.intersection(img_bbox)
                    if clipped_poly.area < (raw_poly.area * 0.20): continue
                        
                    overlap = False
                    for ex_poly in existing_polys:
                        if clipped_poly.intersects(ex_poly) and clipped_poly.intersection(ex_poly).area > 0:
                            overlap = True
                            break
                    
                    if not overlap:
                        sx1, sy1 = x + (tx1 - new_x), y + (ty1 - new_y)
                        sx2, sy2 = x + (tx2 - new_x), y + (ty2 - new_y)
                        
                        roi_cropped = img[sy1:sy2, sx1:sx2]
                        roi_mask_cropped = mask[sy1:sy2, sx1:sx2]
                        img_target = img[ty1:ty2, tx1:tx2]
                        
                        roi_inv_mask = cv2.bitwise_not(roi_mask_cropped)
                        img_bg = cv2.bitwise_and(img_target, img_target, mask=roi_inv_mask)
                        img_fg = cv2.bitwise_and(roi_cropped, roi_cropped, mask=roi_mask_cropped)
                        img[ty1:ty2, tx1:tx2] = cv2.add(img_bg, img_fg)
                        
                        existing_polys.append(clipped_poly)
                        valid_coords = extract_valid_coords(clipped_poly)
                        if valid_coords:
                            current_objects.append({"class_id": src_obj["class_id"], "points": valid_coords})
                        break 
        
        # Determine rotation
        angle = random.choice([90, 180, 270])
        if angle == 90:
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            new_w, new_h = h, w
        elif angle == 180:
            rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            new_w, new_h = w, h
        else: 
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_w, new_h = h, w
            
        # Determine Target Directory (Train vs Valid)
        if valid_count < TARGET_VALID_NEW:
            dest_split = "valid"
            valid_count += 1
        else:
            dest_split = "train"
            train_count += 1
            
        # Ensure the destination directories exist
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path))) # climbs out of train/images/
        new_img_dir = os.path.join(base_dir, dest_split, "images")
        new_lbl_dir = os.path.join(base_dir, dest_split, "labels")
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_lbl_dir, exist_ok=True)
        
        # Save files
        base_name = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
        new_img_name = f"{base_name}_aug_rot{angle}_{i}.jpg"
        new_lbl_name = f"{base_name}_aug_rot{angle}_{i}.txt"
        
        new_img_path = os.path.join(new_img_dir, new_img_name)
        new_lbl_path = os.path.join(new_lbl_dir, new_lbl_name)
        
        cv2.imwrite(new_img_path, rotated_img)
        
        with open(new_lbl_path, "w") as f:
            for obj in current_objects:
                rotated_pts = rotate_points(obj["points"], angle, w, h)
                f.write(yolo_format(obj["class_id"], rotated_pts, new_w, new_h) + "\n")
                
    print(f"✅ Augmentation Complete!")
    print(f"   -> Added {train_count} new images to the Train set.")
    print(f"   -> Added {valid_count} new images to the Valid set.")

if __name__ == "__main__":
    main()