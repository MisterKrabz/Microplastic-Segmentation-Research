import os
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# Adjusted to match your screenshot structure
SOURCE_ROOT = "../Data/Microplastics.v2-v2.coco-segmentation"
OUTPUT_DIR = "../ready_to_train_dataset"

def create_mask_from_polygon(image_shape, segmentation):
    """Draws binary mask from COCO polygon coordinates."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for polygon in segmentation:
        # Convert list of floats to integer points
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)  # Draw with value 1
    return mask

def save_augmentations(img, mask, base_name, out_img_dir, out_mask_dir):
    """Generates 6 variations of the image/mask pair."""
    
    transforms = [
        ("orig",    lambda i, m: (i, m)),
        ("flipH",   lambda i, m: (cv2.flip(i, 1), cv2.flip(m, 1))),
        ("flipV",   lambda i, m: (cv2.flip(i, 0), cv2.flip(m, 0))),
        ("rot90",   lambda i, m: (cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE))),
        ("rot180",  lambda i, m: (cv2.rotate(i, cv2.ROTATE_180), cv2.rotate(m, cv2.ROTATE_180))),
        ("rot270",  lambda i, m: (cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE))),
    ]

    for suffix, func in transforms:
        aug_img, aug_mask = func(img, mask)
        
        # Paths
        img_out = os.path.join(out_img_dir, f"{base_name}_{suffix}.jpg")
        mask_out = os.path.join(out_mask_dir, f"{base_name}_{suffix}.png")
        
        cv2.imwrite(img_out, aug_img)
        # Masks must be png to preserve values without compression artifacts
        cv2.imwrite(mask_out, aug_mask) 

def process_coco_folder(split_name, target_split_name):
    """Reads _annotations.coco.json and processes images."""
    json_path = os.path.join(SOURCE_ROOT, split_name, "_annotations.coco.json")
    
    if not os.path.exists(json_path):
        print(f"⚠️  Skipping {split_name}: JSON not found at {json_path}")
        return 0

    print(f"Processing {split_name} (saving as {target_split_name})...")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create output directories
    out_img_dir = os.path.join(OUTPUT_DIR, target_split_name, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, target_split_name, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Map Image ID to Annotations
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann['segmentation'])

    count = 0
    for img_info in tqdm(data['images']):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        # Load Image
        src_img_path = os.path.join(SOURCE_ROOT, split_name, file_name)
        img = cv2.imread(src_img_path)
        
        if img is None:
            print(f"❌ Could not load image: {src_img_path}")
            continue

        # Create Mask
        if img_id in img_to_anns:
            # Combine all polygons for this image into one mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for seg in img_to_anns[img_id]:
                # Draw each polygon
                poly_mask = create_mask_from_polygon(img.shape, seg)
                mask = np.maximum(mask, poly_mask)
        else:
            # No annotations = empty black mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Save & Augment
        base_name = os.path.splitext(file_name)[0]
        save_augmentations(img, mask, base_name, out_img_dir, out_mask_dir)
        count += 1

    return count

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Process 'train' folder -> output to 'train'
    train_count = process_coco_folder("train", "train")
    
    # Process 'test' folder -> output to 'valid' (SAM2 expects 'valid')
    # Your screenshot shows 'test', so we map it to 'valid' for training
    valid_count = process_coco_folder("test", "valid")
    
    print(f"\n--- Summary ---")
    print(f"Train Images: {train_count} (x6 variations = {train_count * 6})")
    print(f"Valid Images: {valid_count} (x6 variations = {valid_count * 6})")
    print(f"Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()