import json
import os
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
SOURCE_ROOT = "../Data/Microplastics.v2-v2.coco-segmentation"
OUTPUT_DIR = "../yolo_training/yolo_dataset"

def convert_coco_to_yolo_bbox(coco_json_path, source_img_dir, out_img_dir, out_label_dir):
    if not os.path.exists(coco_json_path):
        print(f"Skipping: {coco_json_path} not found.")
        return

    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Map Image ID -> File Name & Dimensions
    images = {img['id']: img for img in data['images']}

    # Process Annotations
    for ann in tqdm(data['annotations'], desc="Converting Labels"):
        img_id = ann['image_id']
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = img_info['file_name']

        # Copy Image
        src_img = os.path.join(source_img_dir, file_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(out_img_dir, file_name))
        
        # Convert Polygon/Bbox to YOLO format (class x_center y_center w h)
        # COCO bbox is [x_min, y_min, width, height]
        x, y, w, h = ann['bbox']
        
        # Normalize to 0-1
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Class ID (Assuming 1 class 'microplastic', so ID 0)
        class_id = 0 

        # Write label file
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(out_label_dir, label_file)
        
        with open(label_path, 'a') as lf:
            lf.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    for split, coco_split in [("train", "train"), ("val", "test")]: # Map your 'test' folder to YOLO 'val'
        print(f"Processing {split}...")
        out_img = os.path.join(OUTPUT_DIR, split, "images")
        out_lbl = os.path.join(OUTPUT_DIR, split, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        
        json_path = os.path.join(SOURCE_ROOT, coco_split, "_annotations.coco.json")
        img_dir = os.path.join(SOURCE_ROOT, coco_split)
        
        convert_coco_to_yolo_bbox(json_path, img_dir, out_img, out_lbl)
        
    # Create YOLO YAML config
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images
names:
  0: microplastic
"""
    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"\n✅ Conversion Complete. Dataset ready at {OUTPUT_DIR}")
    print(f"YAML Config created at {os.path.join(OUTPUT_DIR, 'dataset.yaml')}")

if __name__ == "__main__":
    main()