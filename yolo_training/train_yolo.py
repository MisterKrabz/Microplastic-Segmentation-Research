from ultralytics import YOLO
import os

def main():
    # 1. LOAD A BETTER MODEL
    # Switch from 'yolov8n.pt' (Nano) to 'yolov8m.pt' (Medium)
    # The Medium model has more layers and parameters, allowing it to learn
    # the subtle differences between the "grid" texture and the "plastic" texture.
    # This is the single biggest factor for increasing confidence.
    model = YOLO('yolov8m.pt')  

    # 2. Path to dataset
    dataset_yaml = os.path.abspath("./yolo_dataset/dataset.yaml")
    print(f"Training on dataset: {dataset_yaml}")

    # 3. TRAIN WITH AGGRESSIVE AUGMENTATION
    results = model.train(
        data=dataset_yaml,
        
        # --- Training Hyperparameters ---
        epochs=300,          # Increased: Heavy augmentation makes learning harder but better.
        patience=50,         # Early Stopping: Stop if no improvement for 50 epochs.
        imgsz=1024,          # High resolution is critical for small particles.
        device='mps',        # Apple Metal
        batch=8,             # Reduced batch size because 'Medium' model is larger.
        name='microplastic_hunter_v2',
        
        # --- THE "ZOOM" & "SECTIONS" AUGMENTATION ---
        mosaic=1.0,          # (100%) Stitches 4 images into 1. This creates new "sections" 
                             # and forces the model to find objects in complex contexts.
        scale=0.8,           # (Zoom) Scales image by +/- 80%. The model will learn 
                             # to detect tiny dots AND huge blurry blobs.
        
        # --- PHYSICS-BASED AUGMENTATION ---
        degrees=180,         # (Rotation) Microplastics have no "up" or "down". 
                             # We allow full 180-degree random rotation.
        fliplr=0.5,          # Flip Left-Right (50% chance)
        flipud=0.5,          # Flip Up-Down (50% chance)
        
        # --- LIGHTING/COLOR AUGMENTATION ---
        # Darkfield microscopy lighting varies wildly. We augment to handle this.
        hsv_h=0.015,         # Slight Hue shift
        hsv_s=0.7,           # High Saturation variance (some plastics are colorful, some gray)
        hsv_v=0.4,           # High Value (Brightness) variance
        
        # --- ADVANCED MIXING ---
        mixup=0.1,           # (10%) Blends two images together transparency-wise.
                             # Helps the model ignore transparent ghosts/artifacts.
        
        # --- REFINEMENT ---
        close_mosaic=20,     # Turn OFF mosaic for the last 20 epochs to 
                             # fine-tune on realistic, un-stitched images.
    )

    # 4. Validate
    # We validate on the best model found during training
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # 5. Export
    path = model.export(format="onnx")
    print(f"Model exported to {path}")

if __name__ == "__main__":
    main()