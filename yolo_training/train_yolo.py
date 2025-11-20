from ultralytics import YOLO
import os

def main():
    # 1. Load a pretrained Nano model (Fastest, good for simple blobs)
    # 'yolov8n.pt' is standard detection. 'yolov8n-obb.pt' is for oriented boxes.
    # Start with standard detection to prove the pipeline.
    model = YOLO('yolov8n.pt')  

    # 2. Path to the dataset.yaml we just generated
    # We use absolute path to be safe
    dataset_yaml = os.path.abspath("./yolo_dataset/dataset.yaml")

    print(f"Training on dataset: {dataset_yaml}")

    # 3. Train
    # device='mps' uses Mac GPU
    results = model.train(
        data=dataset_yaml,
        epochs=50,          # 50 is usually plenty for this
        imgsz=640,          # Standard YOLO size
        device='mps',       # Use Apple Metal acceleration
        batch=16,
        name='microplastic_hunter'
    )

    # 4. Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")

    # 5. Export for later use
    path = model.export(format="onnx")
    print(f"Model exported to {path}")

if __name__ == "__main__":
    main()