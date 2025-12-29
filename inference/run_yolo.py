import os
import sys
from ultralytics import YOLO

# --- CONFIGURATION ---
# PASTE YOUR COPIED IMAGE PATH INSIDE THE QUOTES BELOW:
TEST_IMAGE_PATH = "/Users/patrickwang/Workspace/URS_Project/yolo_training/yolo_dataset/train/images/YOUR_REAL_IMAGE_NAME.jpg"
# ---------------------

def main():
    # 1. Smart Load Model
    # Checks current folder first, then 'models/' folder (based on your screenshot)
    if os.path.exists("my_model.pt"):
        model_path = "my_model.pt"
    elif os.path.exists("models/my_model.pt"):
        model_path = "models/my_model.pt"
    else:
        print("❌ Error: Could not find 'my_model.pt' in current folder or 'models/' folder.")
        return

    print(f"🚀 Loading model from: {model_path}")
    model = YOLO(model_path)

    # 2. Check Image Path
    # Removes quotes if you accidentally pasted them
    image_path = TEST_IMAGE_PATH.strip().strip("'").strip('"')
    
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image file not found!")
        print(f"   Looked for: {image_path}")
        print("👉 TIP: Drag and drop an image file from Finder into the terminal to get the correct path.")
        return

    # 3. Run Inference
    print(f"📷 Running prediction on: {os.path.basename(image_path)}...")
    results = model.predict(image_path, save=True, conf=0.25)

    # 4. Success Message
    print("\n✅ Prediction complete!")
    print(f"   Results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    main()