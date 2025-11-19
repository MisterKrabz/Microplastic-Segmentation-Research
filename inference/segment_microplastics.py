# segment_microplastics.py
#
# End-to-end script for segmenting, counting, and sizing microplastics
# using a Keras U-Net model and OpenCV for post-processing.
#
# Workflow:
# 1. Build a U-Net model for semantic segmentation.
# 2. Train the model on annotated images and masks.
# 3. Predict on a new image to generate a class mask.
# 4. Post-process the mask with OpenCV to count and measure instances.

import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import argparse # Import argparse for command-line functionality

# --- 1. MODEL DEFINITION: U-NET ARCHITECTURE ---
# Standard U-Net model for semantic segmentation.
def build_unet(input_shape, num_classes):
    """
    Builds a U-Net model using the Keras functional API.
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for segmentation.
    Returns:
        A Keras Model instance.
    """
    inputs = keras.Input(shape=input_shape)

    # --- Encoder Path ---
    # Downsampling path to capture context.
    # Block 1
    c1 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)

    # Block 2
    c2 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)

    # --- Bottleneck ---
    c_mid = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c_mid = layers.Dropout(0.2)(c_mid)
    c_mid = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c_mid)

    # --- Decoder Path ---
    # Upsampling path for precise localization.
    # Block 3
    u3 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c_mid)
    u3 = layers.concatenate([u3, c2])
    c3 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u3)
    c3 = layers.Dropout(0.1)(c3)
    c3 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Block 4
    u4 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c1])
    c4 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    # --- Output Layer ---
    # Final layer uses softmax for multi-class pixel classification.
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c4)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# --- 2. DATA HANDLING ---
# IMPORTANT: You must prepare your own dataset and adapt this function.
def load_and_preprocess_data(image_dir, mask_dir, target_size=(256, 256)):
    """
    Loads images and masks from directories and preprocesses them.
    
    Args:
        image_dir (str): Path to the directory containing original images.
        mask_dir (str): Path to the directory containing segmentation masks.
        target_size (tuple): The size to resize images and masks to.

    Returns:
        A tuple of (numpy_array_of_images, numpy_array_of_masks).
    """
    print("Loading and preprocessing data...")
    # This is a placeholder. You need to implement this based on your file structure.
    # Example assumes filenames match between images and masks.
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    
    images = []
    masks = []

    for img_fn, mask_fn in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_fn)
        mask_path = os.path.join(mask_dir, mask_fn)

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}. Skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
        img = cv2.resize(img, target_size)
        images.append(img)

        # Load and resize mask
        # Ensure mask is loaded as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: could not read mask {mask_path}. Skipping corresponding image.")
            images.pop() # Remove the image that has no mask
            continue
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    # Normalize images and expand mask dimensions for training
    images = np.array(images) / 255.0
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    
    print(f"Loaded {len(images)} images and masks.")
    return images, masks


# --- 3. ANALYSIS AND VISUALIZATION ---
# Takes a trained model and a new image, performs segmentation,
# counts and measures microplastics, and visualizes the result.
def segment_and_analyze(model, image_path, target_size=(256, 256), pixel_to_micron_ratio=1.0):
    """
    Takes a trained model and a new image, performs segmentation,
    counts and measures microplastics, and visualizes the result.

    Args:
        model: The trained Keras model.
        image_path (str): Path to the new image to analyze.
        target_size (tuple): The size the model expects.
        pixel_to_micron_ratio (float): Calibration factor to convert pixel area to real units.
    """
    # Load and preprocess the single image
    if not os.path.exists(image_path):
        print(f"Error: Image path not found at {image_path}")
        return

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    h, w, _ = original_image.shape
    
    # Resize for model prediction
    image_resized = cv2.resize(original_image, target_size)
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # --- Prediction ---
    predicted_mask = model.predict(image_batch)
    # Get the class with the highest probability for each pixel
    predicted_mask = np.argmax(predicted_mask[0], axis=-1)
    
    # Resize mask back to original image size for overlay
    predicted_mask_full_size = cv2.resize(predicted_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # --- Post-processing with OpenCV ---
    # Isolate the 'microplastic' class (assuming class ID 1)
    # Your class IDs: 0=background, 1=microplastic, 2=bubble
    microplastic_mask = (predicted_mask_full_size == 1).astype(np.uint8)

    # Find connected components (individual particles)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(microplastic_mask, 8, cv2.CV_32S)

    # The first label (0) is the background, so we ignore it
    particle_count = num_labels - 1
    particle_areas_pixels = stats[1:, cv2.CC_STAT_AREA]
    particle_areas_microns = particle_areas_pixels * (pixel_to_micron_ratio ** 2)
    
    print(f"\n--- Analysis Results for {os.path.basename(image_path)} ---")
    print(f"Detected {particle_count} microplastic particles.")
    
    # Create an overlay image for visualization
    overlay = original_image.copy()
    for i in range(1, num_labels): # Iterate from 1 to skip background
        # Draw a bounding box and label each particle
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        
        area_micron = particle_areas_microns[i-1]
        label = f"Area: {area_micron:.1f} um^2"
        cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Visualization ---
    # The output is now displayed in an OpenCV window instead of a plot
    # for better integration.
    # We need to convert the RGB overlay to BGR for OpenCV display.
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"Analysis Overlay: {os.path.basename(image_path)}", overlay_bgr)
    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 4. MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    # Define a parser to handle command-line arguments for different modes.
    parser = argparse.ArgumentParser(description='Microplastic Segmentation and Analysis.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='Set the script to "train" a new model or "predict" on an image.')
    parser.add_argument('--image', type=str, help='Path to the image for prediction.')
    args = parser.parse_args()

    # --- Configuration ---
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3
    NUM_CLASSES = 3 # 0=background, 1=microplastic, 2=bubble
    MODEL_PATH = 'microplastic_segmentation_model.keras'
    
    # --- Mode: Training ---
    if args.mode == 'train':
        print("\n--- Running in TRAIN mode ---")
        # TODO: dataset paths; replace 
        IMAGE_DIR = 'data/images'
        MASK_DIR = 'data/masks'
        
        # Create dummy directories and a sample image if they don't exist
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)
            dummy_img = np.random.randint(0, 255, size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
            cv2.imwrite(os.path.join(IMAGE_DIR, 'sample_image.png'), dummy_img)
        if not os.path.exists(MASK_DIR):
            os.makedirs(MASK_DIR)
            dummy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            dummy_mask[100:150, 100:150] = 1 # A fake microplastic
            dummy_mask[200:220, 50:70] = 2 # A fake bubble
            cv2.imwrite(os.path.join(MASK_DIR, 'sample_image_mask.png'), dummy_mask)

        # Build the model
        model = build_unet(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            num_classes=NUM_CLASSES
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        
        # Load Data and Train
        print("\n--- Starting Model Training ---")
        images, masks = load_and_preprocess_data(IMAGE_DIR, MASK_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH))
        
        # --- NEW CODE: Robust handling of small datasets ---
        # If the dataset is too small for a validation split, train on the full set.
        # This prevents a crash during initial testing with placeholder data.
        if len(images) < 10:
            print("NOTE: Dataset is too small for a validation split. Training on the full dataset.")
            history = model.fit(images, masks, batch_size=4, epochs=5)
        else:
            print("NOTE: Dataset is large enough for a validation split.")
            history = model.fit(images, masks, batch_size=4, epochs=5, validation_split=0.1)
        
        # Save the trained model
        model.save(MODEL_PATH)
        print(f"\nTraining complete. Model saved to {MODEL_PATH}")

    # --- Mode: Prediction ---
    elif args.mode == 'predict':
        print("\n--- Running in PREDICT mode ---")
        if not args.image:
            print("Error: For predict mode, you must provide an image path using --image")
        elif not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}. Please run in 'train' mode first.")
        else:
            # Load the pre-trained model
            print(f"Loading model from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH)
            
            # You must determine this from your microscope/camera setup.
            MICRON_PER_PIXEL = 0.5 
            
            # Run the analysis on the specified image.
            segment_and_analyze(
                model, 
                args.image, 
                target_size=(IMG_HEIGHT, IMG_WIDTH), 
                pixel_to_micron_ratio=MICRON_PER_PIXEL
            )

