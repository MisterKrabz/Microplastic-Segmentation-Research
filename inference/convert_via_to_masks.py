# convert_via_to_masks.py
#
# Converts a VIA (VGG Image Annotator) JSON project file into 
# grayscale PNG masks suitable for semantic segmentation model training.

import json
import os
import argparse
import cv2
import numpy as np
from skimage.draw import polygon

def convert_via_to_masks(json_path, image_dir, output_dir, class_mapping):
    """
    Processes a VIA JSON file and generates segmentation masks.

    Args:
        json_path (str): Path to the VIA project JSON file.
        image_dir (str): Path to the directory containing the original images.
        output_dir (str): Directory where the output masks will be saved.
        class_mapping (dict): A dictionary mapping class names to pixel values.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_metadata_key = '_via_img_metadata' if '_via_img_metadata' in data else list(data.keys())[0]
    images_data = data[image_metadata_key]

    for image_id, image_info in images_data.items():
        filename = image_info['filename']
        regions = image_info['regions']

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Warning: Could not find original image at {image_path}. Skipping mask creation for it.")
            continue

        original_image = cv2.imread(image_path)
        height, width, _ = original_image.shape

        # Create a blank mask (background class = 0)
        mask = np.zeros((height, width), dtype=np.uint8)

        for region in regions:
            region_class = region['region_attributes'].get('class')
            if not region_class:
                print(f"Warning: A region in {filename} has no 'class' attribute. Skipping.")
                continue

            pixel_value = class_mapping.get(region_class)
            if pixel_value is None:
                print(f"Warning: Class '{region_class}' in {filename} is not in your class_mapping. Skipping.")
                continue

            shape_attributes = region['shape_attributes']
            if shape_attributes['name'] != 'polygon':
                continue

            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']

            rr, cc = polygon(all_points_y, all_points_x, shape=(height, width))
            mask[rr, cc] = pixel_value

        # Save the final mask
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, mask)
        print(f"Successfully created mask: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VIA JSON annotations to PNG segmentation masks.')
    parser.add_argument('--json', type=str, required=True, help='Path to the input VIA project JSON file.')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to the directory with original images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory to save output masks.')
    args = parser.parse_args()

    # This mapping MUST match your script's logic: 0=background, 1=microplastic, 2=bubble
    CLASS_TO_PIXEL_VALUE = {
        'microplastic': 1,
        'bubble': 2
    }

    convert_via_to_masks(args.json, args.img_dir, args.output_dir, CLASS_TO_PIXEL_VALUE)