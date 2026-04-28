# check_plate_sizes.py
import cv2
import os
from glob import glob
import numpy as np

def check_plate_sizes(plates_dir):
    """Analyze the sizes of your cropped license plates"""
    image_paths = glob(os.path.join(plates_dir, "*.jpg")) + \
                   glob(os.path.join(plates_dir, "*.png")) + \
                   glob(os.path.join(plates_dir, "*.jpeg"))
    
    if not image_paths:
        print(f"No images found in {plates_dir}")
        return
    
    heights = []
    widths = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)
    
    print(f"Analyzed {len(heights)} license plate images")
    print(f"Average dimensions: {np.mean(widths):.0f}×{np.mean(heights):.0f}")
    print(f"Min dimensions: {min(widths)}×{min(heights)}")
    print(f"Max dimensions: {max(widths)}×{max(heights)}")
    print(f"\nRecommended DRAW image size: at least {max(widths)}×{max(heights)}?")
    print("But 28×28 is clearly too small!")

# Change this to your plates directory
CROPPED_PLATES_DIR = "european_license_plates_dataset/dataset_final/train"
check_plate_sizes(CROPPED_PLATES_DIR)