import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("https://huggingface.co/wuriyanto/yolo8-indonesian-license-plate-detection/resolve/main/model.pt")

INPUT_DIR = "license_plates"
OUTPUT_DIR = "license_plates_cropped_whole"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Run detection
        results = model(image, conf=0.5)
        
        # Extract and save crops
        for i, result in enumerate(results):
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    height = y2 - y1
                    # Expand the crop downward by 25%. This is an adjustment since 
                    # it was cropping about 1/3 of the plate
                    y2_expanded = min(image.shape[0], y2 + int(height * 0.25))
                    cropped = image[y1:y2_expanded, x1:x2]
                    # cropped = image[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        out_path = os.path.join(OUTPUT_DIR, 
                                               f"{os.path.splitext(filename)[0]}_plate_{j}.jpg")
                        cv2.imwrite(out_path, cropped)
                        print(f"Saved: {out_path}")

print("Done!")