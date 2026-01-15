# 2_mask_panos.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

INPUT_DIR = "input_panos"
OUTPUT_IMG_DIR = "masked_panos/images"
OUTPUT_MASK_DIR = "masked_panos/masks"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# COCO classes we want to mask
MASK_CLASSES = {
    "person",
    "car",
    "bus",
    "truck"
}

def mask_image(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    results = model(img, imgsz=1024, conf=0.3)

    for r in results:
        if r.masks is None:
            continue

        for cls, seg in zip(r.boxes.cls, r.masks.data):
            label = model.names[int(cls)]
            if label in MASK_CLASSES:
                seg = seg.cpu().numpy().astype(np.uint8) * 255
                seg = cv2.resize(seg, (w, h))
                mask = cv2.bitwise_or(mask, seg)

    masked = img.copy()
    masked[mask > 0] = 0  # black-out masked regions

    return masked, mask

for f in os.listdir(INPUT_DIR):
    if not f.lower().endswith(".jpg"):
        continue

    print(f"🎭 Masking {f}")
    img_path = os.path.join(INPUT_DIR, f)

    masked, mask = mask_image(img_path)

    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f), masked)
    cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f.replace(".jpg", "_mask.png")), mask)
