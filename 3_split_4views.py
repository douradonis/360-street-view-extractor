# 3_split_4views.py
import os
import math
import numpy as np
from PIL import Image

INPUT_DIR = "masked_panos/images"
MASK_DIR = "masked_panos/masks"
OUT_IMG_DIR = "views/images"
OUT_MASK_DIR = "views/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

DIRECTIONS = {
    "front": 0,
    "right": 90,
    "back": 180,
    "left": 270
}

SIZE = 1024
FOV = math.radians(90)

def extract_view(pano, yaw):
    w, h = pano.size
    pano = np.array(pano)
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    for y in range(SIZE):
        for x in range(SIZE):
            nx = (x / SIZE - 0.5) * 2
            ny = (y / SIZE - 0.5) * 2

            theta = math.radians(yaw) + nx * (FOV / 2)
            phi = ny * (FOV / 2)

            u = int((theta / (2 * math.pi) + 0.5) * w) % w
            v = int((0.5 - phi / math.pi) * h)
            v = max(0, min(h - 1, v))

            out[y, x] = pano[v, u]

    return Image.fromarray(out)

def split_all():
    for f in os.listdir(INPUT_DIR):
        if not f.endswith(".jpg"):
            continue

        base = f.replace(".jpg", "")
        pano = Image.open(os.path.join(INPUT_DIR, f)).convert("RGB")
        mask = Image.open(os.path.join(MASK_DIR, base + "_mask.png")).convert("L")

        for name, yaw in DIRECTIONS.items():
            img = extract_view(pano, yaw)
            msk = extract_view(mask.convert("RGB"), yaw).convert("L")

            img.save(os.path.join(OUT_IMG_DIR, f"{base}_{name}.jpg"))
            msk.save(os.path.join(OUT_MASK_DIR, f"{base}_{name}_mask.png"))

if __name__ == "__main__":
    split_all()
