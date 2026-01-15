import os
import math
import cv2
import numpy as np
from PIL import Image
import piexif
from xml.dom import minidom
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
INPUT_PANOS = "input_panos"          # equirectangular pano images
INPUT_KML = "route.kml"              # KML με διαδρομές
OUT_IMG = "output/images"
OUT_MASK = "output/masks"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

SIZE = 1024
FOV = math.radians(90)
# offsets από forward yaw
YAW_OFFSETS = [-70, -50, -30, -10, 10, 30, 50, 70]

# Masking classes
MASK_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}

# Default altitude
DEFAULT_ALT = 2.5

# =========================
# UTILS
# =========================
def deg_to_dms_rational(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return [(d,1),(m,1),(int(s*100),100)]

def write_gps(path, lat, lon, alt=DEFAULT_ALT):
    img = Image.open(path)
    exif = piexif.load(img.info.get("exif", b""))
    exif["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >=0 else "S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >=0 else "W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lon)),
        piexif.GPSIFD.GPSAltitude: (int(alt*100),100),
    }
    img.save(path, exif=piexif.dump(exif))

def bearing(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    θ = math.atan2(x, y)
    return math.degrees(θ) % 360

def extract_view(img, yaw_deg):
    h, w = img.shape[:2]
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    yaw = math.radians(yaw_deg)
    for y in range(SIZE):
        for x in range(SIZE):
            nx = (x / SIZE - 0.5) * 2
            ny = (y / SIZE - 0.5) * 2
            theta = yaw + nx * (FOV/2)
            phi = ny * (FOV/2)
            u = int((theta/(2*math.pi)+0.5)*w) % w
            v = int((0.5 - phi/math.pi)*h)
            v = max(0, min(h-1,v))
            out[y,x] = img[v,u]
    return out

def parse_kml_points(kml_file):
    """Επιστρέφει λίστα από (lat, lon) κατά μήκος της διαδρομής"""
    doc = minidom.parse(kml_file)
    coords = []
    for linestring in doc.getElementsByTagName("LineString"):
        for coord_tag in linestring.getElementsByTagName("coordinates"):
            raw = coord_tag.firstChild.nodeValue.strip()
            for line in raw.split():
                lon, lat, *_ = map(float, line.split(","))
                coords.append((lat, lon))
    return coords

# =========================
# INIT YOLO
# =========================
model = YOLO("yolov8n-seg.pt")

# =========================
# PIPELINE
# =========================
coords = parse_kml_points(INPUT_KML)

for i, pano_file in enumerate(sorted(os.listdir(INPUT_PANOS))):
    if not pano_file.lower().endswith(".jpg"):
        continue
    print(f"📸 Processing {pano_file}")

    pano_path = os.path.join(INPUT_PANOS, pano_file)
    pano = cv2.imread(pano_path)
    h, w = pano.shape[:2]

    # Υπολογισμός forward yaw από KML
    if i < len(coords)-1:
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        center_yaw = bearing(lat1, lon1, lat2, lon2)
        lat, lon = lat1, lon1
    else:
        center_yaw = 0  # fallback
        lat, lon = coords[-1]

    # ---------- FULL PANO MASK ----------
    full_mask = np.zeros((h, w), dtype=np.uint8)
    results = model(pano, imgsz=1024, conf=0.3)
    for r in results:
        if r.masks is None:
            continue
        for cls, seg in zip(r.boxes.cls, r.masks.data):
            label = model.names[int(cls)]
            if label in MASK_CLASSES:
                seg = seg.cpu().numpy().astype(np.uint8)*255
                seg = cv2.resize(seg,(w,h))
                full_mask = cv2.bitwise_or(full_mask, seg)

    # Sky mask: top 25%
    full_mask[:int(h*0.25), :] = 255
    pano[full_mask>0] = 0

    base = pano_file.replace(".jpg","")

    # ---------- SPLIT 8 FORWARD-CAMERAS ----------
    for j, offset in enumerate(YAW_OFFSETS):
        yaw = (center_yaw + offset) % 360
        img = extract_view(pano, yaw)
        msk = extract_view(full_mask, yaw)

        img_name = f"{base}_cam{j}.jpg"
        mask_name = f"{base}_cam{j}_mask.png"

        img_path = os.path.join(OUT_IMG, img_name)
        mask_path = os.path.join(OUT_MASK, mask_name)

        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, msk)

        write_gps(img_path, lat, lon)

print("\n✅ PIPELINE COMPLETE — 8 forward-biased cameras ready for RealityScan")
