import os
import math
import cv2
import numpy as np
from PIL import Image
import piexif
from xml.dom import minidom
from streetview import search_panoramas, get_panorama
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
INPUT_KML = "route.kml"
OUT_IMG = "output/images"
OUT_MASK = "output/masks"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

OUT_SIZE = 4096
FOV = 90
YAW_OFFSETS = [-70, -50, -30, -10, 10, 30, 50, 70]
PITCH_ROAD = -10
MASK_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
DEFAULT_ALT = 2.5
STEP_METERS = 5

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
    exif_bytes = img.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "Interop":{}, "1st":{}, "thumbnail":None}
    exif_dict["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >=0 else "S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >=0 else "W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lon)),
        piexif.GPSIFD.GPSAltitude: (int(alt*100),100),
    }
    piexif.insert(piexif.dump(exif_dict), path)

def bearing(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    return math.degrees(math.atan2(x, y)) % 360

# ✅ Η ΣΥΝΑΡΤΗΣΗ ΣΟΥ – ΟΠΩΣ ΤΗΝ ΕΔΩΣΕΣ
def interpolate_path(coords, step_meters=5):
    points = []
    for i in range(len(coords)-1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]

        R = 6371000
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist = R * c

        steps = max(1, int(dist/step_meters))
        for s in range(steps):
            f = s/steps
            lat = lat1 + f*(lat2-lat1)
            lon = lon1 + f*(lon2-lon1)
            points.append((lat, lon))
    points.append(coords[-1])
    return points

def extract_view_spherical(img, yaw_deg, pitch_deg=-10, fov_deg=90, out_size=4096):
    h, w = img.shape[:2]
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    f = (out_size/2) / np.tan(fov/2)
    xs = np.linspace(-out_size/2, out_size/2-1, out_size)
    ys = np.linspace(-out_size/2, out_size/2-1, out_size)
    xx, yy = np.meshgrid(xs, ys)

    z = f * np.ones_like(xx)
    x = xx
    y = -yy
    vec = np.stack([x, y, z], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    vec = vec @ (Ry @ Rx).T

    theta = np.arctan2(vec[...,0], vec[...,2])
    phi = np.clip(np.arcsin(vec[...,1]), -np.pi/2+0.05, np.pi/2-0.05)

    u = (theta/np.pi + 1) * w/2
    v = (0.5 - phi/np.pi) * h

    return cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

def parse_kml_points(kml_file):
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
# MAIN
# =========================
raw_coords = parse_kml_points(INPUT_KML)
coords = interpolate_path(raw_coords, STEP_METERS)

print(f"🚗 Interpolated route: {len(raw_coords)} → {len(coords)} points")

for i, (lat, lon) in enumerate(coords):
    print(f"📍 {i+1}/{len(coords)}")

    panos = search_panoramas(lat, lon)
    if not panos:
        continue
    pano = panos[0]

    img = get_panorama(pano.pano_id)
    pano_img = np.array(img)
    h, w = pano_img.shape[:2]

    # ---------- MASK ----------
    full_mask = np.zeros((h, w), dtype=np.uint8)
    results = model(pano_img, imgsz=1024, conf=0.3)
    for r in results:
        if r.masks is None:
            continue
        for cls, seg in zip(r.boxes.cls, r.masks.data):
            if model.names[int(cls)] in MASK_CLASSES:
                seg = cv2.resize((seg.cpu().numpy()*255).astype(np.uint8), (w,h))
                full_mask |= seg

    full_mask[:int(h*0.25), :] = 255
    pano_img[full_mask > 0] = 0

    if i < len(coords)-1:
        center_yaw = bearing(lat, lon, *coords[i+1])
    else:
        center_yaw = 0

    base = f"pano_{i:04d}"

    for j, off in enumerate(YAW_OFFSETS):
        yaw = (center_yaw + off) % 360
        view = extract_view_spherical(pano_img, yaw, PITCH_ROAD, FOV, OUT_SIZE)
        mask = extract_view_spherical(full_mask, yaw, PITCH_ROAD, FOV, OUT_SIZE)

        img_path = f"{OUT_IMG}/{base}_cam{j}.jpg"
        mask_path = f"{OUT_MASK}/{base}_cam{j}_mask.png"

        # ✅ ΔΙΟΡΘΩΣΗ ΧΡΩΜΑΤΩΝ
        cv2.imwrite(img_path, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)

        write_gps(img_path, lat, lon)

print("\n✅ DONE — Correct colors, 5m steps, ready for RealityScan")
