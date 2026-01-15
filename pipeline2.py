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

OUT_SIZE = 4096          # High-res output
FOV = 90                  # Perspective FOV
YAW_OFFSETS = [-70, -50, -30, -10, 10, 30, 50, 70]
PITCH_ROAD = -10          # Slightly downward to focus on road
MASK_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
DEFAULT_ALT = 2.5
STEP_METERS = 5           # Distance between interpolated points

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
    θ = math.atan2(x, y)
    return math.degrees(θ) % 360

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
    x_cam = xx
    y_cam = -yy
    vec = np.stack([x_cam, y_cam, z], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    R = Ry @ Rx

    vec_rot = vec @ R.T
    theta = np.arctan2(vec_rot[...,0], vec_rot[...,2])
    phi = np.arcsin(vec_rot[...,1])

    phi = np.clip(phi, -np.pi/2 + 0.05, np.pi/2 - 0.05)
    u = (theta/np.pi + 1) * w/2
    v = (0.5 - phi/np.pi) * h

    persp = cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp

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

def interpolate_path(coords, step_meters=5):
    points = []
    for i in range(len(coords)-1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]

        # Haversine distance
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

# =========================
# INIT YOLO
# =========================
model = YOLO("yolov8n-seg.pt")

# =========================
# MAIN PIPELINE
# =========================
coords = parse_kml_points(INPUT_KML)
interp_coords = interpolate_path(coords, STEP_METERS)
seen_panos = set()
pano_index = 0

for i, (lat, lon) in enumerate(interp_coords):
    print(f"📍 Processing interpolated point {i+1}/{len(interp_coords)}: {lat},{lon}")

    try:
        panos = search_panoramas(lat, lon)  # No radius, only lat/lon
        if not panos:
            continue
    except Exception as e:
        print("⚠️ Street View search failed:", e)
        continue

    for pano in panos:
        if pano.pano_id in seen_panos:
            continue
        seen_panos.add(pano.pano_id)

        try:
            img = get_panorama(pano.pano_id)
            pano_img = np.array(img)
            h, w = pano_img.shape[:2]
        except Exception as e:
            print("⚠️ Failed to download pano:", e)
            continue

        # ---------- MASKING ----------
        full_mask = np.zeros((h, w), dtype=np.uint8)
        results = model(pano_img, imgsz=1024, conf=0.3)
        for r in results:
            if r.masks is None:
                continue
            for cls, seg in zip(r.boxes.cls, r.masks.data):
                label = model.names[int(cls)]
                if label in MASK_CLASSES:
                    seg = seg.cpu().numpy().astype(np.uint8)*255
                    seg = cv2.resize(seg,(w,h))
                    full_mask = cv2.bitwise_or(full_mask, seg)
        full_mask[:int(h*0.25), :] = 255  # Sky mask
        pano_img[full_mask>0] = 0

        # ---------- CALCULATE FORWARD YAW ----------
        if i < len(interp_coords)-1:
            next_lat, next_lon = interp_coords[i+1]
            center_yaw = bearing(lat, lon, next_lat, next_lon)
        else:
            center_yaw = 0

        base = f"pano_{pano_index:03d}"
        pano_index += 1

        # ---------- SPLIT 8 FORWARD-CAMERAS ----------
        for j, offset in enumerate(YAW_OFFSETS):
            yaw = (center_yaw + offset) % 360
            view = extract_view_spherical(pano_img, yaw_deg=yaw, pitch_deg=PITCH_ROAD, fov_deg=FOV, out_size=OUT_SIZE)
            mask_view = extract_view_spherical(full_mask, yaw_deg=yaw, pitch_deg=PITCH_ROAD, fov_deg=FOV, out_size=OUT_SIZE)

            img_name = f"{base}_cam{j}.jpg"
            mask_name = f"{base}_cam{j}_mask.png"

            img_path = os.path.join(OUT_IMG, img_name)
            mask_path = os.path.join(OUT_MASK, mask_name)

            cv2.imwrite(img_path, view)
            cv2.imwrite(mask_path, mask_view)
            write_gps(img_path, pano.lat, pano.lon)

print("\n✅ PIPELINE COMPLETE — All available panos processed with 8 forward-biased high-res cameras")
