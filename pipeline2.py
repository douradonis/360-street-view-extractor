import os
import math
import cv2
import numpy as np
from PIL import Image
import piexif
from xml.dom import minidom
from streetview import search_panoramas, get_panorama
from ultralytics import YOLO
import signal
import time

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
# YAW offsets from center direction (forward = 0)
# Prioritizes forward-facing views: 0° is straight ahead, then left/right angles
YAW_OFFSETS = [0, -45, 45, -90, 90, -135, 135, 180]
PITCH_ROAD = 0  # Look forward/horizontal
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
    """
    Extract a flat rectilinear view from an equirectangular panorama.
    Simple, clean, straightforward projection - no warping.
    """
    h, w = img.shape[:2]
    
    # Convert to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    fov = np.deg2rad(fov_deg)
    
    # Create output coordinate grid
    y_idx, x_idx = np.mgrid[0:out_size, 0:out_size]
    
    # Normalize to [-1, 1]
    nx = (x_idx - out_size/2) / (out_size/2)
    ny = (y_idx - out_size/2) / (out_size/2)
    
    # Compute viewing angles
    half_fov = fov / 2
    angle_x = nx * half_fov
    angle_y = ny * half_fov
    
    # Create camera rays using spherical angles
    ray_x = np.sin(angle_x)
    ray_y = -np.sin(angle_y)
    ray_z = np.cos(angle_x) * np.cos(angle_y)
    
    # Apply pitch rotation (around X-axis)
    ray_y_rot = ray_y * np.cos(pitch) - ray_z * np.sin(pitch)
    ray_z_rot = ray_y * np.sin(pitch) + ray_z * np.cos(pitch)
    ray_y = ray_y_rot
    ray_z = ray_z_rot
    
    # Apply yaw rotation (around Z-axis)
    ray_x_rot = ray_x * np.cos(yaw) + ray_z * np.sin(yaw)
    ray_z_rot = -ray_x * np.sin(yaw) + ray_z * np.cos(yaw)
    ray_x = ray_x_rot
    ray_z = ray_z_rot
    
    # Convert to equirectangular coordinates
    lon = np.arctan2(ray_x, ray_z)
    lat = np.arcsin(np.clip(ray_y, -0.9999, 0.9999))
    
    # Map to panorama texture coordinates
    pano_x = (lon / np.pi + 1) / 2 * (w - 1)
    pano_y = (0.5 - lat / np.pi) * (h - 1)
    
    # Remap using linear interpolation
    result = cv2.remap(img, pano_x.astype(np.float32), pano_y.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return result

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
model = None

def get_model():
    global model
    if model is None:
        model = YOLO("yolov8n-seg.pt")
    return model


# =========================
# MAIN
# =========================
def generate_points(raw_coords, step_meters=STEP_METERS):
    """Return combined list of points with type 'kml' or 'interp'."""
    interp = interpolate_path(raw_coords, step_meters)
    raw_set = set((round(r[0],6), round(r[1],6)) for r in raw_coords)
    points = []
    for p in interp:
        tag = 'kml' if (round(p[0],6), round(p[1],6)) in raw_set else 'interp'
        points.append({'lat': p[0], 'lon': p[1], 'type': tag})
    return points


def write_status(status_path, data):
    tmp = status_path + '.tmp'
    with open(tmp, 'w') as f:
        import json
        json.dump(data, f)
    os.replace(tmp, status_path)


def read_status(status_path):
    try:
        import json
        with open(status_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def add_processed_pano(pano_id, lat, lon, pano_index, status='done'):
    """Append processed pano metadata to the STATUS_FILE under 'processed' array."""
    s = read_status(STATUS_FILE) if os.path.exists(STATUS_FILE) else {}
    processed = s.get('processed', [])
    processed.append({'pano_id': pano_id, 'lat': lat, 'lon': lon, 'pano_index': pano_index, 'status': status, 'ts': time.time()})
    s['processed'] = processed
    # also update pano_index count
    s['pano_index'] = pano_index + 1
    write_status(STATUS_FILE, s)

STATUS_FILE = 'status.json'


def process_point(point, idx, raw_coords, out_img=OUT_IMG, out_mask=OUT_MASK):
    lat, lon, ptype = point['lat'], point['lon'], point['type']
    write_status(STATUS_FILE, {'status': 'searching', 'index': idx, 'lat': lat, 'lon': lon, 'type': ptype})

    panos = search_panoramas(lat, lon)
    if not panos:
        write_status(STATUS_FILE, {'status': 'no_pano', 'index': idx, 'lat': lat, 'lon': lon, 'type': ptype})
        return False

    pano = panos[0]
    img = get_panorama(pano.pano_id)
    pano_img = np.array(img)
    pano_img = cv2.cvtColor(pano_img, cv2.COLOR_RGB2BGR)
    h, w = pano_img.shape[:2]

    # Mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    results = get_model()(pano_img, imgsz=1024, conf=0.3)
    for r in results:
        if r.masks is None:
            continue
        for cls, seg in zip(r.boxes.cls, r.masks.data):
            if get_model().names[int(cls)] in MASK_CLASSES:
                seg = cv2.resize((seg.cpu().numpy()*255).astype(np.uint8), (w,h))
                full_mask |= seg
    # Do NOT force mask the sky here — only remove detected object pixels in view-space

    # Bearing
    center_yaw = 0
    for j in range(len(raw_coords)-1):
        if abs(raw_coords[j][0] - lat) < 0.01 and abs(raw_coords[j][1] - lon) < 0.01:
            center_yaw = bearing(lat, lon, *raw_coords[j+1])
            break
    if center_yaw == 0:
        # fallback to nothing or leave zero
        pass

    base = f"pano_{idx:04d}"
    # Respect UI per-cam keep_objects settings
    def read_ui_config(config_path='ui_config.json'):
        try:
            import json
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            return cfg
        except Exception:
            return {}

    cfg = read_ui_config()
    keep_objects = set(cfg.get('keep_objects_cams', []))

    for j, off in enumerate(YAW_OFFSETS):
        yaw = (center_yaw + off) % 360
        view = extract_view_spherical(pano_img, yaw, PITCH_ROAD, FOV, OUT_SIZE)
        mask = extract_view_spherical(full_mask, yaw, PITCH_ROAD, FOV, OUT_SIZE)
        # If this cam is not marked to keep objects, zero only the object pixels in view-space
        if j not in keep_objects:
            view[mask > 0] = 0
        img_path = f"{out_img}/{base}_cam{j}.jpg"
        mask_path = f"{out_mask}/{base}_cam{j}_mask.png"
        cv2.imwrite(img_path, view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(mask_path, mask)
        # write GPS for consistency
        try:
            write_gps(img_path, pano.lat, pano.lon)
        except Exception:
            pass

    write_status(STATUS_FILE, {'status': 'done', 'index': idx, 'lat': lat, 'lon': lon, 'type': ptype})
    return True


import threading

def process_pano(pano, pano_index, center_yaw, raw_coords, out_img=OUT_IMG, out_mask=OUT_MASK):
    """Download a pano, mask it and save 8 forward-biased views."""
    write_status(STATUS_FILE, {'status': 'downloading', 'pano_id': pano.pano_id, 'pano_index': pano_index})
    try:
        img = get_panorama(pano.pano_id)
        pano_img = np.array(img)
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_RGB2BGR)
        h, w = pano_img.shape[:2]
    except Exception as e:
        write_status(STATUS_FILE, {'status': 'download_error', 'error': str(e), 'pano_id': pano.pano_id})
        try:
            add_processed_pano(pano.pano_id, getattr(pano, 'lat', None), getattr(pano, 'lon', None), pano_index, status='failed')
        except Exception:
            pass
        return False

    # Mask (detect objects to remove) — do NOT mask sky; only detected object masks are used
    full_mask = np.zeros((h, w), dtype=np.uint8)
    results = get_model()(pano_img, imgsz=1024, conf=0.3)
    for r in results:
        if r.masks is None:
            continue
        for cls, seg in zip(r.boxes.cls, r.masks.data):
            if get_model().names[int(cls)] in MASK_CLASSES:
                seg = cv2.resize((seg.cpu().numpy()*255).astype(np.uint8), (w,h))
                full_mask |= seg

    # Prepare masked and original versions (masked_pano zeros only the detected objects)
    original_pano = pano_img.copy()
    masked_pano = pano_img.copy()
    masked_pano[full_mask > 0] = 0

    # Read UI config to see which cams should keep objects (no unmask)
    def read_ui_config(config_path='ui_config.json'):
        try:
            import json
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            return cfg
        except Exception:
            return {}

    cfg = read_ui_config()
    keep_objects = set(cfg.get('keep_objects_cams', []))

    base = f"pano_{pano_index:04d}"
    for j, off in enumerate(YAW_OFFSETS):
        yaw = (center_yaw + off) % 360
        # Extract original view and remap the pano-space full_mask into view-space
        view = extract_view_spherical(original_pano, yaw, PITCH_ROAD, FOV, OUT_SIZE)
        mask = extract_view_spherical(full_mask, yaw, PITCH_ROAD, FOV, OUT_SIZE)
        # If this cam is not marked to keep objects, zero only the object pixels in view-space
        if j not in keep_objects:
            view[mask > 0] = 0
        img_path = f"{out_img}/{base}_cam{j}.jpg"
        mask_path = f"{out_mask}/{base}_cam{j}_mask.png"
        cv2.imwrite(img_path, view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(mask_path, mask)
        write_gps(img_path, pano.lat, pano.lon)

    write_status(STATUS_FILE, {'status': 'pano_done', 'pano_id': pano.pano_id, 'pano_index': pano_index})
    try:
        add_processed_pano(pano.pano_id, pano.lat, pano.lon, pano_index, status='done')
    except Exception:
        pass
    return True


def process_route(raw_coords, step_meters=STEP_METERS, start_idx=0):
    # iterate interpolated points and process unique panos (deduplicated)
    interp = interpolate_path(raw_coords, step_meters)
    total = len(interp)
    write_status(STATUS_FILE, {'status': 'running', 'total_interp_points': total, 'pano_index': 0})

    seen_panos = set()
    pano_index = 0

    for i in range(start_idx, len(interp)):
        cmd = read_status(STATUS_FILE).get('cmd')
        if cmd == 'stop':
            write_status(STATUS_FILE, {'status': 'stopped', 'interp_index': i, 'pano_index': pano_index})
            break

        lat, lon = interp[i]
        write_status(STATUS_FILE, {'status': 'searching', 'interp_index': i, 'lat': lat, 'lon': lon})
        try:
            panos = search_panoramas(lat, lon)
        except Exception as e:
            write_status(STATUS_FILE, {'status': 'search_error', 'error': str(e), 'interp_index': i})
            continue

        if not panos:
            write_status(STATUS_FILE, {'status': 'no_pano', 'interp_index': i, 'lat': lat, 'lon': lon})
            continue

        # For each pano returned at this location, process if we haven't yet
        for pano in panos:
            if pano.pano_id in seen_panos:
                continue
            seen_panos.add(pano.pano_id)

            # Determine center yaw: bearing to next interp point if exists
            if i < len(interp)-1:
                next_lat, next_lon = interp[i+1]
                center_yaw = bearing(lat, lon, next_lat, next_lon)
            else:
                center_yaw = 0

            write_status(STATUS_FILE, {'status': 'processing_pano', 'pano_id': pano.pano_id, 'interp_index': i, 'pano_index': pano_index})
            success = process_pano(pano, pano_index, center_yaw, raw_coords)
            if success:
                pano_index += 1

    write_status(STATUS_FILE, {'status': 'finished', 'pano_index': pano_index})


def main_process(kml=INPUT_KML):
    raw_coords = parse_kml_points(kml)
    process_route(raw_coords)


if __name__ == '__main__':
    main_process()
