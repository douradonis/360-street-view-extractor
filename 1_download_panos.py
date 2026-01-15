import os
import math
from lxml import etree
from geopy.distance import geodesic
from streetview import search_panoramas, get_panorama



# ==============================
# CONFIG (PHOTOGRAMMETRY READY)
# ==============================

KML_PATH = "kml_route.kml"
OUTPUT_DIR = "output/panos"
SAMPLE_EVERY_METERS = 8        # πιο πυκνό = καλύτερο overlap
SEARCH_RADIUS = 20             # meters
MAX_ZOOM = 5                   # 5 ή 6 (δοκίμασε 6)
MIN_YEAR = 2019                # κόβει πολύ παλιά panos

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# KML PARSER
# ==============================

def parse_kml_linestring(kml_path):
    with open(kml_path, "rb") as f:
        tree = etree.parse(f)

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords = tree.xpath("//kml:LineString/kml:coordinates", namespaces=ns)

    points = []
    for c in coords:
        for line in c.text.strip().split():
            lon, lat, *_ = map(float, line.split(","))
            points.append((lat, lon))

    return points

# ==============================
# ROUTE SAMPLING
# ==============================

def sample_route(points, step_m):
    sampled = [points[0]]
    acc = 0.0

    for i in range(1, len(points)):
        d = geodesic(points[i-1], points[i]).meters
        acc += d
        if acc >= step_m:
            sampled.append(points[i])
            acc = 0.0

    return sampled

# ==============================
# PANORAMA SELECTION
# ==============================

def pano_year(pano):
    if not pano.date:
        return 0
    return int(pano.date.split("-")[0])

def get_latest_valid_pano(lat, lon):
    panos = search_panoramas(lat, lon)
    if not panos:
        return None

    panos = [p for p in panos if pano_year(p) >= MIN_YEAR]

    if not panos:
        return None

    panos.sort(key=lambda p: p.date, reverse=True)
    return panos[0]

# ==============================
# DOWNLOAD
# ==============================

def download_pano(pano):
    out_path = os.path.join(OUTPUT_DIR, pano.pano_id + ".jpg")

    if os.path.exists(out_path):
        return

    print(f"📸 {pano.pano_id} | {pano.date}")

    image = get_panorama(
        pano_id=pano.pano_id,
        zoom=MAX_ZOOM
    )

    image.save(out_path, quality=95)

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    route = parse_kml_linestring(KML_PATH)
    sampled = sample_route(route, SAMPLE_EVERY_METERS)

    seen = set()

    for lat, lon in sampled:
        pano = get_latest_valid_pano(lat, lon)
        if not pano:
            continue

        if pano.pano_id in seen:
            continue

        seen.add(pano.pano_id)
        download_pano(pano)

    print("\n✅ DONE – Panos ready for photogrammetry")

if __name__ == "__main__":
    main()
