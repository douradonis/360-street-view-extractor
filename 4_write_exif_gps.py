# 4_write_exif_gps.py
import os
import piexif
from PIL import Image

INPUT_DIR = "views/images"

# ⚠️ ΒΑΖΟΥΜΕ ΤΑ GPS ΑΠΟ ΤΟ PANO (ή KML)
# Αν έχεις διαφορετικά ανά εικόνα, μπορείς να τα φορτώνεις από json
LATITUDE = 37.9838
LONGITUDE = 23.7275
ALTITUDE = 2.5  # meters

def deg_to_dms_rational(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return [(d, 1), (m, 1), (int(s * 100), 100)]

def write_gps(img_path):
    img = Image.open(img_path)
    exif_dict = piexif.load(img.info.get("exif", b""))

    lat_ref = "N" if LATITUDE >= 0 else "S"
    lon_ref = "E" if LONGITUDE >= 0 else "W"

    exif_dict["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(LATITUDE)),
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(LONGITUDE)),
        piexif.GPSIFD.GPSAltitude: (int(ALTITUDE * 100), 100),
    }

    exif_bytes = piexif.dump(exif_dict)
    img.save(img_path, exif=exif_bytes)

for f in os.listdir(INPUT_DIR):
    if f.endswith(".jpg"):
        write_gps(os.path.join(INPUT_DIR, f))
