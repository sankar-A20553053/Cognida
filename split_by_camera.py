# split_by_camera.py

import os, shutil

# 1️⃣ adjust these if you moved things around
TRAIN_DIR = "data/Market1501/bounding_box_train"
OUT_ROOT = "data/Market1501/cams"

os.makedirs(OUT_ROOT, exist_ok=True)

for fn in os.listdir(TRAIN_DIR):
    if not fn.lower().endswith(".jpg"):
        print(" skipping (not a jpg):", fn)
        continue

    parts = fn.split("_")
    if len(parts) < 2 or not parts[1].startswith("c"):
        print(" skipping (unexpected format):", fn)
        continue

    cam = parts[1]  # e.g. 'c3'
    dest_dir = os.path.join(OUT_ROOT, cam)
    os.makedirs(dest_dir, exist_ok=True)

    src_path = os.path.join(TRAIN_DIR, fn)
    dst_path = os.path.join(dest_dir, fn)
    shutil.copy(src_path, dst_path)

print("Done! Cameras found:", sorted(os.listdir(OUT_ROOT)))
