# split_by_camera.py
import os, shutil

src = "data/Market1501/bounding_box_train"
dst_root = "data/Market1501/cams"
os.makedirs(dst_root, exist_ok=True)

for fn in os.listdir(src):
    cam = fn.split("_")[1]  # e.g. "c3"
    out = os.path.join(dst_root, cam)
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(src, fn), os.path.join(out, fn))

print("Done splitting cameras:", os.listdir(dst_root))
