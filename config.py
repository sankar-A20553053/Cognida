# config.py

import os

# Frame extraction
FRAME_RATE = 1  # frames per second

# Directories
BASE_DIR     = os.getcwd()
DATA_DIR     = os.path.join(BASE_DIR, "data")
VIDEO_DIR    = os.path.join(DATA_DIR, "videos")
FRAME_DIR    = os.path.join(DATA_DIR, "frames")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
MODEL_DIR    = os.path.join(BASE_DIR, "models")

# CLIP
CLIP_MODEL_NAME    = "ViT-B/32"
TEXT_SIM_THRESHOLD = 0.25

# Re-ID
REID_WEIGHTS      = os.path.join(MODEL_DIR, "osnet_x0_25_msmt17.pth")
REID_SIM_THRESHOLD = 0.45

# Tracking
MAX_TIME_GAP = 10  # seconds

# Camera layout for visualization
CAMERA_COORDS = {
    "c1s1": (0, 0),
    "c1s4": (1, 0),
    "c1s5": (2, 0),
    "c1s6": (3, 0),
    "c3s2": (0, 1),
    "c5s1": (1, 1),
    "c5s2": (2, 1),
}
