import os
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_timestamp_from_filename(filename):
    """
    Assumes filenames like frame_0001.jpg → returns 1 (second).
    Adjust this if your naming scheme differs.
    """
    num = int(os.path.splitext(filename)[0].split('_')[-1])
    return num

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
