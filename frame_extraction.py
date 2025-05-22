import os
import cv2
from config import VIDEO_DIR, FRAME_DIR, FRAME_RATE
from utils import ensure_dir

def extract_frames():
    """
    Reads every video in VIDEO_DIR (sub-folders per camera or
    video files named camera_01.mp4, etc.), and writes frames
    at FRAME_RATE into FRAME_DIR/<camera>/
    """
    ensure_dir(FRAME_DIR)
    for vid in os.listdir(VIDEO_DIR):
        cam_name, _ = os.path.splitext(vid)
        out_dir = os.path.join(FRAME_DIR, cam_name)
        ensure_dir(out_dir)

        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, vid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / FRAME_RATE))
        count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                fname = f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame)
                frame_idx += 1
            count += 1

        cap.release()

if __name__ == "__main__":
    extract_frames()
