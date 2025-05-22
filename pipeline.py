import os
from PIL import Image
from config import FRAME_DIR, SNAPSHOT_DIR, TEXT_SIM_THRESHOLD, REID_SIM_THRESHOLD, MAX_TIME_GAP
from detection_embedding import DetectorEmbedder
from reid_extraction import ReIDExtractor
from utils import extract_timestamp_from_filename, cosine_similarity, ensure_dir

class PersonSearchPipeline:
    def __init__(self):
        ensure_dir(SNAPSHOT_DIR)
        self.detector = DetectorEmbedder()
        self.reid     = ReIDExtractor()

    def search_and_track(self, description, camera_dirs=None):
        """
        Search for a person by natural-language description across one or more
        camera frame folders, then stitch their path by Re-ID feature matching.

        Args:
            description (str): text query e.g. "a person wearing a blue jacket"
            camera_dirs (list of str, optional): list of folder paths containing frames.
                If None, uses FRAME_DIR/<subfolders>.

        Returns:
            dict: { person_id, confidence, path: [ {camera, frame, timestamp, bbox} ] }
        """
        # 1) Text embedding
        text_emb = self.detector.query_embedding(description)

        # 2) Determine camera paths
        if camera_dirs:
            cam_paths = camera_dirs
        else:
            cam_paths = [os.path.join(FRAME_DIR, cam)
                         for cam in sorted(os.listdir(FRAME_DIR))]

        # 3) Gather candidates above text threshold
        candidates = []
        for cam_path in cam_paths:
            cam = os.path.basename(cam_path)
            for fname in sorted(os.listdir(cam_path)):
                fpath = os.path.join(cam_path, fname)
                ts = extract_timestamp_from_filename(fname)
                embeds = self.detector.detect_and_embed(fpath)
                for e in embeds:
                    sim = cosine_similarity(text_emb, e["embedding"])
                    if sim >= TEXT_SIM_THRESHOLD:
                        candidates.append({
                            "camera": cam,
                            "camera_path": cam_path,
                            "frame": fname,
                            "frame_path": fpath,
                            "timestamp": ts,
                            "bbox": e["bbox"],
                            "score": sim
                        })

        if not candidates:
            return {"error": "No matching candidates found."}

        # 4) Select best initial match
        init = max(candidates, key=lambda x: x["score"])
        init_img = Image.open(init["frame_path"])
        crop     = init_img.crop(init["bbox"])
        init_reid = self.reid.extract(crop)

        # 5) Build chronological path
        path = [{
            "camera": init["camera"],
            "frame": init["frame"],
            "frame_path": init["frame_path"],
            "enter_ts": init["timestamp"],
            "exit_ts": init["timestamp"],
            "bbox": init["bbox"]
        }]
        last_ts = init["timestamp"]

        for c in sorted(candidates, key=lambda x: x["timestamp"]):
            dt = c["timestamp"] - last_ts
            if dt <= 0 or dt > MAX_TIME_GAP:
                continue
            img  = Image.open(c["frame_path"])
            crop = img.crop(c["bbox"])
            feat = self.reid.extract(crop)
            sim  = cosine_similarity(init_reid, feat)
            if sim >= REID_SIM_THRESHOLD:
                path.append({
                    "camera": c["camera"],
                    "frame": c["frame"],
                    "frame_path": c["frame_path"],
                    "enter_ts": c["timestamp"],
                    "exit_ts": c["timestamp"],
                    "bbox": c["bbox"]
                })
                last_ts = c["timestamp"]

        return {
            "person_id": "0001",
            "confidence": float(init["score"]),
            "path": path
        }

    def save_snapshots(self, result):
        """
        Crops and saves each sighting to SNAPSHOT_DIR,
        and appends 'snapshot' path into result.
        """
        for s in result.get("path", []):
            img      = Image.open(s["frame_path"])
            crop     = img.crop(s["bbox"])
            fname    = f"{result['person_id']}_{s['camera']}_{s['frame']}"
            out_path = os.path.join(SNAPSHOT_DIR, fname)
            crop.save(out_path)
            s["snapshot"] = out_path
        return result
