import torch
import clip
from PIL import Image
from ultralytics import YOLO
from config import CLIP_MODEL_NAME, TEXT_SIM_THRESHOLD
from utils import cosine_similarity

class DetectorEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # person detector (YOLOv8 nanos model)
        self.detector = YOLO("yolov8n.pt")
        # CLIP for cropping + text
        self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
        self.model.eval()

    def query_embedding(self, text: str):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            t_emb = self.model.encode_text(tokens)
            t_emb /= t_emb.norm(dim=-1, keepdim=True)
        return t_emb.cpu().numpy()[0]

    def detect_and_embed(self, img_path: str):
        """
        Returns list of dicts {bbox: (x1,y1,x2,y2), embedding: np.array}
        only for class=person (cls==0).
        """
        img = Image.open(img_path).convert("RGB")
        results = self.detector(img_path, conf=0.20)  # returns list of Results, one per inference
        crops = []
        for res in results:
            boxes = res.boxes
            if boxes is None: continue
            for box in boxes:
                cls = int(box.cls.cpu().numpy())
                if cls != 0:  # filter only 'person' class
                    continue
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
                crop_img = img.crop((x1, y1, x2, y2))
                inp = self.preprocess(crop_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    i_emb = self.model.encode_image(inp)
                    i_emb /= i_emb.norm(dim=-1, keepdim=True)
                crops.append({
                    "bbox": (x1, y1, x2, y2),
                    "embedding": i_emb.cpu().numpy()[0]
                })
        return crops
