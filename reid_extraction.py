import torch
import numpy as np
from torchvision import transforms
from torchreid.utils.torchtools import load_pretrained_weights
from torchreid.models import build_model
from config import REID_WEIGHTS
from PIL import Image

class ReIDExtractor:
    def __init__(self, model_name="osnet_x0_25", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            name=model_name,
            num_classes=1000,
            loss="softmax"
        )
        load_pretrained_weights(self.model, REID_WEIGHTS)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def extract(self, crop):
        """
        crop: PIL.Image or np.ndarray
        returns: 1D feature vector (unit-normalized)
        """
        if not isinstance(crop, Image.Image):
            crop = Image.fromarray(crop)
        x = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
