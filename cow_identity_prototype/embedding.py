from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
from torchvision import models, transforms

from .config import EmbeddingConfig


class DeepEmbeddingExtractor:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._build_model(config.backbone, config.use_pretrained).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((config.input_size, config.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, backbone: str, use_pretrained: bool):
        try:
            if backbone == "vit_b_16":
                weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
                model = models.vit_b_16(weights=weights)
                model.heads = torch.nn.Identity()
                return model
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            return model
        except Exception as exc:
            warnings.warn(f"Falling back to randomly initialised {backbone}: {exc}")
            if backbone == "vit_b_16":
                model = models.vit_b_16(weights=None)
                model.heads = torch.nn.Identity()
                return model
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Identity()
            return model

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor).flatten().cpu().numpy().astype(np.float32)
        features /= max(np.linalg.norm(features), 1e-6)
        return features
