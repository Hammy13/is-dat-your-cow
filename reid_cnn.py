# reid_cnn.py
import os
import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# Config / transforms
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(embedding_size: int = 512, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 backbone with a small projection head producing L2-normalized embeddings.
    """
    backbone = models.resnet18(pretrained=pretrained)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Identity()

    head = nn.Sequential(
        nn.Linear(in_feat, embedding_size),
        nn.BatchNorm1d(embedding_size),
        nn.ReLU(inplace=True),
        nn.Linear(embedding_size, embedding_size),
    )

    class ReIDModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            x = nn.functional.normalize(x, p=2, dim=1)
            return x

    return ReIDModel(backbone, head)


@torch.no_grad()
def extract_embedding_from_tensor(model: nn.Module, img_t: torch.Tensor, device=None) -> np.ndarray:
    device = device or get_device()
    model = model.to(device).eval()
    img_t = img_t.unsqueeze(0).to(device)
    emb = model(img_t)  # (1, D)
    return emb.cpu().numpy().reshape(-1)


def pil_from_bgr_array(bgr: np.ndarray) -> Image.Image:
    # Convert BGR (OpenCV) -> RGB PIL image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class ReIDSystem:
    """
    Re-ID helper: hold model and gallery, provide embed / identify / gallery build utilities.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 embedding_size: int = 512,
                 pretrained: bool = True,
                 gallery_path: str = "gallery.npz"):
        self.device = get_device()
        self.model = model or build_model(embedding_size=embedding_size, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.gallery_path = gallery_path
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)
        if os.path.exists(self.gallery_path):
            self.load_gallery(self.gallery_path)

    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Accepts an OpenCV BGR crop (H,W,3) -> returns L2-normalized embedding (D,)"""
        img = pil_from_bgr_array(crop_bgr)
        t = _transform(img)
        return extract_embedding_from_tensor(self.model, t, device=self.device)

    def build_gallery_from_labeled_folder(self, labeled_root: str, avg_per_id: bool = True) -> None:
        """
        Expect folder structure:
          labeled_root/
            cow_001/
              img001.jpg
            cow_002/
              ...
        Produces internal gallery (and optionally calls save_gallery).
        """
        ids = []
        embs = []
        for id_dir in sorted(os.listdir(labeled_root)):
            full_dir = os.path.join(labeled_root, id_dir)
            if not os.path.isdir(full_dir):
                continue
            emb_list = []
            for img_path in sorted(glob.glob(os.path.join(full_dir, "*.*"))):
                try:
                    img = Image.open(img_path).convert("RGB")
                    t = _transform(img)
                    emb = extract_embedding_from_tensor(self.model, t, device=self.device)
                    emb_list.append(emb)
                except Exception:
                    continue
            if not emb_list:
                continue
            arr = np.stack(emb_list, axis=0)
            if avg_per_id:
                emb_vec = np.mean(arr, axis=0)
            else:
                emb_vec = np.mean(arr, axis=0)
            emb_vec = emb_vec / (np.linalg.norm(emb_vec) + 1e-8)
            ids.append(id_dir)
            embs.append(emb_vec.astype(np.float32))

        if not embs:
            raise RuntimeError("No embeddings computed from labeled folder.")
        self.ids = ids
        self.embeddings = np.stack(embs, axis=0)
        self.save_gallery(self.gallery_path)

    def save_gallery(self, out_npz: Optional[str] = None) -> None:
        path = out_npz or self.gallery_path
        if self.embeddings is None:
            # allow saving empty gallery
            np.savez_compressed(path, ids=np.array([], dtype='<U256'), embeddings=np.zeros((0, self.model.head[-1].out_features), dtype=np.float32))
            return
        ids_arr = np.array(self.ids, dtype='<U256')
        np.savez_compressed(path, ids=ids_arr, embeddings=self.embeddings.astype(np.float32))

    def load_gallery(self, npz_path: str) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.embeddings = data["embeddings"].astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: (D,), b: (N,D)
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(b, a)

    def identify(self, emb: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Return (best_id or None, score). If score < threshold returns (None, score).
        """
        if self.embeddings is None or len(self.ids) == 0:
            return None, 0.0
        sims = self._cosine_similarity(emb, self.embeddings)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < threshold:
            return None, best_score
        return self.ids[best_idx], best_score

    def add_or_update(self, cow_id: str, emb: np.ndarray, avg: bool = True) -> None:
        """Add new ID or update existing by averaging embeddings."""
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        if self.embeddings is None or len(self.ids) == 0:
            self.ids = [cow_id]
            self.embeddings = np.stack([emb.astype(np.float32)], axis=0)
            return
        if cow_id in self.ids:
            idx = self.ids.index(cow_id)
            if avg:
                self.embeddings[idx] = ((self.embeddings[idx] + emb) / 2.0).astype(np.float32)
            else:
                self.embeddings[idx] = emb.astype(np.float32)
        else:
            self.ids.append(cow_id)
            self.embeddings = np.vstack([self.embeddings, emb.astype(np.float32)])
