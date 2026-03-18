from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str
    crop_bgr: np.ndarray
    track_id: int | None = None
    side_view_score: float = 0.0
    sharpness: float = 0.0


@dataclass
class MatchDecision:
    model_name: str
    cow_id: str
    score: float
    is_new_identity: bool
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GalleryEntry:
    cow_id: str
    yolo_mesh_centroid: np.ndarray
    cnn_centroid: np.ndarray
    hybrid_centroid: np.ndarray
    observation_count: int
    source_images: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
