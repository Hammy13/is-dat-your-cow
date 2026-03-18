from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return int(x1), int(y1), int(x2), int(y2)


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    return image[y1:y2, x1:x2].copy()


def compute_side_view_score(box: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> float:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    aspect_ratio = box_width / box_height
    area_ratio = (box_width * box_height) / float(frame_shape[0] * frame_shape[1])
    aspect_component = min(aspect_ratio / 1.5, 1.0)
    area_component = min(area_ratio / 0.10, 1.0)
    return float(0.65 * aspect_component + 0.35 * area_component)


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_quality_gate(
    box: tuple[int, int, int, int],
    crop_bgr: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_aspect_ratio: float,
    min_area_ratio: float,
    min_sharpness: float,
) -> tuple[bool, dict[str, float]]:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    aspect_ratio = box_width / box_height
    area_ratio = (box_width * box_height) / float(frame_shape[0] * frame_shape[1])
    sharpness = compute_sharpness(crop_bgr)
    side_view_score = compute_side_view_score(box, frame_shape)
    passed = (
        aspect_ratio >= min_aspect_ratio
        and area_ratio >= min_area_ratio
        and sharpness >= min_sharpness
    )
    return passed, {
        "aspect_ratio": float(aspect_ratio),
        "area_ratio": float(area_ratio),
        "sharpness": float(sharpness),
        "side_view_score": float(side_view_score),
    }
