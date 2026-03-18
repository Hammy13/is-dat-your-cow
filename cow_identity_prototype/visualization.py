from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import PrototypeConfig
from .types import Detection, MatchDecision


def draw_grid_on_crop(crop_bgr: np.ndarray, grid_size: int) -> np.ndarray:
    output = crop_bgr.copy()
    height, width = output.shape[:2]
    for row in range(1, grid_size):
        y = int(row * height / grid_size)
        cv2.line(output, (0, y), (width, y), (255, 180, 0), 1)
    for col in range(1, grid_size):
        x = int(col * width / grid_size)
        cv2.line(output, (x, 0), (x, height), (255, 180, 0), 1)
    return output


def draw_detection(frame: np.ndarray, detection: Detection, decisions: dict[str, MatchDecision]) -> np.ndarray:
    output = frame.copy()
    x1, y1, x2, y2 = detection.box
    cv2.rectangle(output, (x1, y1), (x2, y2), (50, 220, 90), 2)
    hybrid = decisions["hybrid"]
    status = ""
    if hybrid.is_new_identity:
        status = " NEW"
    elif hybrid.metadata.get("rescue_reason"):
        status = " RESCUED"
    label = f"{hybrid.cow_id}{status} | H={hybrid.score:.2f}"
    cv2.putText(output, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(output, f"Y={decisions['yolo11'].score:.2f} C={decisions['cnn'].score:.2f}", (x1, min(output.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
    return output


def draw_mesh_overlay(frame: np.ndarray, detection: Detection, grid_size: int) -> np.ndarray:
    output = frame.copy()
    x1, y1, x2, y2 = detection.box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    for row in range(1, grid_size):
        y = y1 + int(row * height / grid_size)
        cv2.line(output, (x1, y), (x2, y), (255, 180, 0), 1)
    for col in range(1, grid_size):
        x = x1 + int(col * width / grid_size)
        cv2.line(output, (x, y1), (x, y2), (255, 180, 0), 1)
    return output


def save_fingerprint_panel(
    config: PrototypeConfig,
    image_name: str,
    crop_bgr: np.ndarray,
    dark_map: np.ndarray,
    light_map: np.ndarray,
    texture_map: np.ndarray,
) -> Path:
    output_path = Path(config.paths.output_root) / "fingerprints" / f"{Path(image_name).stem}_fingerprint.png"
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Crop")
    axes[1].imshow(dark_map, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Dark Pixel %")
    axes[2].imshow(light_map, cmap="binary", vmin=0, vmax=1)
    axes[2].set_title("Light Pixel %")
    axes[3].imshow(texture_map, cmap="magma")
    axes[3].set_title("Texture")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_match_comparison_panel(
    config: PrototypeConfig,
    panel_name: str,
    query_crop_bgr: np.ndarray,
    matched_gallery_images: list[str],
    decisions: dict[str, MatchDecision],
    grid_size: int,
) -> Path:
    output_path = Path(config.paths.output_root) / "fingerprints" / f"{Path(panel_name).stem}_comparison.png"
    top_gallery = matched_gallery_images[:3]
    total_columns = 1 + max(1, len(top_gallery))
    figure, axes = plt.subplots(1, total_columns, figsize=(4 * total_columns, 4))
    if total_columns == 1:
        axes = [axes]
    query_with_grid = draw_grid_on_crop(query_crop_bgr, grid_size)
    axes[0].imshow(cv2.cvtColor(query_with_grid, cv2.COLOR_BGR2RGB))
    hybrid = decisions["hybrid"]
    axes[0].set_title(f"Query\n{hybrid.cow_id} | {hybrid.score:.2f}")
    axes[0].axis("off")
    for idx, gallery_path in enumerate(top_gallery, start=1):
        gallery_image = cv2.imread(gallery_path)
        if gallery_image is None:
            axes[idx].axis("off")
            continue
        axes[idx].imshow(cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Gallery {idx}\n{Path(gallery_path).stem}")
        axes[idx].axis("off")
    for idx in range(1 + len(top_gallery), total_columns):
        axes[idx].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def create_video_writer(output_path: str | Path, fps: float, frame_size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
