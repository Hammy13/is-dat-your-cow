from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


@dataclass
class GallerySample:
    cow_id: str
    image_path: Path


def list_gallery_samples(root: str | Path) -> list[GallerySample]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    samples: list[GallerySample] = []
    for cow_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for image_path in sorted(p for p in cow_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            samples.append(GallerySample(cow_id=cow_dir.name, image_path=image_path))
    return samples


def list_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [root_path]
    return sorted(path for path in root_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def list_videos(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    if root_path.is_file() and root_path.suffix.lower() in VIDEO_EXTENSIONS:
        return [root_path]
    return sorted(path for path in root_path.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
