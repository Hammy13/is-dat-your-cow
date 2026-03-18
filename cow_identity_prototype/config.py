from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"
DEFAULT_ARTIFACT_ROOT = PACKAGE_ROOT / "artifacts"
DEFAULT_GALLERY_ROOT = PACKAGE_ROOT / "gallery"
DEFAULT_MODELS_ROOT = PACKAGE_ROOT / "models"
DEFAULT_DOCS_ROOT = PACKAGE_ROOT / "docs"
DEFAULT_SRC_ROOT = PACKAGE_ROOT / "src"


@dataclass
class DetectorConfig:
    model_path: str = "yolo11n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 960
    tracker: str = "bytetrack.yaml"
    classes_hint: tuple[str, ...] = ("cow", "cattle", "bull", "calf")


@dataclass
class MeshConfig:
    grid_size: int = 16
    histogram_bins: int = 8
    dark_threshold: int = 75
    light_threshold: int = 180


@dataclass
class EmbeddingConfig:
    backbone: str = "resnet50"
    input_size: int = 224
    use_pretrained: bool = True
    device: str = "cpu"


@dataclass
class MatchingConfig:
    yolo_mesh_threshold: float = 0.80
    cnn_threshold: float = 0.84
    hybrid_threshold: float = 0.83
    hybrid_rescue_margin: float = 0.06
    ensemble_threshold: float = 0.76
    training_priority_margin: float = 0.03
    hybrid_mesh_weight: float = 0.45
    hybrid_deep_weight: float = 0.55
    allow_new_identity: bool = True
    min_track_hits: int = 3


@dataclass
class InferenceConfig:
    sample_every_n_frames: int = 2
    side_view_min_aspect_ratio: float = 1.0
    min_crop_area_ratio: float = 0.01
    min_sharpness: float = 20.0
    save_debug_panels: bool = True


@dataclass
class TrainingConfig:
    auto_cluster_eps: float = 0.18
    auto_cluster_min_samples: int = 1
    save_crop_images: bool = True


@dataclass
class PathsConfig:
    data_root: str = str(DEFAULT_DATA_ROOT)
    train_uploads_dir: str = str(DEFAULT_DATA_ROOT / "train_uploads")
    test_image_dir: str = str(DEFAULT_DATA_ROOT / "test_uploads" / "images")
    test_video_dir: str = str(DEFAULT_DATA_ROOT / "test_uploads" / "videos")
    artifact_root: str = str(DEFAULT_ARTIFACT_ROOT)
    gallery_root: str = str(DEFAULT_GALLERY_ROOT)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    models_root: str = str(DEFAULT_MODELS_ROOT)
    docs_root: str = str(DEFAULT_DOCS_ROOT)
    src_root: str = str(DEFAULT_SRC_ROOT)
    gallery_json: str = str(DEFAULT_ARTIFACT_ROOT / "identity_gallery.json")
    gallery_index_json: str = str(DEFAULT_GALLERY_ROOT / "gallery_index.json")
    gallery_metadata_json: str = str(DEFAULT_GALLERY_ROOT / "gallery_metadata.json")
    similarity_log_csv: str = str(DEFAULT_OUTPUT_ROOT / "similarity_logs.csv")

    @property
    def gallery_dataset_dir(self) -> str:
        return self.train_uploads_dir

    @gallery_dataset_dir.setter
    def gallery_dataset_dir(self, value: str) -> None:
        self.train_uploads_dir = value

    @property
    def query_image_dir(self) -> str:
        return self.test_image_dir

    @query_image_dir.setter
    def query_image_dir(self, value: str) -> None:
        self.test_image_dir = value

    @property
    def query_video_dir(self) -> str:
        return self.test_video_dir

    @query_video_dir.setter
    def query_video_dir(self, value: str) -> None:
        self.test_video_dir = value


@dataclass
class PrototypeConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_directories(self) -> None:
        for raw in (
            self.paths.data_root,
            self.paths.train_uploads_dir,
            self.paths.test_image_dir,
            self.paths.test_video_dir,
            self.paths.artifact_root,
            self.paths.gallery_root,
            self.paths.output_root,
            self.paths.models_root,
            self.paths.docs_root,
            self.paths.src_root,
            str(Path(self.paths.output_root) / "annotated_media"),
            str(Path(self.paths.output_root) / "annotated_images"),
            str(Path(self.paths.output_root) / "annotated_videos"),
            str(Path(self.paths.output_root) / "analytics"),
            str(Path(self.paths.output_root) / "reports"),
            str(Path(self.paths.output_root) / "fingerprints"),
        ):
            Path(raw).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> PrototypeConfig:
    config = PrototypeConfig()
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        _merge_dataclass(config, payload)
    config.ensure_directories()
    return config
