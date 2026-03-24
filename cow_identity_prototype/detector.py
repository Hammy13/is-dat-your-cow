from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import PrototypeConfig
from .preprocessing import crop_image, passes_quality_gate
from .types import Detection


class CowDetector:
    def __init__(self, config: PrototypeConfig) -> None:
        self.config = config
        self.model = YOLO(config.detector.model_path)
        self.cow_class_ids = self._resolve_cow_class_ids()

    def _resolve_cow_class_ids(self) -> list[int] | None:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            ids = []
            for idx, name in names.items():
                lowered = str(name).lower()
                if any(hint in lowered for hint in self.config.detector.classes_hint):
                    ids.append(int(idx))
            return ids or None
        return None

    def _raw_box_count(self, result) -> int:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return 0
        try:
            return int(len(boxes.xyxy))
        except Exception:
            return 0

    def _build_detections(self, image: np.ndarray, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([None] * len(xyxy), dtype=object)
        for idx, raw_box in enumerate(xyxy):
            box = tuple(int(v) for v in raw_box.tolist())
            class_name = str(self.model.names.get(int(cls[idx]), cls[idx])) if isinstance(self.model.names, dict) else str(cls[idx])
            crop_bgr = crop_image(image, box)
            passed, metrics = passes_quality_gate(
                box,
                crop_bgr,
                image.shape,
                self.config.inference.side_view_min_aspect_ratio,
                self.config.inference.min_crop_area_ratio,
                self.config.inference.min_sharpness,
            )
            if not passed:
                continue
            detections.append(
                Detection(
                    box=box,
                    confidence=float(conf[idx]),
                    class_name=class_name,
                    crop_bgr=crop_bgr,
                    track_id=None if track_ids[idx] is None else int(track_ids[idx]),
                    side_view_score=metrics["side_view_score"],
                    sharpness=metrics["sharpness"],
                )
            )
        return detections

    def _predict_image_result(self, image: np.ndarray):
        results = self.model.predict(
            image,
            conf=self.config.detector.confidence,
            iou=self.config.detector.iou,
            imgsz=self.config.detector.image_size,
            classes=self.cow_class_ids,
            verbose=False,
        )
        return results[0]

    def detect_image_with_context(self, image: np.ndarray) -> tuple[list[Detection], int]:
        result = self._predict_image_result(image)
        return self._build_detections(image, result), self._raw_box_count(result)

    def detect_image(self, image: np.ndarray) -> list[Detection]:
        detections, _ = self.detect_image_with_context(image)
        return detections

    def detect_best(self, image: np.ndarray) -> Detection | None:
        detections = self.detect_image(image)
        if not detections:
            return None
        detections.sort(key=lambda item: (item.side_view_score, item.confidence), reverse=True)
        return detections[0]

    def track_video(self, video_path: str | Path):
        for frame_index, frame, detections, _ in self.track_video_with_context(video_path):
            yield frame_index, frame, detections

    def track_video_with_context(self, video_path: str | Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % self.config.inference.sample_every_n_frames != 0:
                    frame_index += 1
                    continue
                results = self.model.track(
                    frame,
                    conf=self.config.detector.confidence,
                    iou=self.config.detector.iou,
                    imgsz=self.config.detector.image_size,
                    classes=self.cow_class_ids,
                    tracker=self.config.detector.tracker,
                    persist=True,
                    verbose=False,
                )
                result = results[0]
                detections = self._build_detections(frame, result)
                yield frame_index, frame, detections, self._raw_box_count(result)
                frame_index += 1
        finally:
            cap.release()
