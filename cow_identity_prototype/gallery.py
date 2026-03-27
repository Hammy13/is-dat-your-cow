from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PrototypeConfig
from .types import GalleryEntry, MatchDecision


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b)), 1e-6)
    return float(np.dot(vec_a, vec_b) / denom)


class IdentityGallery:
    def __init__(self, config: PrototypeConfig) -> None:
        self.config = config
        self.entries: dict[str, GalleryEntry] = {}
        self._vector_cache: dict[tuple[str, str], np.ndarray] = {}

    def save_observation_assets(
        self,
        cow_id: str,
        observation_name: str,
        crop_bgr: np.ndarray,
        mesh_payload: dict[str, Any],
        cnn_vector: np.ndarray,
        hybrid_vector: np.ndarray,
        source_filename: str,
        similarity_stats: dict[str, float] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cow_folder = Path(self.config.paths.gallery_root) / cow_id
        images_dir = cow_folder / "images"
        embeddings_dir = cow_folder / "embeddings"
        descriptors_dir = cow_folder / "descriptors"
        for path in (images_dir, embeddings_dir, descriptors_dir):
            path.mkdir(parents=True, exist_ok=True)

        stem = Path(observation_name).stem
        crop_path = images_dir / f"{stem}.png"
        cnn_path = embeddings_dir / f"{stem}_cnn.npy"
        hybrid_path = embeddings_dir / f"{stem}_hybrid.npy"
        mesh_path = descriptors_dir / f"{stem}_mesh.npy"
        mesh_json_path = descriptors_dir / f"{stem}_mesh.json"

        cv2.imwrite(str(crop_path), crop_bgr)
        np.save(cnn_path, cnn_vector)
        np.save(hybrid_path, hybrid_vector)
        np.save(mesh_path, mesh_payload["vector"])
        mesh_json_path.write_text(
            json.dumps(
                {
                    "source_filename": source_filename,
                    "timestamp": timestamp,
                    "dark_map_mean": float(np.mean(mesh_payload["dark_map"])),
                    "light_map_mean": float(np.mean(mesh_payload["light_map"])),
                    "texture_map_mean": float(np.mean(mesh_payload["texture_map"])),
                    "score_map_mean": float(np.mean(mesh_payload["score_map"])),
                    "score_map_max": float(np.max(mesh_payload["score_map"])),
                    "mesh_matrix": mesh_payload["matrix"],
                    "similarity_stats": similarity_stats or {},
                    "notes": notes or "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "source_filename": source_filename,
            "timestamp": timestamp,
            "crop_image_path": str(crop_path),
            "cnn_vector_path": str(cnn_path),
            "hybrid_vector_path": str(hybrid_path),
            "mesh_vector_path": str(mesh_path),
            "mesh_descriptor_path": str(mesh_json_path),
            "similarity_stats": similarity_stats or {},
            "notes": notes or "",
        }

    def add_or_update(
        self,
        cow_id: str,
        yolo_mesh_vector: np.ndarray,
        cnn_vector: np.ndarray,
        hybrid_vector: np.ndarray,
        source_image: str,
        metadata: dict | None = None,
    ) -> None:
        metadata = metadata or {}
        if cow_id not in self.entries:
            self.entries[cow_id] = GalleryEntry(
                cow_id=cow_id,
                yolo_mesh_centroid=yolo_mesh_vector.copy(),
                cnn_centroid=cnn_vector.copy(),
                hybrid_centroid=hybrid_vector.copy(),
                observation_count=1,
                source_images=[source_image],
                metadata={"observations": [metadata]} if metadata else {"observations": []},
            )
            return
        entry = self.entries[cow_id]
        count = entry.observation_count
        entry.yolo_mesh_centroid = ((entry.yolo_mesh_centroid * count) + yolo_mesh_vector) / (count + 1)
        entry.cnn_centroid = ((entry.cnn_centroid * count) + cnn_vector) / (count + 1)
        entry.hybrid_centroid = ((entry.hybrid_centroid * count) + hybrid_vector) / (count + 1)
        entry.observation_count += 1
        entry.source_images.append(source_image)
        entry.metadata.setdefault("observations", [])
        if metadata:
            entry.metadata["observations"].append(metadata)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.config.paths.gallery_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        index_payload = {"generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z", "cows": {}}
        for cow_id, entry in self.entries.items():
            serializable[cow_id] = {
                "cow_id": entry.cow_id,
                "yolo_mesh_centroid": entry.yolo_mesh_centroid.tolist(),
                "cnn_centroid": entry.cnn_centroid.tolist(),
                "hybrid_centroid": entry.hybrid_centroid.tolist(),
                "observation_count": entry.observation_count,
                "source_images": entry.source_images,
                "metadata": entry.metadata,
            }
            observations = entry.metadata.get("observations", [])
            gallery_images = [obs.get("crop_image_path") for obs in observations if obs.get("crop_image_path")]
            index_payload["cows"][cow_id] = {
                "cow_id": cow_id,
                "gallery_folder": str(Path(self.config.paths.gallery_root) / cow_id),
                "observation_count": entry.observation_count,
                "source_images": entry.source_images,
                "gallery_images": gallery_images,
                "mesh_descriptor_paths": [obs.get("mesh_descriptor_path") for obs in observations if obs.get("mesh_descriptor_path")],
                "cnn_vector_paths": [obs.get("cnn_vector_path") for obs in observations if obs.get("cnn_vector_path")],
                "hybrid_vector_paths": [obs.get("hybrid_vector_path") for obs in observations if obs.get("hybrid_vector_path")],
                "notes": [obs.get("notes", "") for obs in observations if obs.get("notes")],
            }
        target.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        Path(self.config.paths.gallery_index_json).write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        Path(self.config.paths.gallery_metadata_json).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, config: PrototypeConfig, path: str | Path | None = None) -> "IdentityGallery":
        gallery = cls(config)
        target = Path(path or config.paths.gallery_json)
        if not target.exists():
            return gallery
        payload = json.loads(target.read_text(encoding="utf-8"))
        for cow_id, raw in payload.items():
            gallery.entries[cow_id] = GalleryEntry(
                cow_id=cow_id,
                yolo_mesh_centroid=np.asarray(raw["yolo_mesh_centroid"], dtype=np.float32),
                cnn_centroid=np.asarray(raw["cnn_centroid"], dtype=np.float32),
                hybrid_centroid=np.asarray(raw["hybrid_centroid"], dtype=np.float32),
                observation_count=int(raw["observation_count"]),
                source_images=list(raw["source_images"]),
                metadata=dict(raw.get("metadata", {})),
            )
        return gallery

    def top_matches(self, vector: np.ndarray, model_name: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._top_matches_from_ids(vector, model_name, list(self.entries.keys()), top_k=top_k)

    def _top_matches_from_ids(self, vector: np.ndarray, model_name: str, candidate_ids: list[str], top_k: int = 3) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for cow_id in candidate_ids:
            score, details = self._score_cow_id(vector, model_name, cow_id)
            scored.append({"cow_id": cow_id, "score": score, **details})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _best_match(self, vector: np.ndarray, model_name: str, candidate_ids: list[str] | None = None) -> tuple[str, float]:
        ids = candidate_ids if candidate_ids is not None else list(self.entries.keys())
        top = self._top_matches_from_ids(vector, model_name, ids, top_k=1)
        if not top:
            return "UNKNOWN", -1.0
        return top[0]["cow_id"], float(top[0]["score"])

    def _score_by_id(self, vector: np.ndarray, model_name: str, candidate_ids: list[str] | None = None) -> dict[str, float]:
        scores: dict[str, float] = {}
        ids = candidate_ids if candidate_ids is not None else list(self.entries.keys())
        for cow_id in ids:
            score, _ = self._score_cow_id(vector, model_name, cow_id)
            scores[cow_id] = score
        return scores

    def _centroid_for_model(self, entry: GalleryEntry, model_name: str) -> np.ndarray:
        return {
            "yolo11": entry.yolo_mesh_centroid,
            "cnn": entry.cnn_centroid,
            "hybrid": entry.hybrid_centroid,
        }[model_name]

    def _observation_vector_paths(self, entry: GalleryEntry, model_name: str) -> list[str]:
        key = {
            "yolo11": "mesh_vector_path",
            "cnn": "cnn_vector_path",
            "hybrid": "hybrid_vector_path",
        }[model_name]
        observations = entry.metadata.get("observations", [])
        paths: list[str] = []
        for observation in observations:
            raw_path = observation.get(key)
            if raw_path:
                paths.append(str(raw_path))
        return paths

    def _load_vector(self, path: str) -> np.ndarray | None:
        cache_key = ("npy", path)
        cached = self._vector_cache.get(cache_key)
        if cached is not None:
            return cached
        target = Path(path)
        if not target.exists():
            return None
        try:
            vector = np.asarray(np.load(target), dtype=np.float32)
        except Exception:
            return None
        self._vector_cache[cache_key] = vector
        return vector

    def _score_cow_id(self, vector: np.ndarray, model_name: str, cow_id: str) -> tuple[float, dict[str, Any]]:
        entry = self.entries[cow_id]
        centroid = self._centroid_for_model(entry, model_name)
        centroid_score = cosine_similarity(vector, centroid)

        exemplar_score = -1.0
        exemplar_path = ""
        for vector_path in self._observation_vector_paths(entry, model_name):
            exemplar = self._load_vector(vector_path)
            if exemplar is None:
                continue
            score = cosine_similarity(vector, exemplar)
            if score > exemplar_score:
                exemplar_score = score
                exemplar_path = vector_path

        if exemplar_score < 0.0:
            return centroid_score, {
                "centroid_score": centroid_score,
                "exemplar_score": centroid_score,
                "best_vector_path": "",
            }

        blended_score = max(centroid_score, exemplar_score, (0.35 * centroid_score) + (0.65 * exemplar_score))
        return blended_score, {
            "centroid_score": centroid_score,
            "exemplar_score": exemplar_score,
            "best_vector_path": exemplar_path,
        }

    def _is_training_entry(self, entry: GalleryEntry) -> bool:
        observations = entry.metadata.get("observations", [])
        for observation in observations:
            note = str(observation.get("notes", ""))
            source_filename = str(observation.get("source_filename", ""))
            if "Auto-clustered from train_uploads" in note or "Labeled gallery initialization" in note:
                return True
            if "train_uploads" in source_filename:
                return True
        return False

    def _training_ids(self) -> list[str]:
        return [cow_id for cow_id, entry in self.entries.items() if self._is_training_entry(entry)]

    def _match_for_ids(
        self,
        yolo_mesh_vector: np.ndarray,
        cnn_vector: np.ndarray,
        hybrid_vector: np.ndarray,
        candidate_ids: list[str],
    ) -> dict[str, MatchDecision]:
        thresholds = {
            "yolo11": self.config.matching.yolo_mesh_threshold,
            "cnn": self.config.matching.cnn_threshold,
            "hybrid": self.config.matching.hybrid_threshold,
        }
        vectors = {
            "yolo11": yolo_mesh_vector,
            "cnn": cnn_vector,
            "hybrid": hybrid_vector,
        }
        decisions: dict[str, MatchDecision] = {}
        for model_name, vector in vectors.items():
            if not candidate_ids:
                new_id = self._next_cow_id()
                decisions[model_name] = MatchDecision(
                    model_name=model_name,
                    cow_id=new_id,
                    score=0.0,
                    is_new_identity=True,
                    threshold=thresholds[model_name],
                    metadata={"top_matches": []},
                )
                continue
            best_id, best_score = self._best_match(vector, model_name, candidate_ids)
            is_new = best_score < thresholds[model_name] and self.config.matching.allow_new_identity
            decisions[model_name] = MatchDecision(
                model_name=model_name,
                cow_id=best_id if not is_new else self._next_cow_id(),
                score=best_score,
                is_new_identity=is_new,
                threshold=thresholds[model_name],
                metadata={"top_matches": self._top_matches_from_ids(vector, model_name, candidate_ids)},
            )

        hybrid_decision = decisions["hybrid"]
        if hybrid_decision.is_new_identity and candidate_ids:
            yolo_scores = self._score_by_id(yolo_mesh_vector, "yolo11", candidate_ids)
            cnn_scores = self._score_by_id(cnn_vector, "cnn", candidate_ids)
            hybrid_scores = self._score_by_id(hybrid_vector, "hybrid", candidate_ids)
            ensemble_ranked: list[dict[str, float]] = []
            for cow_id in candidate_ids:
                mesh_score = yolo_scores.get(cow_id, -1.0)
                cnn_score = cnn_scores.get(cow_id, -1.0)
                hybrid_score = hybrid_scores.get(cow_id, -1.0)
                ensemble_score = 0.20 * mesh_score + 0.40 * cnn_score + 0.40 * hybrid_score
                ensemble_ranked.append(
                    {
                        "cow_id": cow_id,
                        "mesh_score": mesh_score,
                        "cnn_score": cnn_score,
                        "hybrid_score": hybrid_score,
                        "ensemble_score": ensemble_score,
                    }
                )
            ensemble_ranked.sort(key=lambda item: item["ensemble_score"], reverse=True)
            best_ensemble = ensemble_ranked[0] if ensemble_ranked else None
            if best_ensemble is not None:
                same_as_cnn_top = best_ensemble["cow_id"] == decisions["cnn"].cow_id
                same_as_yolo_top = best_ensemble["cow_id"] == decisions["yolo11"].cow_id
                hybrid_near = best_ensemble["hybrid_score"] >= (
                    self.config.matching.hybrid_threshold - self.config.matching.hybrid_rescue_margin
                )
                cnn_support = best_ensemble["cnn_score"] >= self.config.matching.cnn_threshold
                mesh_support = best_ensemble["mesh_score"] >= self.config.matching.yolo_mesh_threshold
                ensemble_support = best_ensemble["ensemble_score"] >= self.config.matching.ensemble_threshold
                if ensemble_support and ((same_as_cnn_top and cnn_support) or (same_as_yolo_top and mesh_support) or (hybrid_near and cnn_support)):
                    decisions["hybrid"] = MatchDecision(
                        model_name="hybrid",
                        cow_id=str(best_ensemble["cow_id"]),
                        score=float(best_ensemble["hybrid_score"]),
                        is_new_identity=False,
                        threshold=thresholds["hybrid"],
                        metadata={
                            "top_matches": self._top_matches_from_ids(hybrid_vector, "hybrid", candidate_ids),
                            "rescue_reason": "ensemble_consensus",
                            "ensemble_top": best_ensemble,
                        },
                    )
        return decisions

    def match_all(self, yolo_mesh_vector: np.ndarray, cnn_vector: np.ndarray, hybrid_vector: np.ndarray) -> dict[str, MatchDecision]:
        if not self.entries:
            return self._match_for_ids(yolo_mesh_vector, cnn_vector, hybrid_vector, [])

        all_ids = list(self.entries.keys())
        training_ids = self._training_ids()
        train_decisions = self._match_for_ids(yolo_mesh_vector, cnn_vector, hybrid_vector, training_ids or all_ids)
        if not training_ids or set(training_ids) == set(all_ids):
            return train_decisions

        all_decisions = self._match_for_ids(yolo_mesh_vector, cnn_vector, hybrid_vector, all_ids)
        train_hybrid = train_decisions["hybrid"]
        all_hybrid = all_decisions["hybrid"]
        if not train_hybrid.is_new_identity:
            return train_decisions
        if not all_hybrid.is_new_identity and all_hybrid.score >= (train_hybrid.score + self.config.matching.training_priority_margin):
            return all_decisions
        return train_decisions

    def get_gallery_images(self, cow_id: str) -> list[str]:
        entry = self.entries.get(cow_id)
        if entry is None:
            return []
        observations = entry.metadata.get("observations", [])
        paths: list[str] = []
        for observation in observations:
            raw_path = observation.get("crop_image_path")
            if not raw_path:
                continue
            image_path = Path(str(raw_path))
            if image_path.exists():
                paths.append(str(image_path))
        return paths

    def _next_cow_id(self) -> str:
        existing = []
        for cow_id in self.entries:
            lowered = cow_id.lower()
            if lowered.startswith("cow_"):
                try:
                    existing.append(int(lowered.split("_")[-1]))
                except ValueError:
                    continue
        next_index = max(existing, default=0) + 1
        prefix = "cow_"
        for cow_id in self.entries:
            if str(cow_id).startswith("COW_"):
                prefix = "COW_"
                break
        return f"{prefix}{next_index:03d}"
