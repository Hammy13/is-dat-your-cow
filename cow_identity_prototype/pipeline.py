from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .analytics import build_summary, save_analytics_dashboard, save_prediction_table
from .config import PrototypeConfig
from .dataset import list_gallery_samples, list_images, list_videos
from .detector import CowDetector
from .embedding import DeepEmbeddingExtractor
from .gallery import IdentityGallery, cosine_similarity
from .mesh import MeshDescriptorExtractor
from .preprocessing import load_image_bgr
from .types import MatchDecision
from .visualization import (
    create_video_writer,
    draw_detection,
    draw_mesh_overlay,
    save_fingerprint_panel,
    save_match_comparison_panel,
)


class CowIdentityPipeline:
    def __init__(self, config: PrototypeConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.detector = CowDetector(config)
        self.mesh = MeshDescriptorExtractor(config.mesh)
        self.embedding = DeepEmbeddingExtractor(config.embedding)
        self._orb = cv2.ORB_create(nfeatures=800)

    def _extract_vectors(self, crop_bgr: np.ndarray) -> dict[str, np.ndarray | dict]:
        mesh_payload = self.mesh.extract(crop_bgr)
        mesh_vector = mesh_payload["vector"]
        cnn_vector = self.embedding.extract(crop_bgr)
        hybrid_vector = np.concatenate(
            [
                mesh_vector * self.config.matching.hybrid_mesh_weight,
                cnn_vector * self.config.matching.hybrid_deep_weight,
            ]
        ).astype(np.float32)
        hybrid_vector /= max(np.linalg.norm(hybrid_vector), 1e-6)
        return {
            "mesh": mesh_payload,
            "yolo11_vector": mesh_vector,
            "cnn_vector": cnn_vector,
            "hybrid_vector": hybrid_vector,
        }

    def _generate_query_variants(self, crop_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        variants: list[tuple[str, np.ndarray]] = [("original", crop_bgr)]
        height, width = crop_bgr.shape[:2]
        specs = [
            ("center_90", 0.05, 0.95, 0.05, 0.95),
            ("center_80", 0.10, 0.90, 0.10, 0.90),
            ("left_85", 0.05, 0.95, 0.00, 0.85),
            ("right_85", 0.05, 0.95, 0.15, 1.00),
        ]
        seen_shapes = {(crop_bgr.shape[0], crop_bgr.shape[1], "original")}
        for name, y_start, y_end, x_start, x_end in specs:
            y1 = int(height * y_start)
            y2 = max(y1 + 8, int(height * y_end))
            x1 = int(width * x_start)
            x2 = max(x1 + 8, int(width * x_end))
            candidate = crop_bgr[y1:y2, x1:x2]
            if candidate.size == 0:
                continue
            key = (candidate.shape[0], candidate.shape[1], name)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            variants.append((name, candidate))
        return variants

    def _save_gallery_variants(
        self,
        gallery: IdentityGallery,
        cow_id: str,
        crop_bgr: np.ndarray,
        source_name: str,
        observation_stem: str,
        notes_prefix: str,
        base_similarity_stats: dict[str, float] | None = None,
    ) -> None:
        variants = self._generate_query_variants(crop_bgr)
        for variant_name, variant_crop in variants:
            vectors = self._extract_vectors(variant_crop)
            suffix = "" if variant_name == "original" else f"__{variant_name}"
            stats = dict(base_similarity_stats or {})
            stats["variant"] = variant_name
            self._save_gallery_observation(
                gallery,
                cow_id=cow_id,
                crop_bgr=variant_crop,
                vectors=vectors,
                source_name=source_name,
                observation_name=f"{observation_stem}{suffix}",
                similarity_stats=stats,
                notes=f"{notes_prefix}; variant={variant_name}",
            )

    def _match_crop_best_variant(
        self,
        gallery: IdentityGallery,
        crop_bgr: np.ndarray,
    ) -> tuple[dict[str, np.ndarray | dict], dict[str, MatchDecision], np.ndarray, str]:
        best_payload: tuple[dict[str, np.ndarray | dict], dict[str, MatchDecision], np.ndarray, str] | None = None
        best_rank: tuple[int, float] | None = None
        for variant_name, variant_crop in self._generate_query_variants(crop_bgr):
            vectors = self._extract_vectors(variant_crop)
            decisions = gallery.match_all(vectors["yolo11_vector"], vectors["cnn_vector"], vectors["hybrid_vector"])
            hybrid = decisions["hybrid"]
            rank = (0 if hybrid.is_new_identity else 1, float(hybrid.score))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_payload = (vectors, decisions, variant_crop, variant_name)
        assert best_payload is not None
        return best_payload

    def _local_pattern_score(self, query_crop_bgr: np.ndarray, gallery_image_path: str) -> float:
        gallery_crop = cv2.imread(gallery_image_path)
        if gallery_crop is None:
            return 0.0
        query_gray = cv2.cvtColor(query_crop_bgr, cv2.COLOR_BGR2GRAY)
        gallery_gray = cv2.cvtColor(gallery_crop, cv2.COLOR_BGR2GRAY)
        kp_query, desc_query = self._orb.detectAndCompute(query_gray, None)
        kp_gallery, desc_gallery = self._orb.detectAndCompute(gallery_gray, None)
        if desc_query is None or desc_gallery is None or not kp_query or not kp_gallery:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desc_query, desc_gallery, k=2)
        good_matches = 0
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches += 1
        return float(good_matches / max(min(len(kp_query), len(kp_gallery)), 1))

    def _global_crop_similarity_score(self, query_crop_bgr: np.ndarray, gallery_image_path: str) -> float:
        gallery_crop = cv2.imread(gallery_image_path)
        if gallery_crop is None:
            return 0.0
        query_gray = cv2.cvtColor(cv2.resize(query_crop_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
        gallery_gray = cv2.cvtColor(cv2.resize(gallery_crop, (256, 256)), cv2.COLOR_BGR2GRAY)
        corr = float(np.corrcoef(query_gray.flatten(), gallery_gray.flatten())[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return corr

    def _apply_duplicate_crop_rescue(
        self,
        gallery: IdentityGallery,
        crop_bgr: np.ndarray,
        vectors: dict[str, np.ndarray | dict],
        decisions: dict[str, MatchDecision],
    ) -> dict[str, MatchDecision]:
        hybrid = decisions["hybrid"]
        if not hybrid.is_new_identity:
            return decisions

        candidate_ids: list[str] = []
        for model_name in ("hybrid", "cnn", "yolo11"):
            for item in decisions[model_name].metadata.get("top_matches", [])[:3]:
                cow_id = str(item.get("cow_id", ""))
                if cow_id and cow_id not in candidate_ids:
                    candidate_ids.append(cow_id)
        if not candidate_ids:
            return decisions

        ranked_similarity: list[dict[str, Any]] = []
        for cow_id in candidate_ids:
            best_similarity = 0.0
            best_gallery_image = ""
            for gallery_image in gallery.get_gallery_images(cow_id):
                similarity = self._global_crop_similarity_score(crop_bgr, gallery_image)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gallery_image = gallery_image
            hybrid_score = gallery._score_by_id(vectors["hybrid_vector"], "hybrid", [cow_id]).get(cow_id, -1.0)
            cnn_score = gallery._score_by_id(vectors["cnn_vector"], "cnn", [cow_id]).get(cow_id, -1.0)
            ranked_similarity.append(
                {
                    "cow_id": cow_id,
                    "duplicate_score": best_similarity,
                    "gallery_image": best_gallery_image,
                    "hybrid_score": hybrid_score,
                    "cnn_score": cnn_score,
                }
            )
        ranked_similarity.sort(key=lambda item: (item["duplicate_score"], item["hybrid_score"], item["cnn_score"]), reverse=True)
        best_similarity = ranked_similarity[0]
        second_similarity = ranked_similarity[1]["duplicate_score"] if len(ranked_similarity) > 1 else 0.0

        looks_like_duplicate = best_similarity["duplicate_score"] >= 0.45
        clear_winner = best_similarity["duplicate_score"] >= (second_similarity + 0.20)
        has_model_support = (
            best_similarity["hybrid_score"] >= (self.config.matching.hybrid_threshold - 0.20)
            or best_similarity["cnn_score"] >= (self.config.matching.cnn_threshold - 0.18)
        )
        if not (looks_like_duplicate and clear_winner and has_model_support):
            return decisions

        decisions["hybrid"] = MatchDecision(
            model_name="hybrid",
            cow_id=str(best_similarity["cow_id"]),
            score=float(best_similarity["hybrid_score"]),
            is_new_identity=False,
            threshold=self.config.matching.hybrid_threshold,
            metadata={
                "top_matches": gallery.top_matches(vectors["hybrid_vector"], "hybrid"),
                "rescue_reason": "duplicate_crop_rescue",
                "duplicate_crop": best_similarity,
            },
        )
        return decisions

    def _apply_local_pattern_rescue(
        self,
        gallery: IdentityGallery,
        crop_bgr: np.ndarray,
        vectors: dict[str, np.ndarray | dict],
        decisions: dict[str, MatchDecision],
    ) -> dict[str, MatchDecision]:
        hybrid = decisions["hybrid"]
        if not hybrid.is_new_identity:
            return decisions

        candidate_ids: list[str] = []
        for model_name in ("hybrid", "cnn", "yolo11"):
            for item in decisions[model_name].metadata.get("top_matches", [])[:3]:
                cow_id = str(item.get("cow_id", ""))
                if cow_id and cow_id not in candidate_ids:
                    candidate_ids.append(cow_id)
        if not candidate_ids:
            return decisions

        ranked_local: list[dict[str, Any]] = []
        for cow_id in candidate_ids:
            best_local_score = 0.0
            best_gallery_image = ""
            for gallery_image in gallery.get_gallery_images(cow_id):
                local_score = self._local_pattern_score(crop_bgr, gallery_image)
                if local_score > best_local_score:
                    best_local_score = local_score
                    best_gallery_image = gallery_image
            hybrid_score = gallery._score_by_id(vectors["hybrid_vector"], "hybrid", [cow_id]).get(cow_id, -1.0)
            cnn_score = gallery._score_by_id(vectors["cnn_vector"], "cnn", [cow_id]).get(cow_id, -1.0)
            ranked_local.append(
                {
                    "cow_id": cow_id,
                    "local_score": best_local_score,
                    "gallery_image": best_gallery_image,
                    "hybrid_score": hybrid_score,
                    "cnn_score": cnn_score,
                }
            )
        ranked_local.sort(key=lambda item: (item["local_score"], item["hybrid_score"], item["cnn_score"]), reverse=True)
        best_local = ranked_local[0]
        second_local = ranked_local[1]["local_score"] if len(ranked_local) > 1 else 0.0

        has_strong_local_pattern = best_local["local_score"] >= 0.08
        has_clear_margin = best_local["local_score"] >= (second_local + 0.02)
        has_model_support = (
            best_local["hybrid_score"] >= (self.config.matching.hybrid_threshold - 0.14)
            or best_local["cnn_score"] >= (self.config.matching.cnn_threshold - 0.12)
        )
        if not (has_strong_local_pattern and has_clear_margin and has_model_support):
            return decisions

        decisions["hybrid"] = MatchDecision(
            model_name="hybrid",
            cow_id=str(best_local["cow_id"]),
            score=float(best_local["hybrid_score"]),
            is_new_identity=False,
            threshold=self.config.matching.hybrid_threshold,
            metadata={
                "top_matches": gallery.top_matches(vectors["hybrid_vector"], "hybrid"),
                "rescue_reason": "local_pattern_rescue",
                "local_pattern": best_local,
            },
        )
        return decisions

    def _save_gallery_observation(
        self,
        gallery: IdentityGallery,
        cow_id: str,
        crop_bgr: np.ndarray,
        vectors: dict[str, np.ndarray | dict],
        source_name: str,
        observation_name: str,
        similarity_stats: dict[str, float] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        observation_metadata = gallery.save_observation_assets(
            cow_id=cow_id,
            observation_name=observation_name,
            crop_bgr=crop_bgr,
            mesh_payload=vectors["mesh"],
            cnn_vector=vectors["cnn_vector"],
            hybrid_vector=vectors["hybrid_vector"],
            source_filename=source_name,
            similarity_stats=similarity_stats,
            notes=notes,
        )
        gallery.add_or_update(
            cow_id=cow_id,
            yolo_mesh_vector=vectors["yolo11_vector"],
            cnn_vector=vectors["cnn_vector"],
            hybrid_vector=vectors["hybrid_vector"],
            source_image=source_name,
            metadata=observation_metadata,
        )
        return observation_metadata

    def initialize_gallery(self, dataset_dir: str | Path | None = None, gallery_path: str | Path | None = None) -> Path:
        dataset_dir = Path(dataset_dir or self.config.paths.gallery_dataset_dir)
        gallery = IdentityGallery(self.config)
        samples = list_gallery_samples(dataset_dir)
        for sample in samples:
            image = load_image_bgr(sample.image_path)
            detection = self.detector.detect_best(image)
            crop_bgr = image if detection is None else detection.crop_bgr
            self._save_gallery_variants(
                gallery,
                cow_id=sample.cow_id,
                crop_bgr=crop_bgr,
                source_name=str(sample.image_path),
                observation_stem=sample.image_path.stem,
                notes_prefix="Labeled gallery initialization",
            )
        gallery_path = gallery.save(gallery_path)
        metadata_path = Path(self.config.paths.artifact_root) / "gallery_initialization_summary.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "gallery_path": str(gallery_path),
                    "samples": len(samples),
                    "cow_ids": sorted(gallery.entries.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return gallery_path

    def build_gallery_from_uploads(self, upload_dir: str | Path | None = None, gallery_path: str | Path | None = None) -> Path:
        upload_dir = Path(upload_dir or self.config.paths.train_uploads_dir)
        gallery_root = Path(self.config.paths.gallery_root)
        if gallery_root.exists():
            for path in gallery_root.iterdir():
                if path.name == "README.md":
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink(missing_ok=True)
        image_paths = list_images(upload_dir)
        observations: list[dict[str, Any]] = []
        for image_path in image_paths:
            image = load_image_bgr(image_path)
            detection = self.detector.detect_best(image)
            crop_bgr = image if detection is None else detection.crop_bgr
            vectors = self._extract_vectors(crop_bgr)
            observations.append(
                {
                    "image_path": image_path,
                    "crop_bgr": crop_bgr,
                    "vectors": vectors,
                }
            )
        gallery = IdentityGallery(self.config)
        if not observations:
            return gallery.save(gallery_path)

        feature_matrix = np.stack([obs["vectors"]["hybrid_vector"] for obs in observations])
        clustering = DBSCAN(
            eps=self.config.training.auto_cluster_eps,
            min_samples=self.config.training.auto_cluster_min_samples,
            metric="cosine",
        )
        labels = clustering.fit_predict(feature_matrix)

        normalized_labels: list[int] = []
        next_noise_label = max([label for label in labels if label >= 0], default=-1) + 1
        for label in labels.tolist():
            if label == -1:
                normalized_labels.append(next_noise_label)
                next_noise_label += 1
            else:
                normalized_labels.append(label)

        label_to_cow_id = {label: f"cow_{index:03d}" for index, label in enumerate(sorted(set(normalized_labels)), start=1)}
        grouped_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
        for label, observation in zip(normalized_labels, observations):
            grouped_vectors[label].append(observation["vectors"]["hybrid_vector"])

        for label, observation in zip(normalized_labels, observations):
            cow_id = label_to_cow_id[label]
            centroid = np.mean(np.stack(grouped_vectors[label]), axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-6)
            similarity = cosine_similarity(observation["vectors"]["hybrid_vector"], centroid)
            self._save_gallery_variants(
                gallery,
                cow_id=cow_id,
                crop_bgr=observation["crop_bgr"],
                source_name=str(observation["image_path"]),
                observation_stem=observation["image_path"].stem,
                notes_prefix="Auto-clustered from train_uploads",
                base_similarity_stats={"cluster_similarity": similarity},
            )

        gallery_path = gallery.save(gallery_path)
        summary_path = Path(self.config.paths.gallery_root) / "gallery_build_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "upload_dir": str(upload_dir),
                    "images_processed": len(image_paths),
                    "clusters_created": len(label_to_cow_id),
                    "gallery_path": str(gallery_path),
                    "cow_ids": sorted(gallery.entries.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return gallery_path

    def _record_gallery_context(self, gallery: IdentityGallery, decisions: dict[str, MatchDecision]) -> dict[str, Any]:
        hybrid_id = decisions["hybrid"].cow_id
        return {
            "hybrid_gallery_folder": str(Path(self.config.paths.gallery_root) / hybrid_id),
            "hybrid_gallery_images": gallery.get_gallery_images(hybrid_id),
            "hybrid_top_matches": decisions["hybrid"].metadata.get("top_matches", []),
            "yolo11_top_matches": decisions["yolo11"].metadata.get("top_matches", []),
            "cnn_top_matches": decisions["cnn"].metadata.get("top_matches", []),
        }

    def _build_conclusion(self, decisions: dict[str, MatchDecision]) -> str:
        hybrid = decisions["hybrid"]
        yolo = decisions["yolo11"]
        cnn = decisions["cnn"]
        if hybrid.is_new_identity:
            top_matches = hybrid.metadata.get("top_matches", [])
            if top_matches:
                best_existing = top_matches[0]
                return (
                    f"No match. Created {hybrid.cow_id} because hybrid score {hybrid.score:.3f} "
                    f"was below threshold {hybrid.threshold:.3f}. Best existing candidate was "
                    f"{best_existing['cow_id']} at {best_existing['score']:.3f}."
                )
            return (
                f"No match. Created {hybrid.cow_id} because the gallery did not contain a confident "
                f"existing match."
            )
        if hybrid.metadata.get("rescue_reason") == "ensemble_consensus":
            return (
                f"Match to {hybrid.cow_id}. Hybrid score {hybrid.score:.3f} was near the threshold, "
                f"but CNN score {cnn.score:.3f} and ensemble agreement rescued the existing identity."
            )
        if hybrid.metadata.get("rescue_reason") == "duplicate_crop_rescue":
            duplicate_crop = hybrid.metadata.get("duplicate_crop", {})
            return (
                f"Match to {hybrid.cow_id}. Hybrid score {hybrid.score:.3f} was below the direct threshold, "
                f"but direct crop similarity {duplicate_crop.get('duplicate_score', 0.0):.3f} strongly matched a stored gallery crop."
            )
        if hybrid.metadata.get("rescue_reason") == "local_pattern_rescue":
            local_pattern = hybrid.metadata.get("local_pattern", {})
            return (
                f"Match to {hybrid.cow_id}. Global hybrid score {hybrid.score:.3f} was below the direct threshold, "
                f"but local coat-pattern alignment score {local_pattern.get('local_score', 0.0):.3f} matched the stored gallery crop."
            )
        return (
            f"Match to {hybrid.cow_id}. Hybrid score {hybrid.score:.3f} exceeded threshold "
            f"{hybrid.threshold:.3f}; supporting scores were YOLO11={yolo.score:.3f} and CNN={cnn.score:.3f}."
        )

    def _build_record(
        self,
        gallery: IdentityGallery,
        source_name: str,
        source_type: str,
        frame_index: int | None,
        box: tuple[int, int, int, int],
        track_id: int | None,
        decisions: dict[str, MatchDecision],
        fingerprint_path: str,
        comparison_path: str,
        gallery_observation_path: str,
    ) -> dict:
        hybrid_metadata = decisions["hybrid"].metadata
        duplicate_crop = hybrid_metadata.get("duplicate_crop", {})
        local_pattern = hybrid_metadata.get("local_pattern", {})
        rescue_gallery_image = (
            duplicate_crop.get("gallery_image")
            or local_pattern.get("gallery_image")
            or ""
        )
        rescue_score = (
            duplicate_crop.get("duplicate_score")
            if duplicate_crop.get("gallery_image")
            else local_pattern.get("local_score", 0.0)
        )
        payload = {
            "source_name": source_name,
            "source_type": source_type,
            "frame_index": frame_index,
            "track_id": track_id,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
            "yolo11_cow_id": decisions["yolo11"].cow_id,
            "yolo11_score": decisions["yolo11"].score,
            "cnn_cow_id": decisions["cnn"].cow_id,
            "cnn_score": decisions["cnn"].score,
            "hybrid_cow_id": decisions["hybrid"].cow_id,
            "hybrid_score": decisions["hybrid"].score,
            "hybrid_is_new_identity": decisions["hybrid"].is_new_identity,
            "hybrid_rescue_reason": hybrid_metadata.get("rescue_reason", ""),
            "hybrid_rescue_gallery_image": rescue_gallery_image,
            "hybrid_rescue_score": rescue_score,
            "fingerprint_path": fingerprint_path,
            "comparison_panel_path": comparison_path,
            "saved_gallery_image_path": gallery_observation_path,
            "conclusion": self._build_conclusion(decisions),
        }
        payload.update(self._record_gallery_context(gallery, decisions))
        return payload

    def _write_canonical_outputs(self, records: list[dict], summary: dict, source_type: str) -> None:
        output_root = Path(self.config.paths.output_root)
        save_prediction_table(records, output_root / "predictions.csv", output_root / "match_results.json")
        (output_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if source_type == "video":
            save_prediction_table(records, output_root / "video_tracking_results.csv", output_root / "video_tracking_results.json")
        Path(self.config.paths.similarity_log_csv).write_text("", encoding="utf-8")
        if records:
            log_lines = ["source_name,source_type,hybrid_cow_id,hybrid_score,yolo11_score,cnn_score"]
            for record in records:
                log_lines.append(
                    f"{record['source_name']},{record['source_type']},{record['hybrid_cow_id']},{record['hybrid_score']:.6f},{record['yolo11_score']:.6f},{record['cnn_score']:.6f}"
                )
            Path(self.config.paths.similarity_log_csv).write_text("\n".join(log_lines), encoding="utf-8")

    def infer_images(
        self,
        image_dir: str | Path | None = None,
        gallery_path: str | Path | None = None,
        output_prefix: str = "image_inference",
        update_gallery: bool = False,
    ) -> dict:
        image_dir = Path(image_dir or self.config.paths.query_image_dir)
        gallery = IdentityGallery.load(self.config, gallery_path)
        records: list[dict] = []
        annotated_dir = Path(self.config.paths.output_root) / "annotated_images"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for image_path in list_images(image_dir):
            image = load_image_bgr(image_path)
            detections = self.detector.detect_image(image)
            annotated = image.copy()
            if not detections:
                crop_bgr = image
                box = (0, 0, image.shape[1], image.shape[0])
                vectors, decisions, best_crop, variant_name = self._match_crop_best_variant(gallery, crop_bgr)
                decisions = self._apply_duplicate_crop_rescue(gallery, best_crop, vectors, decisions)
                decisions = self._apply_local_pattern_rescue(gallery, best_crop, vectors, decisions)
                fingerprint = save_fingerprint_panel(
                    self.config,
                    image_path.name,
                    best_crop,
                    vectors["mesh"]["dark_map"],
                    vectors["mesh"]["light_map"],
                    vectors["mesh"]["texture_map"],
                )
                observation_metadata = {"crop_image_path": ""}
                if update_gallery:
                    observation_metadata = self._save_gallery_observation(
                        gallery,
                        cow_id=decisions["hybrid"].cow_id,
                        crop_bgr=best_crop,
                        vectors=vectors,
                        source_name=image_path.name,
                        observation_name=image_path.name,
                        similarity_stats={
                            "hybrid_score": decisions["hybrid"].score,
                            "yolo11_score": decisions["yolo11"].score,
                            "cnn_score": decisions["cnn"].score,
                        },
                        notes=f"Inference fallback using best variant: {variant_name}",
                    )
                comparison = save_match_comparison_panel(
                    self.config,
                    image_path.name,
                    best_crop,
                    gallery.get_gallery_images(decisions["hybrid"].cow_id),
                    decisions,
                    self.config.mesh.grid_size,
                )
                records.append(
                    self._build_record(
                        gallery,
                        image_path.name,
                        "image",
                        None,
                        box,
                        None,
                        decisions,
                        str(fingerprint),
                        str(comparison),
                        observation_metadata.get("crop_image_path", ""),
                    )
                )
            else:
                for index, detection in enumerate(detections, start=1):
                    vectors, decisions, best_crop, variant_name = self._match_crop_best_variant(gallery, detection.crop_bgr)
                    decisions = self._apply_duplicate_crop_rescue(gallery, best_crop, vectors, decisions)
                    decisions = self._apply_local_pattern_rescue(gallery, best_crop, vectors, decisions)
                    fingerprint = save_fingerprint_panel(
                        self.config,
                        f"{image_path.stem}_{index}",
                        best_crop,
                        vectors["mesh"]["dark_map"],
                        vectors["mesh"]["light_map"],
                        vectors["mesh"]["texture_map"],
                    )
                    annotated = draw_mesh_overlay(annotated, detection, self.config.mesh.grid_size)
                    annotated = draw_detection(annotated, detection, decisions)
                    observation_metadata = {"crop_image_path": ""}
                    if update_gallery:
                        observation_metadata = self._save_gallery_observation(
                            gallery,
                            cow_id=decisions["hybrid"].cow_id,
                            crop_bgr=best_crop,
                            vectors=vectors,
                            source_name=image_path.name,
                            observation_name=f"{image_path.stem}_{index}",
                            similarity_stats={
                                "hybrid_score": decisions["hybrid"].score,
                                "yolo11_score": decisions["yolo11"].score,
                                "cnn_score": decisions["cnn"].score,
                            },
                            notes=f"Image inference match via variant: {variant_name}",
                        )
                    comparison = save_match_comparison_panel(
                        self.config,
                        f"{image_path.stem}_{index}",
                        best_crop,
                        gallery.get_gallery_images(decisions["hybrid"].cow_id),
                        decisions,
                        self.config.mesh.grid_size,
                    )
                    records.append(
                        self._build_record(
                            gallery,
                            image_path.name,
                            "image",
                            None,
                            detection.box,
                            detection.track_id,
                            decisions,
                            str(fingerprint),
                            str(comparison),
                            observation_metadata.get("crop_image_path", ""),
                        )
                    )
            output_image = annotated_dir / f"{image_path.stem}_annotated.png"
            cv2.imwrite(str(output_image), annotated)
        if update_gallery:
            gallery.save(gallery_path)
        output_csv = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}.csv"
        output_json = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}.json"
        save_prediction_table(records, output_csv, output_json)
        summary = build_summary(records)
        summary_path = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        charts = save_analytics_dashboard(self.config, records, output_prefix)
        self._write_canonical_outputs(records, summary, "image")
        return {
            "csv": str(output_csv),
            "json": str(output_json),
            "summary": str(summary_path),
            "charts": {key: str(value) for key, value in charts.items()},
            "records": records,
        }

    def infer_videos(
        self,
        video_dir: str | Path | None = None,
        gallery_path: str | Path | None = None,
        output_prefix: str = "video_inference",
        update_gallery: bool = False,
    ) -> dict:
        video_dir = Path(video_dir or self.config.paths.query_video_dir)
        gallery = IdentityGallery.load(self.config, gallery_path)
        records: list[dict] = []
        per_video_summary: list[dict] = []
        annotated_dir = Path(self.config.paths.output_root) / "annotated_videos"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for video_path in list_videos(video_dir):
            track_assignments: dict[int, dict[str, Any]] = {}
            seen_hybrid_ids: set[str] = set()
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            cap.release()
            effective_fps = max(fps / max(self.config.inference.sample_every_n_frames, 1), 1.0)
            writer = create_video_writer(annotated_dir / f"{video_path.stem}_annotated.mp4", effective_fps, (width, height))
            try:
                for frame_index, frame, detections in self.detector.track_video(video_path):
                    annotated = frame.copy()
                    for detection in detections:
                        vectors = self._extract_vectors(detection.crop_bgr)
                        if detection.track_id is not None and detection.track_id in track_assignments:
                            decisions = track_assignments[detection.track_id]["decisions"]
                            track_assignments[detection.track_id]["hits"] += 1
                            existing_images = gallery.get_gallery_images(decisions["hybrid"].cow_id)
                            observation_metadata = {
                                "crop_image_path": existing_images[-1] if existing_images else ""
                            }
                        else:
                            decisions = gallery.match_all(vectors["yolo11_vector"], vectors["cnn_vector"], vectors["hybrid_vector"])
                            decisions = self._apply_duplicate_crop_rescue(gallery, detection.crop_bgr, vectors, decisions)
                            decisions = self._apply_local_pattern_rescue(gallery, detection.crop_bgr, vectors, decisions)
                            if detection.track_id is not None:
                                track_assignments[detection.track_id] = {"hits": 1, "decisions": decisions}
                            observation_metadata = {"crop_image_path": ""}
                            if update_gallery:
                                observation_metadata = self._save_gallery_observation(
                                    gallery,
                                    cow_id=decisions["hybrid"].cow_id,
                                    crop_bgr=detection.crop_bgr,
                                    vectors=vectors,
                                    source_name=video_path.name,
                                    observation_name=f"{video_path.stem}_{frame_index}_{detection.track_id or 'det'}",
                                    similarity_stats={
                                        "hybrid_score": decisions["hybrid"].score,
                                        "yolo11_score": decisions["yolo11"].score,
                                        "cnn_score": decisions["cnn"].score,
                                    },
                                    notes="Video inference track observation",
                                )
                        fingerprint = save_fingerprint_panel(
                            self.config,
                            f"{video_path.stem}_{frame_index}_{detection.track_id or 'det'}",
                            detection.crop_bgr,
                            vectors["mesh"]["dark_map"],
                            vectors["mesh"]["light_map"],
                            vectors["mesh"]["texture_map"],
                        )
                        comparison = save_match_comparison_panel(
                            self.config,
                            f"{video_path.stem}_{frame_index}_{detection.track_id or 'det'}",
                            detection.crop_bgr,
                            gallery.get_gallery_images(decisions["hybrid"].cow_id),
                            decisions,
                            self.config.mesh.grid_size,
                        )
                        records.append(
                            self._build_record(
                                gallery,
                                video_path.name,
                                "video",
                                frame_index,
                                detection.box,
                                detection.track_id,
                                decisions,
                                str(fingerprint),
                                str(comparison),
                                observation_metadata.get("crop_image_path", ""),
                            )
                        )
                        seen_hybrid_ids.add(decisions["hybrid"].cow_id)
                        annotated = draw_mesh_overlay(annotated, detection, self.config.mesh.grid_size)
                        annotated = draw_detection(annotated, detection, decisions)
                    writer.write(annotated)
            finally:
                writer.release()
            source_records = [record for record in records if record["source_name"] == video_path.name]
            per_video_summary.append(
                {
                    "video": video_path.name,
                    "predicted_unique_cows_hybrid": len(seen_hybrid_ids),
                    "predicted_unique_cows_yolo11": len({record["yolo11_cow_id"] for record in source_records}),
                    "predicted_unique_cows_cnn": len({record["cnn_cow_id"] for record in source_records}),
                    "tracked_objects": len(track_assignments),
                }
            )
        if update_gallery:
            gallery.save(gallery_path)
        output_csv = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}.csv"
        output_json = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}.json"
        save_prediction_table(records, output_csv, output_json)
        per_video_path = Path(self.config.paths.output_root) / "reports" / f"{output_prefix}_per_video_summary.json"
        per_video_path.write_text(json.dumps(per_video_summary, indent=2), encoding="utf-8")
        summary = build_summary(records)
        charts = save_analytics_dashboard(self.config, records, output_prefix)
        self._write_canonical_outputs(records, summary, "video")
        return {
            "csv": str(output_csv),
            "json": str(output_json),
            "per_video_summary": str(per_video_path),
            "charts": {key: str(value) for key, value in charts.items()},
            "records": records,
        }
