from __future__ import annotations

import cv2
import numpy as np

from .config import MeshConfig


class MeshDescriptorExtractor:
    def __init__(self, config: MeshConfig) -> None:
        self.config = config

    def extract(self, crop_bgr: np.ndarray) -> dict[str, np.ndarray | list[list[dict[str, float]]]]:
        grid = self.config.grid_size
        resized = cv2.resize(crop_bgr, (grid * 8, grid * 8), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        cell_h = resized.shape[0] // grid
        cell_w = resized.shape[1] // grid
        matrix: list[list[dict[str, float]]] = []
        vector: list[float] = []
        dark_map = np.zeros((grid, grid), dtype=np.float32)
        light_map = np.zeros((grid, grid), dtype=np.float32)
        texture_map = np.zeros((grid, grid), dtype=np.float32)
        for row in range(grid):
            matrix_row: list[dict[str, float]] = []
            for col in range(grid):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                cell_bgr = resized[y1:y2, x1:x2]
                cell_hsv = hsv[y1:y2, x1:x2]
                cell_gray = gray[y1:y2, x1:x2]
                dark_pct = float((cell_gray < self.config.dark_threshold).mean())
                light_pct = float((cell_gray > self.config.light_threshold).mean())
                rgb_mean = cell_bgr.mean(axis=(0, 1)) / 255.0
                rgb_std = cell_bgr.std(axis=(0, 1)) / 255.0
                hue_hist = cv2.calcHist([cell_hsv], [0], None, [self.config.histogram_bins], [0, 180]).flatten()
                hue_hist = hue_hist / max(hue_hist.sum(), 1e-6)
                texture = float(cv2.Laplacian(cell_gray, cv2.CV_32F).var() / 255.0)
                gradient_x = cv2.Sobel(cell_gray, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(cell_gray, cv2.CV_32F, 0, 1, ksize=3)
                edge_strength = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)) / 255.0)
                features = {
                    "dark_pct": dark_pct,
                    "light_pct": light_pct,
                    "blue_mean": float(rgb_mean[0]),
                    "green_mean": float(rgb_mean[1]),
                    "red_mean": float(rgb_mean[2]),
                    "blue_std": float(rgb_std[0]),
                    "green_std": float(rgb_std[1]),
                    "red_std": float(rgb_std[2]),
                    "texture": texture,
                    "edge_strength": edge_strength,
                }
                matrix_row.append(features)
                dark_map[row, col] = dark_pct
                light_map[row, col] = light_pct
                texture_map[row, col] = texture
                vector.extend(
                    [
                        dark_pct,
                        light_pct,
                        float(rgb_mean[0]),
                        float(rgb_mean[1]),
                        float(rgb_mean[2]),
                        float(rgb_std[0]),
                        float(rgb_std[1]),
                        float(rgb_std[2]),
                        texture,
                        edge_strength,
                    ]
                )
                vector.extend(float(value) for value in hue_hist.tolist())
            matrix.append(matrix_row)
        descriptor = np.asarray(vector, dtype=np.float32)
        descriptor /= max(np.linalg.norm(descriptor), 1e-6)
        return {
            "vector": descriptor,
            "matrix": matrix,
            "dark_map": dark_map,
            "light_map": light_map,
            "texture_map": texture_map,
            "preview": resized,
        }
