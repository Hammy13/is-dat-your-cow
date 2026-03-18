from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from .config import PrototypeConfig


def save_prediction_table(records: list[dict], output_csv: str | Path, output_json: str | Path) -> None:
    dataframe = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(records, indent=2), encoding="utf-8")


def build_summary(records: list[dict]) -> dict:
    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        return {
            "records": 0,
            "unique_hybrid_ids": 0,
            "unique_yolo11_ids": 0,
            "unique_cnn_ids": 0,
        }
    return {
        "records": int(len(dataframe)),
        "unique_hybrid_ids": int(dataframe["hybrid_cow_id"].nunique()),
        "unique_yolo11_ids": int(dataframe["yolo11_cow_id"].nunique()),
        "unique_cnn_ids": int(dataframe["cnn_cow_id"].nunique()),
        "mean_hybrid_score": float(dataframe["hybrid_score"].mean()),
        "mean_yolo11_score": float(dataframe["yolo11_score"].mean()),
        "mean_cnn_score": float(dataframe["cnn_score"].mean()),
    }


def save_analytics_dashboard(config: PrototypeConfig, records: list[dict], prefix: str) -> dict[str, Path]:
    analytics_dir = Path(config.paths.output_root) / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    outputs: dict[str, Path] = {}
    if dataframe.empty:
        return outputs

    unique_counts = pd.DataFrame(
        {
            "model": ["YOLO11", "CNN", "Hybrid"],
            "unique_ids": [
                dataframe["yolo11_cow_id"].nunique(),
                dataframe["cnn_cow_id"].nunique(),
                dataframe["hybrid_cow_id"].nunique(),
            ],
        }
    )
    fig = px.bar(unique_counts, x="model", y="unique_ids", title="Unique Cow IDs by Model")
    html_path = analytics_dir / f"{prefix}_unique_ids.html"
    fig.write_html(str(html_path))
    outputs["unique_ids_html"] = html_path

    plt.figure(figsize=(8, 4))
    dataframe[["yolo11_score", "cnn_score", "hybrid_score"]].plot(kind="hist", bins=20, alpha=0.65)
    plt.title("Similarity Score Distribution")
    plt.xlabel("Similarity")
    plt.tight_layout()
    png_path = analytics_dir / f"{prefix}_score_distribution.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    outputs["score_distribution_png"] = png_path

    if "source_name" in dataframe.columns:
        counts = dataframe.groupby("source_name").size().reset_index(name="detections")
        fig = px.bar(counts, x="source_name", y="detections", title="Detections per Source")
        html_counts = analytics_dir / f"{prefix}_detections_per_source.html"
        fig.write_html(str(html_counts))
        outputs["detections_per_source_html"] = html_counts

    return outputs
