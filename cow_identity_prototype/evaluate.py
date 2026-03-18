from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import CowIdentityPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize gallery and run a full prototype evaluation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gallery-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = CowIdentityPipeline(config)
    gallery_path = pipeline.initialize_gallery(args.gallery_dir, config.paths.gallery_json)
    images = pipeline.infer_images(args.image_dir, gallery_path, f"{args.prefix}_images", update_gallery=False)
    videos = pipeline.infer_videos(args.video_dir, gallery_path, f"{args.prefix}_videos", update_gallery=False)
    summary_path = Path(config.paths.output_root) / "reports" / f"{args.prefix}_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "gallery_path": str(gallery_path),
                "images": images,
                "videos": videos,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation summary written to {summary_path}")


if __name__ == "__main__":
    main()
