from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import CowIdentityPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image/video cow identity inference.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override.")
    parser.add_argument("--mode", choices=["images", "videos", "both"], default="both")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--gallery-path", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="demo")
    parser.add_argument("--update-gallery", action="store_true", help="Persist inference detections into the gallery.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = CowIdentityPipeline(config)
    payload = {}
    if args.mode in {"images", "both"}:
        payload["images"] = pipeline.infer_images(args.image_dir, args.gallery_path, f"{args.output_prefix}_images", update_gallery=args.update_gallery)
    if args.mode in {"videos", "both"}:
        payload["videos"] = pipeline.infer_videos(args.video_dir, args.gallery_path, f"{args.output_prefix}_videos", update_gallery=args.update_gallery)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
