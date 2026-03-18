from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import CowIdentityPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gallery from uploaded cow images using clustering.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override.")
    parser.add_argument("--upload-dir", type=str, default=None, help="Directory of uploaded training images.")
    parser.add_argument("--gallery-path", type=str, default=None, help="Output gallery JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = CowIdentityPipeline(config)
    gallery_path = pipeline.build_gallery_from_uploads(args.upload_dir, args.gallery_path)
    print(f"Auto-clustered gallery saved to {gallery_path}")


if __name__ == "__main__":
    main()
