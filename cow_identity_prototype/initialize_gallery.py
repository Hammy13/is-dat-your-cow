from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import CowIdentityPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the cow identity gallery from labeled images.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Labeled gallery dataset directory.")
    parser.add_argument("--gallery-path", type=str, default=None, help="Output gallery JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = CowIdentityPipeline(config)
    gallery_path = pipeline.initialize_gallery(args.dataset_dir, args.gallery_path)
    print(f"Gallery saved to {gallery_path}")


if __name__ == "__main__":
    main()
