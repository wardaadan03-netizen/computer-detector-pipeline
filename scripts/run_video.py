import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.pipeline import CVPipeline
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    pipeline = CVPipeline(cfg, enable_yolo=not args.no_yolo)
    summary = pipeline.run(
        source=args.video,
        max_frames=args.max_frames,
        save_output=True,
    )
    print("Summary:", summary)


if __name__ == "__main__":
    main()