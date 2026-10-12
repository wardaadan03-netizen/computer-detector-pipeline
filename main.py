"""Entry point."""
import argparse
from src.pipeline import CVPipeline
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    src = int(args.source) if args.source.isdigit() else args.source
    pipeline = CVPipeline(cfg, enable_yolo=not args.no_yolo)
    summary = pipeline.run(src, args.max_frames, args.save)
    print("Summary:", summary)


if __name__ == "__main__":
    main()