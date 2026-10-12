import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import CVPipeline
from src.utils import load_config


if __name__ == "__main__":
    cfg = load_config()
    cfg["video"]["source"] = 0
    pipeline = CVPipeline(cfg, enable_yolo=True)
    summary = pipeline.run(source=0, save_output=False)
    print("Summary:", summary)