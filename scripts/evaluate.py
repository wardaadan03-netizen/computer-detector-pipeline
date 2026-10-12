import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from src.pipeline import CVPipeline
from src.utils import load_config


def main():
    cfg = load_config()
    pipeline = CVPipeline(cfg, enable_yolo=True)
    pipeline.run(source=cfg["video"]["source"],
                 max_frames=300, save_output=True)

    df = pd.read_csv(cfg["output"]["metrics_path"])
    df.to_csv("data/outputs/metrics_eval.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df["frame"], df["fps"], color="tab:blue")
    axes[0].set_ylabel("FPS")
    axes[1].plot(df["frame"], df["latency_ms"], color="tab:red")
    axes[1].set_ylabel("Latency (ms)")
    axes[2].plot(df["frame"], df["detections"], label="YOLO", color="tab:green")
    axes[2].plot(df["frame"], df["contours"], label="Contours", color="tab:orange")
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("data/outputs/performance.png", dpi=150)
    print("Saved plot to data/outputs/performance.png")


if __name__ == "__main__":
    main()