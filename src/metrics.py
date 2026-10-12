import time
import csv
import numpy as np
from pathlib import Path

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MetricsTracker:
    def __init__(self, log_path: str = "data/outputs/metrics.csv"):
        p = Path(log_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        self.log_path = p
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.frame_id = 0

    @staticmethod
    def iou(box_a,box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    def log_frame(self,latency_ms:float,n_det:int,n_contours:int,mean_conf:float):
        self.frame_id+=1
        fps=1000.0/latency_ms if latency_ms>0 else 0.0
        self.rows.append({
            "frame":self.frame_id,
            "latency_ms":round(latency_ms,3),
            "fps":round(fps,2),
            "detections":n_contours,
            "mean_conf":round(mean_conf,4),
        })

    def flush(self):
        if not self.rows:
            return 
            with open(self.log_path,"w",newline=" ")as f:
                writer=csv.DictWriter(f,fieldnames=self.rows[0].keys())
                writer.writeheader()
                writer.writerows(self.rows)

    def summary(self)->dict:
        if not self.rows:
            return {}
        lat = np.array([r["latency_ms"] for r in self.rows])
        fps = np.array([r["fps"] for r in self.rows])
        det = np.array([r["detections"] for r in self.rows])
        return {
            "frames": len(self.rows),
            "latency_mean_ms": float(lat.mean()),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "fps_mean": float(fps.mean()),
            "fps_min": float(fps.min()),
            "detections_mean": float(det.mean()),
            "detections_total": int(det.sum()),
        }