"""Main pipeline: orchestration + output writers."""
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import Preprocessor
from src.detector_yolo import YOLODetector
from src.segmenter import ContourSegmenter
from src.overlay import Overlay
from src.metrics import MetricsTracker
from src.writer import VideoWriterWrapper, CSVLogger, SummaryWriter
from src.utils import resize_keep_aspect, ensure_dir


class CVPipeline:
    def __init__(self, cfg: dict, enable_yolo: bool = True):
        self.cfg = cfg
        self.pre = Preprocessor(cfg["preprocess"])
        self.seg = ContourSegmenter(cfg["segmenter"])
        self.ovl = Overlay(cfg["overlay"])
        self.metrics = MetricsTracker(cfg["output"]["metrics_path"])
        self.detector = YOLODetector(cfg["detector"]) if enable_yolo else None
        self.enable_yolo = enable_yolo

    def process_frame(self, frame: np.ndarray, frame_idx: int):
        t0 = time.perf_counter()

        # 1) Preprocess
        gray, blurred, mask = self.pre.process(frame)

        # 2) Contour segmentation
        contours = self.seg.find_contours(mask)
        self.seg.draw_contours(frame, contours, color=(0, 200, 255))

        # 3) YOLO detection
        detections = []
        if self.detector is not None:
            detections = self.detector.infer(frame)
            self.ovl.draw_detections(frame, detections)

        # 4) Overlay counts
        if self.ovl.show_counts:
            self.ovl.draw_counts(frame, detections)

        # 5) Metrics + FPS overlay
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        self.ovl.draw_fps(frame, fps)

        mean_conf = (float(np.mean([d["conf"] for d in detections]))
                     if detections else 0.0)
        class_counts = Counter(d["cls_name"] for d in detections)

        row = {
            "frame": frame_idx,
            "timestamp_s": round(time.time(), 4),
            "latency_ms": round(latency_ms, 3),
            "fps": round(fps, 2),
            "detections": len(detections),
            "contours": len(contours),
            "mean_conf": round(mean_conf, 4),
            "classes_json": json.dumps(dict(class_counts)),
            "resolution_w": frame.shape[1],
            "resolution_h": frame.shape[0],
        }
        self.metrics.log_frame(latency_ms, len(detections),
                               len(contours), mean_conf)
        return frame, mask, contours, detections, row

    def run(self, source=0, max_frames=None, save_output=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Video properties
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_size = (w, h)

        # Output writers
        video_writer = None
        mask_writer = None
        csv_logger = None

        if save_output:
            out_path = self.cfg["output"]["output_path"]
            video_writer = VideoWriterWrapper(out_path, fps_in, target_size)

            mask_path = str(Path(out_path).with_name("mask_preview.mp4"))
            mask_writer = VideoWriterWrapper(mask_path, fps_in, target_size)

            csv_path = self.cfg["output"]["metrics_path"]
            csv_logger = CSVLogger(csv_path)

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = resize_keep_aspect(
                    frame,
                    self.cfg["video"]["width"],
                    self.cfg["video"]["height"],
                )

                frame, mask, contours, detections, row = self.process_frame(
                    frame, idx
                )

                if video_writer:
                    video_writer.write(frame)
                if mask_writer:
                    mask_writer.write(mask)
                if csv_logger:
                    csv_logger.log(row)

                cv2.imshow("Detection", frame)
                cv2.imshow("Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                idx += 1
                if max_frames and idx >= max_frames:
                    break
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if mask_writer:
                mask_writer.release()
            if csv_logger:
                csv_logger.close()
            cv2.destroyAllWindows()

        # Persist aggregated summary
        summary = self.metrics.summary()
        summary["frames_written"] = idx
        summary["output_video"] = self.cfg["output"]["output_path"] if save_output else None
        summary["metrics_csv"] = self.cfg["output"]["metrics_path"] if save_output else None

        if save_output:
            summary_path = str(
                Path(self.cfg["output"]["metrics_path"]).with_name(
                    "metrics_summary.json"
                )
            )
            SummaryWriter.write(summary_path, summary)

        return summary