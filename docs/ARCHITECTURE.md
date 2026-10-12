# Architecture — Module Reference

## High-Level View
main.py
└── src.pipeline.CVPipeline
├── src.preprocess.Preprocessor
├── src.detector_yolo.YOLODetector
├── src.segmenter.ContourSegmenter
├── src.overlay.Overlay
└── src.metrics.MetricsTracker


## Module Responsibilities

| Module                  | Responsibility                                     |
|-------------------------|----------------------------------------------------|
| `src/preprocess.py`     | Grayscale, Gaussian blur, adaptive threshold, morph clean |
| `src/detector_yolo.py`  | Load YOLOv8, run inference, return structured detections |
| `src/segmenter.py`      | Extract & filter contours by pixel area            |
| `src/overlay.py`        | Draw boxes, labels, FPS, class-count vectors       |
| `src/metrics.py`        | Per-frame latency, FPS, IoU, CSV logging           |
| `src/pipeline.py`       | Orchestrate all above per frame                    |
| `src/utils.py`          | Config loading, path helpers, resizing             |

## Data Flow

1. Frame captured (webcam/video).
2. `Preprocessor.process()` returns `(gray, blurred, mask)`.
3. `ContourSegmenter.find_contours()` filters by area.
4. `YOLODetector.infer()` returns list of detections.
5. `Overlay.draw_detections()` and `Overlay.draw_counts()` render overlays.
6. `MetricsTracker.log_frame()` records latency, detections, confidence.
7. Optional `cv2.VideoWriter` saves annotated output.

## Threading Model

Single-threaded by design for reproducibility. Optional concurrency:

- Detector on dedicated thread with a frame queue.
- Preprocessor on main thread.

## Failure Modes & Mitigation

| Failure                          | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Camera not opened                | Explicit `RuntimeError` in `CVPipeline.run` |
| YOLO weights missing             | Fail fast at `YOLODetector.__init__`        |
| Illumination shift               | Adaptive threshold (Gaussian)               |
| Overlapping contours             | YOLO boxes drawn independently              |