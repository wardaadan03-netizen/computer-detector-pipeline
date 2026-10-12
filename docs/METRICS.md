# Metrics Definitions

## Latency (ms/frame)
## Latency (ms/frame)
latency_ms = (t_after_process - t_before_process) * 1000

text

Includes preprocessing, YOLO inference, contour extraction, overlay drawing, and CSV logging.

## Throughput (FPS)
fps = 1000 / latency_ms

text

## Mean Confidence
mean_conf = mean([det.conf for det in detections]) if detections else 0

text

## Intersection over Union (IoU)
IoU(A, B) = |A ∩ B| / |A ∪ B|

text

Used to measure temporal overlap for the same tracked object across consecutive frames.

## Detection Stability
stability = std([n_detections_frame_i for i in 1..N])

text

Lower is better. Values < 1.0 indicate stable detection.

## Contour Precision
precision = valid_contours / total_contour_candidates

text

A contour is "valid" if its area is within `[min_area, max_area]`.

## CSV Output Schema

| Column       | Type    | Description                       |
|--------------|---------|-----------------------------------|
| `frame`      | int     | Frame index (1-based)             |
| `latency_ms` | float   | End-to-end pipeline time          |
| `fps`        | float   | Instantaneous throughput          |
| `detections` | int     | Number of YOLO detections         |
| `contours`   | int     | Number of valid contours          |
| `mean_conf`  | float   | Mean YOLO confidence              |
