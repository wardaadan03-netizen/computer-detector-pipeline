# CV Detector Pipeline — Real-Time Object Detector & Frame-By-Frame Segmenter

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

A **hybrid real-time computer vision pipeline** that combines classical OpenCV image processing (Gaussian filtering, adaptive thresholding, contour extraction) with a deep learning object detector (YOLOv8n). The pipeline performs **frame-by-frame object detection, segmentation, class-count overlays, and performance logging** for both webcam and file-based video streams.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Metrics & Evaluation](#metrics--evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Performance Whitepaper](#performance-whitepaper)
- [Contributing](#contributing)
- [License](#license)

---

## Features

-  **Real-time inference** on webcam, video files, or RTSP streams
-  **Dual-branch architecture**: classical CV + YOLOv8 deep learning
-  **Adaptive thresholding** with Gaussian preprocessing (illumination invariant)
-  **Contour extraction & filtering** by pixel bounds (min/max area)
-  **Dynamic overlay** of class count vectors and bounding boxes
-  **Frame-by-frame metrics**: latency, FPS, IoU, mean confidence
-  **CSV logging** + matplotlib performance plots
-  **Modular & testable** codebase (pytest)
-  **Docker-ready** for reproducible deployment

---

## Architecture
┌──────────────────────────────────────────────────────────────────────┐
│ Input Frame                                                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────┐             ┌─────────────────────┐
│ Classical Branch    │             │  Deep Branch        │
│ Gaussian Blur       │             │ YOLOv8n             │
│ Adaptive Threshold  │             │ Inference           │
│ Morph Clean         │             │ NMS                 │
│ Contour Extract     │             │ Class Assignment    │
└──────────┬──────────┘             └──────────┬──────────┘
           │                                   │
            └────────────┬───────────────────┘
                         ▼
              ┌────────────────────────┐
              │ Overlay & Metrics      │
              │ - Bounding Boxes       │
              │ - Class Count Vectors  │
              │ - FPS / Latency        │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │ Output Frame + CSV     │
              └────────────────────────┘



---

## Project Structure
cv-detector-pipeline/
├── configs/ # YAML configuration files
├── src/ # Core pipeline modules
├── models/ # Pretrained weights (YOLOv8n)
├── data/ # Input videos + output artifacts
├── scripts/ # CLI entry points
├── tests/ # pytest unit tests
├── docs/ # Whitepaper, diagrams, changelog
├── main.py # Main entry point
├── requirements.txt
├── Dockerfile
├── .gitignore
├── LICENSE
└── README.md


See `docs/ARCHITECTURE.md` for a detailed module diagram.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- (Optional) CUDA 12.x + cuDNN 8.9 for GPU acceleration
- (Optional) Docker 24+

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/cv-detector-pipeline.git
cd cv-detector-pipeline

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download YOLOv8n weights
mkdir -p models
wget -O models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
Usage
Real-Time Webcam Demo
bash
python scripts/run_realtime.py
Press q to exit. Two windows appear: the annotated detection frame and the binary mask.

Process a Video File
bash
python scripts/run_video.py --video data/videos/sample.mp4 --max-frames 500
Output is written to data/outputs/annotated.mp4.

Classical-Only Mode (no YOLO)
bash
python main.py --source data/videos/sample.mp4 --no-yolo --max-frames 300
Batch Evaluation
bash
python scripts/evaluate.py
Produces:

data/outputs/metrics_eval.csv

data/outputs/performance.png

Configuration
All pipeline behavior is controlled via configs/default.yaml:

yaml
preprocess:
  gaussian_kernel: 5
  block_size: 11
  c_constant: 2

detector:
  conf_threshold: 0.35
  iou_threshold: 0.45
  device: "cpu"    # change to "cuda" for GPU

segmenter:
  min_area: 500
  max_area: 200000
Override at runtime:

bash
python main.py --source 0 --save
Metrics & Evaluation
Frame-by-frame metrics are logged to data/outputs/metrics.csv:

Column	Description
frame	Frame index
latency_ms	End-to-end pipeline time (ms)
fps	Instantaneous FPS
detections	Number of YOLO detections
contours	Number of valid contours
mean_conf	Mean YOLO confidence
Summary statistics (mean, P95, min) are printed at the end of each run.

Testing
bash
pytest tests/ -v
Coverage:

bash
pytest --cov=src tests/
Docker
Build
bash
docker build -t cv-detector-pipeline:latest .
Run (with webcam passthrough on Linux)
bash
docker run --rm -it \
  --device /dev/video0 \
  -v $(pwd)/data:/app/data \
  cv-detector-pipeline:latest
Run (video file)
bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  cv-detector-pipeline:latest \
  python main.py --source data/videos/sample.mp4 --save
Performance Whitepaper
A detailed evaluation is available in docs/whitepaper.md (and docs/whitepaper.pdf).
It covers:

Dataset descriptions

Metrics methodology

CPU vs GPU benchmarks

Latency distributions

Illumination robustness study

Future work

Contributing
Fork the repository.

Create a feature branch: git checkout -b feature/my-feature.

Commit changes with clear messages.

Add/update tests under tests/.

Open a Pull Request against main.

Please follow PEP 8 and run pytest before submitting.

License
This project is licensed under the MIT License — see LICENSE for details.

Acknowledgements
Ultralytics YOLOv8

OpenCV

Original task specification: Task 4 — Mini Project: Computer Vision Object Detector & Frame-By-Frame Segmenter Pipeline.
