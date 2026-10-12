# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-01-15

### Added
- Real-time pipeline with YOLOv8n detection
- Classical preprocessing (Gaussian + adaptive threshold)
- Contour extraction with area filtering
- Dynamic class-count overlays
- Frame-by-frame metrics (latency, FPS, detections, confidence)
- CSV logging and matplotlib performance plots
- Unit tests for preprocess and metrics modules
- Dockerfile and requirements.txt
- Full documentation: README, whitepaper, architecture, metrics

### Known Issues
- CPU-only deployment limited to ~35 FPS at 720p
- No temporal tracker by default
- Adaptive threshold requires per-environment tuning

## [Unreleased]

### Planned
- Kalman-filter tracker integration
- INT8 quantization for embedded deployment
- FastSAM lightweight segmentation head
- mAP@0.5:0.95 evaluation on labeled subset