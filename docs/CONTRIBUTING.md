# Contributing Guide

Thank you for considering contributions to **CV Detector Pipeline**.

## Development Setup

```bash
git clone https://github.com/your-username/cv-detector-pipeline.git
cd cv-detector-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Coding Standards

- **PEP 8** compliance enforced via `flake8` or `ruff`.
- Type hints encouraged on all public functions.
- Docstrings in Google style for all modules and classes.
- No hard-coded paths — use `configs/default.yaml`.

## Branching Model

- `main` — stable releases only
- `dev` — integration branch
- `feature/<name>` — new features
- `fix/<name>` — bug fixes

## Pull Request Checklist

- [ ] Code passes `pytest tests/ -v`
- [ ] New features have accompanying tests
- [ ] Documentation updated (README, docs/)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No large binary files committed (videos, weights)

## Commit Message Convention

```
<type>(<scope>): <short summary>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Example:

```
feat(segmenter): add max-area filter for contour extraction

Adds configurable max_area to drop oversized background blobs.
Closes #42.
```

## Reporting Issues

Open a GitHub issue with:

- Reproduction steps
- Expected vs actual behavior
- Environment (OS, Python version, OpenCV version, GPU)
- Sample frame if applicable

## Code of Conduct

Be respectful, constructive, and inclusive.
```

## 4. Final Complete Directory Tree

```
cv-detector-pipeline/
│
├── configs/
│   ├── default.yaml
│   └── yolo_params.yaml
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── detector_yolo.py
│   ├── segmenter.py
│   ├── overlay.py
│   ├── metrics.py
│   └── utils.py
│
├── models/
│   ├── .gitkeep
│   └── yolov8n.pt          # downloaded at setup, ignored by git
│
├── data/
│   ├── videos/
│   │   └── .gitkeep
│   └── outputs/
│       └── .gitkeep
│
├── scripts/
│   ├── run_realtime.py
│   ├── run_video.py
│   └── evaluate.py
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_segmenter.py
│   └── test_metrics.py
│
├── docs/
│   ├── whitepaper.md
│   ├── whitepaper.pdf
│   ├── ARCHITECTURE.md
│   ├── METRICS.md
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   └── assets/
│       └── .gitkeep
│
├── main.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── LICENSE
└── README.md
```

