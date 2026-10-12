"""Shared pytest fixtures for the CV Detector test suite."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def default_segmenter_cfg():
    """Segmenter config shared across the test session."""
    return {"min_area": 500, "max_area": 200_000}


@pytest.fixture
def blank_mask():
    """480x640 all-black binary mask."""
    return np.zeros((480, 640), dtype=np.uint8)


@pytest.fixture
def full_mask():
    """480x640 all-white binary mask."""
    return np.full((480, 640), 255, dtype=np.uint8)


@pytest.fixture
def random_frame():
    """Random BGR frame for preprocessing tests."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)