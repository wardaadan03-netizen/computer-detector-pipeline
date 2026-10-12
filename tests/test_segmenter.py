
"""
Unit tests for src.segmenter.ContourSegmenter.

Tests cover:
- Contour extraction on synthetic shapes (circle, rectangle, triangle)
- Area-based filtering (min_area / max_area boundaries)
- Edge cases: empty mask, all-white mask, all-black mask
- Contour drawing does not raise and modifies the frame in-place
- Bounding rectangle helper correctness
- Integration with Preprocessor output
"""

import numpy as np
import cv2
import pytest

from src.segmenter import ContourSegmenter


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """Default segmenter configuration used across tests."""
    return {
        "min_area": 500,
        "max_area": 200_000,
    }


@pytest.fixture
def segmenter(cfg):
    return ContourSegmenter(cfg)


@pytest.fixture
def blank_mask():
    """A 480x640 uint8 mask filled with zeros (all black)."""
    return np.zeros((480, 640), dtype=np.uint8)


@pytest.fixture
def full_mask():
    """A 480x640 uint8 mask filled with 255 (all white)."""
    return np.full((480, 640), 255, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────

def make_circle_mask(radius: int, center=(320, 240), size=(480, 640)):
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


def make_rect_mask(x, y, w, h, size=(480, 640)):
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    return mask


def make_triangle_mask(size=(480, 640)):
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    pts = np.array([[100, 100], [300, 100], [200, 300]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────

class TestContourSegmenterInit:
    def test_init_stores_bounds(self, segmenter, cfg):
        assert segmenter.min_area == cfg["min_area"]
        assert segmenter.max_area == cfg["max_area"]

    def test_init_accepts_zero_min_area(self):
        seg = ContourSegmenter({"min_area": 0, "max_area": 100})
        assert seg.min_area == 0

    def test_init_accepts_zero_max_area(self):
        seg = ContourSegmenter({"min_area": 0, "max_area": 0})
        assert seg.max_area == 0


# ─────────────────────────────────────────────────────────────
# find_contours — happy path
# ─────────────────────────────────────────────────────────────

class TestFindContoursHappyPath:
    def test_single_circle_detected(self, segmenter):
        mask = make_circle_mask(radius=40)  # area ≈ 5026 px
        contours = segmenter.find_contours(mask)
        assert len(contours) == 1
        assert cv2.contourArea(contours[0]) > 0

    def test_single_rectangle_detected(self, segmenter):
        mask = make_rect_mask(100, 100, 60, 80)  # area = 4800 px
        contours = segmenter.find_contours(mask)
        assert len(contours) == 1
        assert abs(cv2.contourArea(contours[0]) - 4800) < 200

    def test_triangle_detected(self, segmenter):
        mask = make_triangle_mask()
        contours = segmenter.find_contours(mask)
        assert len(contours) == 1
        # Triangle area = 0.5 * base * height = 0.5 * 200 * 200 = 20000
        assert cv2.contourArea(contours[0]) > 15_000

    def test_multiple_disjoint_shapes(self, segmenter):
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 30, 255, -1)     # ~2827 px
        cv2.circle(mask, (300, 100), 30, 255, -1)
        cv2.circle(mask, (500, 100), 30, 255, -1)
        contours = segmenter.find_contours(mask)
        assert len(contours) == 3

    def test_nested_shapes_counted_as_two(self, segmenter):
        """RETR_EXTERNAL returns only outer contours."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 120, 255, -1)   # outer
        cv2.circle(mask, (320, 240), 60, 0, -1)      # hole
        contours = segmenter.find_contours(mask)
        # RETR_EXTERNAL → only 1 outer contour
        assert len(contours) == 1


# ─────────────────────────────────────────────────────────────
# find_contours — area filtering
# ─────────────────────────────────────────────────────────────

class TestAreaFiltering:
    def test_shape_below_min_area_rejected(self, cfg):
        seg = ContourSegmenter({"min_area": 10_000, "max_area": 200_000})
        mask = make_rect_mask(100, 100, 50, 50)  # 2500 px < 10000
        assert seg.find_contours(mask) == []

    def test_shape_above_max_area_rejected(self, cfg):
        seg = ContourSegmenter({"min_area": 500, "max_area": 1_000})
        mask = make_rect_mask(100, 100, 200, 200)  # 40000 > 1000
        assert seg.find_contours(mask) == []

    def test_shape_within_bounds_accepted(self, cfg):
        seg = ContourSegmenter({"min_area": 100, "max_area": 100_000})
        mask = make_rect_mask(100, 100, 100, 100)  # 10000 px
        assert len(seg.find_contours(mask)) == 1

    def test_boundary_exact_min_area_inclusive(self):
        seg = ContourSegmenter({"min_area": 2500, "max_area": 200_000})
        mask = make_rect_mask(100, 100, 50, 50)  # exactly 2500 px
        contours = seg.find_contours(mask)
        # >= min_area should pass
        assert len(contours) == 1

    def test_boundary_exact_max_area_inclusive(self):
        seg = ContourSegmenter({"min_area": 500, "max_area": 2500})
        mask = make_rect_mask(100, 100, 50, 50)  # exactly 2500 px
        contours = seg.find_contours(mask)
        # <= max_area should pass
        assert len(contours) == 1

    def test_mixed_sizes_only_valid_kept(self):
        seg = ContourSegmenter({"min_area": 2_000, "max_area": 20_000})
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)       # 400 px → reject
        cv2.rectangle(mask, (100, 100), (160, 160), 255, -1)   # 3600 px → keep
        cv2.rectangle(mask, (300, 300), (500, 400), 255, -1)   # 20000 px → keep
        contours = seg.find_contours(mask)
        assert len(contours) == 2


# ─────────────────────────────────────────────────────────────
# find_contours — edge cases
# ─────────────────────────────────────────────────────────────

class TestFindContoursEdgeCases:
    def test_blank_mask_returns_empty(self, segmenter, blank_mask):
        assert segmenter.find_contours(blank_mask) == []

    def test_all_white_mask_returns_one_contour(self, segmenter, full_mask):
        """A fully white 480x640 image produces one large rectangular contour."""
        seg = ContourSegmenter({"min_area": 100, "max_area": 1_000_000})
        contours = seg.find_contours(full_mask)
        assert len(contours) == 1

    def test_zero_area_bounds_rejects_everything(self):
        seg = ContourSegmenter({"min_area": 0, "max_area": 0})
        mask = make_rect_mask(100, 100, 50, 50)
        assert seg.find_contours(mask) == []

    def test_single_pixel_rejected_by_default_min_area(self, segmenter):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        assert segmenter.find_contours(mask) == []

    def test_non_binary_mask_accepted(self, segmenter):
        """Grayscale values above 0 are treated as foreground by findContours."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (200, 200), 128, -1)  # gray 128
        contours = segmenter.find_contours(mask)
        assert len(contours) == 1

    def test_large_frame_performance(self, segmenter):
        mask = np.zeros((2160, 3840), dtype=np.uint8)
        cv2.circle(mask, (1920, 1080), 400, 255, -1)
        contours = segmenter.find_contours(mask)
        assert len(contours) == 1

    def test_returns_list_type(self, segmenter, blank_mask):
        result = segmenter.find_contours(blank_mask)
        assert isinstance(result, list)

    def test_returns_tuple_of_arrays(self, segmenter):
        mask = make_rect_mask(100, 100, 60, 60)
        contours = segmenter.find_contours(mask)
        for c in contours:
            assert isinstance(c, np.ndarray)
            assert c.ndim == 3
            assert c.shape[1] == 1  # single channel of points
            assert c.shape[2] == 2  # (x, y) coordinates


# ─────────────────────────────────────────────────────────────
# draw_contours
# ─────────────────────────────────────────────────────────────

class TestDrawContours:
    def test_draw_contours_in_place(self, segmenter):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = make_rect_mask(100, 100, 100, 100)
        contours = segmenter.find_contours(mask)
        before = frame.copy()
        segmenter.draw_contours(frame, contours)
        # Frame must have changed (non-zero pixels added)
        assert not np.array_equal(frame, before)

    def test_draw_contours_empty_list_no_error(self, segmenter):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        segmenter.draw_contours(frame, [])  # should not raise
        assert np.all(frame == 0)

    def test_draw_contours_uses_custom_color(self, segmenter):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = make_rect_mask(100, 100, 100, 100)
        contours = segmenter.find_contours(mask)
        segmenter.draw_contours(frame, contours, color=(0, 0, 255))
        # BGR red should appear somewhere
        assert np.any(np.all(frame == [0, 0, 255], axis=-1))

    def test_draw_bounding_rectangles_added(self, segmenter):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = make_rect_mask(100, 100, 100, 100)
        contours = segmenter.find_contours(mask)
        segmenter.draw_contours(frame, contours, color=(0, 255, 0))
        # Blue rectangle color (255, 0, 0) should exist since draw_contours
        # draws bbox with hardcoded (255, 0, 0)
        assert np.any(np.all(frame == [255, 0, 0], axis=-1))


# ─────────────────────────────────────────────────────────────
# Integration with Preprocessor
# ─────────────────────────────────────────────────────────────

class TestSegmenterPreprocessorIntegration:
    def test_pipeline_extracts_shape_after_threshold(self):
        """Full classical-CV branch on a synthetic frame."""
        from src.preprocess import Preprocessor

        pre = Preprocessor({
            "gaussian_kernel": 5,
            "gaussian_sigma": 1.4,
            "adaptive_method": "gaussian",
            "block_size": 11,
            "c_constant": 2,
            "morph_kernel": 3,
        })
        seg = ContourSegmenter({"min_area": 1000, "max_area": 100_000})

        # Build a frame with a bright square on a dark background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)

        _, _, mask = pre.process(frame)
        contours = seg.find_contours(mask)
        assert len(contours) >= 1

    def test_no_contour_on_flat_frame(self):
        from src.preprocess import Preprocessor

        pre = Preprocessor({
            "gaussian_kernel": 5,
            "gaussian_sigma": 1.4,
            "adaptive_method": "gaussian",
            "block_size": 11,
            "c_constant": 2,
            "morph_kernel": 3,
        })
        seg = ContourSegmenter({"min_area": 500, "max_area": 200_000})

        # A constant-value frame produces no edges after adaptive threshold
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        _, _, mask = pre.process(frame)
        # Adaptive threshold on a flat image yields all-zero (or all-255)
        # depending on C; either way no meaningful contours should pass area filter
        contours = seg.find_contours(mask)
        assert len(contours) == 0