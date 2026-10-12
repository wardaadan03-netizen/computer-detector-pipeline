from src.metrics import MetricsTracker


def test_iou_identical():
    m = MetricsTracker()
    assert m.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint():
    m = MetricsTracker()
    assert m.iou([0, 0, 10, 10], [20, 20, 30, 30]) < 0.01