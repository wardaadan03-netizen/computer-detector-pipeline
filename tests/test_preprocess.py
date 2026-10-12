import numpy as np
from src.preprocess import Preprocessor
from src.utils import load_config


def test_preprocess_shapes():
    cfg = load_config()["preprocess"]
    pre = Preprocessor(cfg)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray, blurred, mask = pre.process(frame)
    assert gray.shape == (480, 640)
    assert blurred.shape == (480, 640)
    assert mask.shape == (480, 640)
    assert set(np.unique(mask)).issubset({0, 255})