"""測試: detector 人物偵測（使用 mock）"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import Detector, Detection
from src.config import DetectionConfig


@pytest.fixture
def detector():
    """建立偵測器（不載入模型）"""
    config = DetectionConfig(
        confidence_threshold=0.5,
        iou_threshold=0.45,
        frame_skip=1,  # 每幀都偵測
    )
    return Detector(config)


def test_detect_without_model(detector):
    """模型未載入時應回傳空列表"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)
    assert results == []


def test_detection_dataclass():
    """測試 Detection dataclass"""
    det = Detection(
        x1=100, y1=100, x2=200, y2=200,
        confidence=0.9,
        bbox_center=(150.0, 150.0),
        bbox_area=10000.0,
    )
    assert det.bbox == (100, 100, 200, 200)
    assert det.confidence == 0.9
    assert det.bbox_center == (150.0, 150.0)


@patch("src.detector.YOLO")
def test_load_model_success(mock_yolo, detector):
    """模型載入成功"""
    result = detector.load_model()
    assert result is True
    assert detector._model is not None


def test_utils_iou():
    """測試 IoU 計算"""
    from src.utils import iou

    # 完全重疊
    assert iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    # 無重疊
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0

    # 部分重疊
    val = iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.1 < val < 1.0


def test_utils_euclidean_distance():
    """測試歐氏距離計算"""
    from src.utils import euclidean_distance

    assert euclidean_distance((0, 0), (3, 4)) == 5.0
    assert euclidean_distance((0, 0), (0, 0)) == 0.0