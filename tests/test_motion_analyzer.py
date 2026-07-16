"""測試: motion_analyzer 動作量分析"""
import numpy as np
import pytest

from src.motion_analyzer import MotionAnalyzer, MotionState
from src.config import MotionConfig
from src.detector import Detection


@pytest.fixture
def analyzer():
    """建立預設動作分析器"""
    config = MotionConfig(
        method="centroid",
        centroid_displacement_threshold=30.0,
        area_change_threshold=0.15,
    )
    return MotionAnalyzer(config)


def test_no_detections_stationary(analyzer):
    """無偵測結果應回傳 stationary"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(frame, [])
    assert result.state == MotionState.STATIONARY
    assert result.score == 0.0


def test_first_frame_stationary(analyzer):
    """第一幀應回傳 stationary"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(100, 100, 200, 200, 0.9, (150.0, 150.0), 10000.0)
    ]
    result = analyzer.analyze(frame, detections)
    assert result.state == MotionState.STATIONARY
    assert result.score == 0.0


def test_stationary_detection(analyzer):
    """前後幀 bbox 位置相同應判定 stationary"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det1 = [Detection(100, 100, 200, 200, 0.9, (150.0, 150.0), 10000.0)]
    det2 = [Detection(100, 100, 200, 200, 0.9, (150.0, 150.0), 10000.0)]

    analyzer.analyze(frame, det1)  # 第一幀
    result = analyzer.analyze(frame, det2)  # 第二幀
    assert result.state == MotionState.STATIONARY


def test_large_movement_detected(analyzer):
    """大位移應判定 moving"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det1 = [Detection(100, 100, 200, 200, 0.9, (150.0, 150.0), 10000.0)]
    det2 = [Detection(300, 100, 400, 200, 0.9, (350.0, 150.0), 10000.0)]

    analyzer.analyze(frame, det1)  # 第一幀
    result = analyzer.analyze(frame, det2)  # 第二幀
    assert result.state == MotionState.MOVING
    assert result.centroid_displacement > 30.0  # 位移 200 px


def test_area_change_detected(analyzer):
    """面積大幅變化應判定 moving"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det1 = [Detection(100, 100, 200, 200, 0.9, (150.0, 150.0), 10000.0)]
    det2 = [Detection(100, 100, 300, 300, 0.9, (200.0, 200.0), 40000.0)]

    analyzer.analyze(frame, det1)  # 第一幀
    result = analyzer.analyze(frame, det2)  # 第二幀
    assert result.state == MotionState.MOVING
    assert result.area_change > 0.15