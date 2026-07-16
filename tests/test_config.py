"""測試: config 設定載入器"""
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import yaml

from src.config import load_config, Config


def test_load_default_config():
    """測試載入預設設定"""
    config = load_config()
    assert isinstance(config, Config)
    assert config.camera.device_id == 0
    assert config.detection.confidence_threshold == 0.5
    assert config.state_machine.stationary_alert_threshold_sec == 1500.0
    assert config.motion.method == "centroid"


def test_load_custom_config():
    """測試載入自訂設定"""
    custom_config = {
        "camera": {"device_id": 1, "width": 640, "height": 480},
        "detection": {"confidence_threshold": 0.7},
        "state_machine": {"stationary_alert_threshold_sec": 1800.0},
    }

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(custom_config, f)

        config = load_config(str(config_path))
        assert config.camera.device_id == 1
        assert config.camera.width == 640
        assert config.camera.height == 480
        assert config.detection.confidence_threshold == 0.7
        assert config.state_machine.stationary_alert_threshold_sec == 1800.0
        # 未指定的欄位應保留預設值
        assert config.camera.fps == 30
        assert config.motion.motion_duration_sec == 120.0


def test_env_override(monkeypatch):
    """測試環境變數覆蓋"""
    monkeypatch.setenv("LRM_CAMERA_DEVICE_ID", "2")
    monkeypatch.setenv("LRM_DETECTION_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("LRM_VERBOSE", "true")

    config = load_config()
    assert config.camera.device_id == 2
    assert config.detection.confidence_threshold == 0.8
    assert config.general.verbose is True