"""設定載入器 — 讀取 config.yaml + 環境變數覆蓋"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class CameraConfig:
    device_id: int = 0
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: int = 30
    auto_reconnect: bool = True
    reconnect_interval_sec: int = 5


@dataclass
class DetectionConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    use_onnx: bool = False
    frame_skip: int = 2


@dataclass
class RecognitionConfig:
    enabled: bool = True
    known_faces_dir: str = "known_faces/"
    encoding_cache: str = "known_faces/encodings.pickle"
    recognition_interval_frames: int = 30
    tolerance: float = 0.5


@dataclass
class MotionConfig:
    method: str = "centroid"
    centroid_displacement_threshold: float = 30.0
    area_change_threshold: float = 0.15
    flow_magnitude_threshold: float = 5.0
    motion_duration_sec: float = 120.0


@dataclass
class StateMachineConfig:
    stationary_alert_threshold_sec: float = 1500.0
    reset_on_moving_duration_sec: float = 120.0
    grace_period_sec: float = 10.0
    alert_cooldown_sec: float = 300.0


@dataclass
class AlertConfig:
    methods: List[str] = field(default_factory=lambda: ["log", "tts", "notification"])
    tts_engine: str = "pyttsx3"
    tts_message: str = "請起身運動5分鐘"
    tts_language: str = "zh-TW"
    notification_title: str = "久坐提醒"
    notification_message: str = "你已經坐了 25 分鐘，請起身運動 5 分鐘！"


@dataclass
class LoggingConfig:
    log_dir: str = "logs/"
    log_format: str = "json"
    log_rotation_days: int = 30
    save_snapshot_on_alert: bool = True
    save_snapshot_on_person_enter: bool = False
    snapshot_dir: str = "snapshots/"
    snapshot_format: str = "jpg"
    snapshot_quality: int = 85


@dataclass
class GeneralConfig:
    frame_buffer_size: int = 10
    verbose: bool = False


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    project_root: Path = field(default_factory=Path.cwd)


def _env_override(key: str, default):
    """從環境變數覆蓋設定值，格式: LRM_<SECTION>_<KEY>"""
    env_key = f"LRM_{key.upper()}"
    val = os.environ.get(env_key)
    if val is None:
        return default
    # 型別轉換
    if isinstance(default, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def _dict_to_dataclass(cls, data: dict, env_prefix: str = ""):
    """遞迴將 dict 轉換為 dataclass，支援環境變數覆蓋"""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, default_val in cls.__dataclass_fields__.items():
        raw_val = data.get(key, default_val.default)
        env_key = f"{env_prefix}_{key}" if env_prefix else key

        # 環境變數覆蓋
        raw_val = _env_override(env_key, raw_val)

        # 遞迴處理巢狀 dataclass
        if hasattr(raw_val, "__dataclass_fields__"):
            sub_data = data.get(key, {})
            kwargs[key] = _dict_to_dataclass(type(raw_val), sub_data, env_key)
        else:
            kwargs[key] = raw_val

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """載入設定檔

    Args:
        config_path: 設定檔路徑，預設為專案根目錄的 config.yaml

    Returns:
        Config 物件
    """
    if config_path is None:
        config_path = os.environ.get("LRM_CONFIG_PATH", "config.yaml")

    config_path = Path(config_path)
    project_root = config_path.parent.resolve()

    # 讀取 yaml
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # 轉換為 dataclass
    config = _dict_to_dataclass(Config, raw)
    config.project_root = project_root

    # 解析相對路徑為絕對路徑
    config.recognition.known_faces_dir = str(project_root / config.recognition.known_faces_dir)
    config.recognition.encoding_cache = str(project_root / config.recognition.encoding_cache)
    config.logging.log_dir = str(project_root / config.logging.log_dir)
    config.logging.snapshot_dir = str(project_root / config.logging.snapshot_dir)

    return config
