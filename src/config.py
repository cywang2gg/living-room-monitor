"""設定載入器 — 讀取 config.yaml + 環境變數覆蓋"""
import os
import typing
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
class TTSConfig:
    """TTS 設定（巢狀結構）"""
    engine: str = "pyttsx3"
    message: str = "請起身運動5分鐘"
    language: str = "zh-TW"


@dataclass
class NotificationConfig:
    """桌面通知設定（巢狀結構）"""
    title: str = "久坐提醒"
    message: str = "你已經坐了 25 分鐘，請起身運動 5 分鐘！"


@dataclass
class AlertConfig:
    """提醒設定（支援巢狀 tts / notification）"""
    methods: List[str] = field(default_factory=lambda: ["log", "tts", "notification"])
    tts: TTSConfig = field(default_factory=TTSConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)


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


def _env_override(key: str, default) -> typing.Any:
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


def _dict_to_dataclass(cls: type, data: dict, env_prefix: str = "") -> typing.Any:
    """遞迴將 dict 轉換為 dataclass，支援環境變數覆蓋
    
    Args:
        cls: 目標 dataclass 類型
        data: 來源 dict
        env_prefix: 環境變數前綴（如 "CAMERA", "ALERT_TTS"）
    
    Returns:
        dataclass 實例
    """
    if not isinstance(data, dict):
        return data

    kwargs = {}
    for key, field_info in cls.__dataclass_fields__.items():
        # 取得預設值
        if field_info.default is not field_info.default_factory:
            default_val = field_info.default
        elif field_info.default_factory is not field_info.default_factory:
            default_val = field_info.default_factory()
        else:
            default_val = None
        
        # 從 data 取值
        raw_val = data.get(key, default_val)
        # 正確的環境變數 key：LRM_{PREFIX}_{KEY}（全大寫）
        env_key = f"{env_prefix}_{key}".upper() if env_prefix else key.upper()
        
        # 環境變數覆蓋
        raw_val = _env_override(env_key, raw_val)
        
        # 檢查是否為巢狀 dataclass
        field_type = field_info.type
        if hasattr(field_type, "__dataclass_fields__"):
            # 巢狀 dataclass — 遞迴處理，傳遞正確的 prefix
            sub_data = data.get(key, {})
            if isinstance(sub_data, dict):
                kwargs[key] = _dict_to_dataclass(field_type, sub_data, env_key)
            elif hasattr(sub_data, "__dataclass_fields__"):
                # 已經是 dataclass 實例
                kwargs[key] = sub_data
            else:
                kwargs[key] = _dict_to_dataclass(field_type, {}, env_key)
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
