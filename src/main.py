"""系統進入點 — 初始化、事件迴圈"""
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import load_config, Config
from .camera import Camera
from .detector import Detector, Detection
from .recognizer import Recognizer
from .motion_analyzer import MotionAnalyzer, MotionState
from .state_machine import StateMachine, State
from .alert import Alert
from .logger import Logger
from .utils import iou

logger = logging.getLogger(__name__)


class LivingRoomMonitor:
    """客廳監控系統主類別

    主事件迴圈: capture → detect → recognize → analyze → update state → alert → log
    """

    def __init__(self, config: Config):
        self.config = config
        self.camera = Camera(config.camera)
        self.detector = Detector(config.detection)
        self.recognizer = Recognizer(config.recognition)
        self.motion_analyzer = MotionAnalyzer(config.motion)
        self.state_machine = StateMachine(config.state_machine)
        self.alert_module = Alert(config.alert)
        self.logger_module = Logger(config.logging)

        self._running = False
        self._prev_detections: List[Detection] = []
        self._track_ids: List[int] = []
        self._next_track_id = 0
        self._frame_count = 0

    def initialize(self) -> bool:
        """初始化所有子模組

        Returns:
            是否全部初始化成功
        """
        success = True

        # 攝影機
        if not self.camera.open():
            logger.error("攝影機初始化失敗")
            success = False

        # YOLOv8 模型
        if not self.detector.load_model():
            logger.error("YOLOv8 模型載入失敗")
            success = False

        # 臉部辨識
        if not self.recognizer.load_known_faces():
            logger.warning("臉部辨識初始化有問題，將繼續運作")

        # 狀態機回呼
        self.state_machine.set_callbacks(
            on_alert=self._on_alert,
            on_person_entered=self._on_person_entered,
            on_person_left=self._on_person_left,
        )

        # 日誌
        self.logger_module.start()

        return success

    def run(self):
        """主事件迴圈"""
        self._running = True
        last_time = time.time()

        logger.info("客廳監控系統啟動")
        logger.info("按 Ctrl+C 停止")

        try:
            while self._running:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                # 1. 擷取影像
                frame, ts = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. 人物偵測
                detections = self.detector.detect(frame)

                # 3. 人物追蹤（IoU matching）
                self._track_ids = self._update_tracks(detections)

                # 4. 臉部辨識
                person_ids = self.recognizer.identify(
                    frame, detections, self._track_ids
                )

                # 5. 動作分析
                motion_result = self.motion_analyzer.analyze(frame, detections)

                # 6. 更新狀態機
                person_present = len(detections) > 0
                primary_id = person_ids[0][0] if person_ids else "unknown"
                motion_score = motion_result.score

                self.state_machine.update(
                    person_present=person_present,
                    person_id=primary_id,
                    motion_score=motion_score,
                    dt=dt,
                    current_time=current_time,
                )

                # 7. 檢查警報（防重複觸發）
                if self.state_machine.should_alert() and not self.state_machine._alert_fired:
                    self.alert_module.trigger(self.state_machine.alert_message())
                    self.state_machine._alert_fired = True

                # 8. 日誌記錄
                snapshot = self.state_machine.get_snapshot()
                self.logger_module.log(snapshot)
                self.logger_module.save_snapshot(frame, snapshot)

                # 9. 更新前一幀偵測
                self._prev_detections = detections
                self._frame_count += 1

                # Debug 輸出
                if self.config.general.verbose and self._frame_count % 30 == 0:
                    logger.info(
                        "Frame %d: state=%s, person=%s, stationary=%.1fs, motion=%.1f",
                        self._frame_count,
                        snapshot.state.value,
                        snapshot.person_id,
                        snapshot.stationary_timer,
                        snapshot.motion_score,
                    )

        except KeyboardInterrupt:
            logger.info("收到中斷信號，正在關閉...")
        finally:
            self.shutdown()

    def _update_tracks(self, detections: List[Detection]) -> List[int]:
        """用 IoU matching 更新追蹤 ID

        Args:
            detections: 當前幀偵測結果

        Returns:
            對應的追蹤 ID 列表
        """
        if not self._prev_detections:
            # 第一幀，分配新 ID
            new_ids = list(range(self._next_track_id, self._next_track_id + len(detections)))
            self._next_track_id += len(detections)
            return new_ids

        track_ids = []
        used_prev = set()

        for det in detections:
            best_iou = 0.0
            best_idx = -1

            for j, prev_det in enumerate(self._prev_detections):
                if j in used_prev:
                    continue
                iou_val = iou(det.bbox, prev_det.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j

            if best_iou > 0.3 and best_idx >= 0 and best_idx < len(self._track_ids):
                # 繼承前一幀的 ID
                track_ids.append(self._track_ids[best_idx])
                used_prev.add(best_idx)
            else:
                # 新人物
                track_ids.append(self._next_track_id)
                self._next_track_id += 1

        return track_ids

    def _on_alert(self, stationary_seconds: float):
        """警報回呼"""
        logger.warning("久坐警報觸發！已靜止 %.1f 分鐘", stationary_seconds / 60.0)

    def _on_person_entered(self, person_id: str):
        """人物進入回呼"""
        logger.info("人物進入畫面: %s", person_id)

    def _on_person_left(self):
        """人物離開回呼"""
        logger.info("人物離開畫面")

    def shutdown(self):
        """優雅關閉"""
        self._running = False
        self.camera.release()
        self.alert_module.cleanup()
        self.logger_module.stop()
        logger.info("客廳監控系統已關閉")


def setup_logging(verbose: bool = False):
    """設定 Python logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """主進入點"""
    # 載入設定
    config = load_config()

    # 設定日誌
    setup_logging(verbose=config.general.verbose)

    # 建立並初始化監控系統
    monitor = LivingRoomMonitor(config)

    if not monitor.initialize():
        logger.error("系統初始化失敗，請檢查設定與硬體")
        sys.exit(1)

    # 處理 Ctrl+C
    def signal_handler(sig, frame):
        monitor._running = False

    signal.signal(signal.SIGINT, signal_handler)

    # 啟動主迴圈
    monitor.run()


if __name__ == "__main__":
    main()
