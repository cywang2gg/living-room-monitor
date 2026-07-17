"""日誌與快照管理模組"""
import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import LoggingConfig
from .state_machine import State, StateSnapshot
from .utils import current_timestamp, ensure_dir

logger = logging.getLogger(__name__)


class Logger:
    """日誌與快照管理

    - 結構化日誌（JSON Lines 格式）
    - 日誌輪替
    - 快照儲存（警報時 / 人物進入時）
    - 背景寫入執行緒（避免 I/O 阻塞主迴圈）
    """

    def __init__(self, config: LoggingConfig):
        self.config = config
        self._log_dir = Path(config.log_dir)
        self._snapshot_dir = Path(config.snapshot_dir)
        self._queue: queue.Queue = queue.Queue(maxsize=100)  # 背景佇列上限
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        self._current_log_file: Optional[Path] = None
        self._last_log_date: Optional[str] = None

    def start(self) -> None:
        """啟動背景寫入執行緒"""
        ensure_dir(self._log_dir)
        ensure_dir(self._snapshot_dir)
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="logger-writer"
        )
        self._writer_thread.start()
        logger.info("日誌寫入執行緒已啟動")

    def stop(self) -> None:
        """停止背景寫入執行緒"""
        self._running = False
        # 傳送哨兵值
        self._queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        logger.info("日誌寫入執行緒已停止")

    def log(self, snapshot: StateSnapshot, timestamp: Optional[str] = None) -> None:
        """記錄狀態日誌

        Args:
            snapshot: 狀態快照
            timestamp: ISO 8601 時間戳
        """
        if timestamp is None:
            timestamp = current_timestamp()

        entry = {
            "timestamp": timestamp,
            "state": snapshot.state.value,
            "person_present": snapshot.person_present,
            "person_id": snapshot.person_id,
            "stationary_timer": round(snapshot.stationary_timer, 2),
            "moving_timer": round(snapshot.moving_timer, 2),
            "grace_timer": round(snapshot.grace_timer, 2),
            "motion_score": round(snapshot.motion_score, 2),
        }

        self._queue.put(("log", entry))

    def save_snapshot(
        self,
        frame: np.ndarray,
        snapshot: StateSnapshot,
        force: bool = False,
    ) -> None:
        """條件儲存快照

        Args:
            frame: BGR 影像
            snapshot: 狀態快照
            force: 是否強制儲存
        """
        should_save = force

        if not should_save and self.config.save_snapshot_on_alert:
            should_save = snapshot.state == State.ALERTED

        if not should_save and self.config.save_snapshot_on_person_enter:
            should_save = (
                snapshot.state == State.STATIONARY
                and snapshot.stationary_timer < 1.0
            )

        if not should_save:
            return

        # 生成檔名: YYYYMMDD_HHMMSS_{person_id}_{state}.jpg
        now = datetime.now()
        filename = (
            f"{now.strftime('%Y%m%d_%H%M%S')}"
            f"_{snapshot.person_id}"
            f"_{snapshot.state.value}"
            f".{self.config.snapshot_format}"
        )

        self._queue.put(("snapshot", {
            "frame": frame.copy(),
            "path": str(self._snapshot_dir / now.strftime("%Y%m%d") / filename),
            "quality": self.config.snapshot_quality,
            "format": self.config.snapshot_format,
        }))

    def _writer_loop(self) -> None:
        """背景寫入執行緒主迴圈"""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break

                msg_type, data = item
                if msg_type == "log":
                    self._write_log(data)
                elif msg_type == "snapshot":
                    self._write_snapshot(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("日誌寫入錯誤: %s", e)

        # 處理剩餘佇列
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is None:
                    continue
                msg_type, data = item
                if msg_type == "log":
                    self._write_log(data)
                elif msg_type == "snapshot":
                    self._write_snapshot(data)
            except Exception:
                break

    def _write_log(self, entry: dict) -> None:
        """寫入日誌"""
        # 檢查是否需要切換日誌檔案（按日期）
        today = datetime.now().strftime("%Y%m%d")
        if today != self._last_log_date:
            self._current_log_file = self._log_dir / f"{today}.jsonl"
            self._last_log_date = today
            self._cleanup_old_logs()

        if self.config.log_format == "json":
            line = json.dumps(entry, ensure_ascii=False)
        else:
            line = (
                f"[{entry['timestamp']}] "
                f"state={entry['state']} "
                f"person={entry['person_id']} "
                f"stationary={entry['stationary_timer']}s "
                f"motion={entry['motion_score']}"
            )

        try:
            with open(self._current_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error("日誌寫入失敗: %s", e)

    def _write_snapshot(self, data: dict) -> None:
        """寫入快照"""
        frame = data["frame"]
        path = Path(data["path"])
        quality = data.get("quality", 85)

        ensure_dir(path.parent)

        try:
            params = []
            if data.get("format") == "jpg":
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif data.get("format") == "png":
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 12]

            cv2.imwrite(str(path), frame, params)
            logger.debug("快照已儲存: %s", path)
        except Exception as e:
            logger.error("快照儲存失敗: %s", e)

    def _cleanup_old_logs(self) -> None:
        """清理過期日誌"""
        if self.config.log_rotation_days <= 0:
            return

        try:
            cutoff = datetime.now().timestamp() - (
                self.config.log_rotation_days * 86400
            )
            for log_file in self._log_dir.glob("*.jsonl"):
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    logger.debug("已刪除過期日誌: %s", log_file)
        except Exception as e:
            logger.error("日誌清理失敗: %s", e)
