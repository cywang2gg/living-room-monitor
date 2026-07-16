"""攝影機抽象層 — 封裝 OpenCV VideoCapture，跨平台支援"""
import sys
import time
import logging
from typing import Optional, Tuple

import cv2

from .config import CameraConfig

logger = logging.getLogger(__name__)


class Camera:
    """攝影機抽象層

    - 跨平台後端選擇（Windows: DSHOW, Linux: V4L2）
    - 自動重連機制
    - 輸出: (BGR frame, timestamp)
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend = self._resolve_backend()
        self._last_frame_time: float = 0.0

    def _resolve_backend(self) -> int:
        """根據設定與平台決定攝影機後端

        Returns:
            OpenCV 後端常數
        """
        backend_map: dict = {
            "dshow": cv2.CAP_DSHOW,
            "v4l2": cv2.CAP_V4L2,
            "any": cv2.CAP_ANY,
            "auto": cv2.CAP_ANY,
        }

        if self.config.backend != "auto":
            return backend_map.get(self.config.backend.lower(), cv2.CAP_ANY)

        # 自動偵測平台
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
        else:
            return cv2.CAP_ANY

    def open(self) -> bool:
        """開啟攝影機

        Returns:
            是否成功開啟
        """
        try:
            self._cap = cv2.VideoCapture(self.config.device_id, self._backend)
            if not self._cap.isOpened():
                logger.error("無法開啟攝影機 device_id=%d", self.config.device_id)
                return False

            # 設定解析度與 FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # 驗證實際設定
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                "攝影機已開啟: %dx%d @ %.1f FPS (backend=%s)",
                actual_w, actual_h, actual_fps,
                "DSHOW" if self._backend == cv2.CAP_DSHOW else
                "V4L2" if self._backend == cv2.CAP_V4L2 else "ANY"
            )
            return True

        except Exception as e:
            logger.error("攝影機初始化失敗: %s", e)
            return False

    def read(self) -> Tuple[Optional[cv2.typing.MatLike], float]:
        """讀取一幀

        Returns:
            (frame, timestamp)，失敗時 frame=None
        """
        if self._cap is None or not self._cap.isOpened():
            if self.config.auto_reconnect:
                if self._try_reconnect():
                    return self.read()
            return None, time.time()

        ret, frame = self._cap.read()
        ts = time.time()

        if not ret or frame is None:
            logger.warning("讀取攝影機失敗")
            if self.config.auto_reconnect:
                self._try_reconnect()
            return None, ts

        self._last_frame_time = ts
        return frame, ts

    def _try_reconnect(self) -> bool:
        """嘗試重新連接攝影機"""
        logger.info("嘗試重新連接攝影機...")
        self.release()
        time.sleep(self.config.reconnect_interval_sec)
        return self.open()

    def release(self) -> None:
        """釋放攝影機資源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.release()
