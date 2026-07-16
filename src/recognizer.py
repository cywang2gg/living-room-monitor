"""臉部身份辨識模組 — 使用 face_recognition"""
import logging
import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import RecognitionConfig
from .detector import Detection

logger = logging.getLogger(__name__)


class Recognizer:
    """臉部身份辨識

    - 載入 known_faces/ 中的已知臉部編碼
    - 使用 face_recognition 比對
    - 辨識快取：每 N 幀才執行完整辨識
    """

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str] = []
        self._frame_count = 0
        self._cache: OrderedDict[int, Tuple[str, float]] = OrderedDict()  # track_id → (name, confidence)
        self._MAX_CACHE_SIZE = 100
        self._face_recognition = None

    def load_known_faces(self) -> bool:
        """載入已知臉部編碼

        優先嘗試從 pickle 快取載入，否則從圖片目錄重新編碼

        Returns:
            是否成功載入
        """
        if not self.config.enabled:
            logger.info("臉部辨識已停用")
            return True

        try:
            import face_recognition
            self._face_recognition = face_recognition
        except ImportError:
            logger.warning("face_recognition 未安裝，臉部辨識功能停用")
            self.config.enabled = False
            return True

        # 嘗試從 pickle 快取載入
        cache_path = Path(self.config.encoding_cache)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                self._known_encodings = data.get("encodings", [])
                self._known_names = data.get("names", [])
                logger.info("從快取載入 %d 個已知臉部", len(self._known_names))
                return True
            except Exception as e:
                logger.warning("快取載入失敗，將從圖片重新編碼: %s", e)

        # 從圖片目錄載入
        faces_dir = Path(self.config.known_faces_dir)
        if not faces_dir.exists():
            logger.warning("已知臉部目錄不存在: %s", faces_dir)
            return True

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_path in sorted(faces_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            if img_path.name.startswith("."):
                continue

            try:
                image = self._face_recognition.load_image_file(str(img_path))
                encodings = self._face_recognition.face_encodings(image)
                if encodings:
                    name = img_path.stem  # 檔名即為人物 ID
                    self._known_encodings.append(encodings[0])
                    self._known_names.append(name)
                    logger.info("載入臉部: %s", name)
                else:
                    logger.warning("圖片中未偵測到臉部: %s", img_path.name)
            except Exception as e:
                logger.error("載入臉部圖片失敗 %s: %s", img_path.name, e)

        # 儲存快取
        if self._known_encodings:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "encodings": self._known_encodings,
                        "names": self._known_names,
                    }, f)
                logger.info("已儲存臉部編碼快取: %s", cache_path)
            except Exception as e:
                logger.warning("快取儲存失敗: %s", e)

        logger.info("共載入 %d 個已知臉部", len(self._known_names))
        return True

    def identify(self, frame: np.ndarray, detections: List[Detection],
                 track_ids: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        """辨識偵測到的人物身份

        Args:
            frame: BGR 影像
            detections: 偵測結果列表
            track_ids: 對應的追蹤 ID（用於快取）

        Returns:
            每個偵測結果對應的 (person_id, confidence) 列表
        """
        if not self.config.enabled or self._face_recognition is None:
            return [("unknown", 0.0)] * len(detections)

        if not self._known_encodings:
            return [("unknown", 0.0)] * len(detections)

        self._frame_count += 1
        results = []

        # 只在指定間隔執行完整辨識
        should_recognize = (
            self._frame_count % self.config.recognition_interval_frames == 0
            or not self._cache
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if should_recognize else None

        for i, det in enumerate(detections):
            track_id = track_ids[i] if track_ids and i < len(track_ids) else i

            # 檢查快取
            if not should_recognize and track_id in self._cache:
                results.append(self._cache[track_id])
                continue

            # 執行臉部辨識
            person_id, confidence = "unknown", 0.0

            if rgb_frame is not None:
                try:
                    # 從 bbox 裁切臉部區域
                    face_locations = self._face_recognition.face_locations(
                        rgb_frame[det.y1:det.y2, det.x1:det.x2]
                    )
                    if face_locations:
                        # 將座標偏移回原圖
                        offset_locations = [
                            (top + det.y1, right + det.x1, bottom + det.y1, left + det.x1)
                            for top, right, bottom, left in face_locations
                        ]
                        encodings = self._face_recognition.face_encodings(
                            rgb_frame, offset_locations
                        )

                        if encodings:
                            matches = self._face_recognition.compare_faces(
                                self._known_encodings, encodings[0],
                                tolerance=self.config.tolerance,
                            )
                            face_distances = self._face_recognition.face_distance(
                                self._known_encodings, encodings[0]
                            )

                            if len(face_distances) > 0:
                                best_match_idx = int(np.argmin(face_distances))
                                if matches[best_match_idx]:
                                    person_id = self._known_names[best_match_idx]
                                    confidence = 1.0 - face_distances[best_match_idx]

                except Exception as e:
                    logger.debug("臉部辨識失敗: %s", e)

            # 更新快取（LRU 上限 100）
            if track_id in self._cache:
                self._cache.move_to_end(track_id)  # 移到最近使用
            else:
                if len(self._cache) >= self._MAX_CACHE_SIZE:
                    self._cache.popitem(last=False)  # 移除最早的一個
            self._cache[track_id] = (person_id, confidence)
            results.append((person_id, confidence))

        return results
