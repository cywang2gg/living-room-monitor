"""YOLOv8 人物偵測模組"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectionConfig

logger = logging.getLogger(__name__)

# YOLO COCO class 0 = person
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """單一偵測結果"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    bbox_center: Tuple[float, float]
    bbox_area: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    """YOLOv8 人物偵測器

    - 只保留 person class (class_id=0)
    - 過濾低信心偵測
    - 支援 frame skip
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None
        self._frame_count = 0

    def load_model(self) -> bool:
        """載入 YOLOv8 模型

        Returns:
            是否成功載入
        """
        try:
            from ultralytics import YOLO

            model_path = self.config.model_path
            if self.config.use_onnx and model_path.endswith(".pt"):
                # 嘗試找 ONNX 版本
                onnx_path = model_path.replace(".pt", ".onnx")
                if Path(onnx_path).exists():
                    model_path = onnx_path
                    logger.info("使用 ONNX 模型: %s", onnx_path)

            self._model = YOLO(model_path)
            logger.info("YOLOv8 模型載入成功: %s", model_path)
            return True

        except Exception as e:
            logger.error("YOLOv8 模型載入失敗: %s", e)
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """執行人物偵測

        Args:
            frame: BGR 影像

        Returns:
            偵測到的人物列表
        """
        if self._model is None:
            logger.error("模型未載入，請先呼叫 load_model()")
            return []

        self._frame_count += 1

        # Frame skip: 只在指定間隔執行完整推理
        if self.config.frame_skip > 1 and self._frame_count % self.config.frame_skip != 0:
            return []

        try:
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if cls_id != PERSON_CLASS_ID:
                        continue
                    if conf < self.config.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    area = float((x2 - x1) * (y2 - y1))

                    detections.append(Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf,
                        bbox_center=(cx, cy),
                        bbox_area=area,
                    ))

            return detections

        except Exception as e:
            logger.error("YOLOv8 推理失敗: %s", e)
            return []
