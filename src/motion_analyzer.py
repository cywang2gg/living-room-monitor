"""動作量分析模組 — 光流 / 關鍵點位移"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import MotionConfig
from .detector import Detection
from .utils import euclidean_distance, iou

logger = logging.getLogger(__name__)


class MotionState(Enum):
    """動作狀態"""
    MOVING = "moving"
    STATIONARY = "stationary"


@dataclass
class MotionResult:
    """動作分析結果"""
    score: float
    state: MotionState
    centroid_displacement: float = 0.0
    area_change: float = 0.0
    flow_magnitude: float = 0.0


class MotionAnalyzer:
    """動作量分析

    方法 A（輕量）: bbox 中心位移 + 面積變化率
    方法 B（精確）: OpenCV 稀疏光流
    """

    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self._prev_detections: List[Detection] = []
        self._prev_gray: Optional[np.ndarray] = None
        # 權重
        self._w1: float = 1.0  # centroid displacement
        self._w2: float = 0.5  # area change
        self._w3: float = 0.0 if config.method == "centroid" else 1.0  # flow magnitude

    def analyze(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> MotionResult:
        """分析動作量

        Args:
            frame: BGR 影像
            detections: 當前幀偵測結果

        Returns:
            MotionResult 包含分數與狀態
        """
        if not detections:
            self._prev_detections = []
            self._prev_gray = None
            return MotionResult(score=0.0, state=MotionState.STATIONARY)

        if not self._prev_detections:
            # 第一幀，無法比較
            self._prev_detections = detections
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.config.method == "optical_flow" else None
            return MotionResult(score=0.0, state=MotionState.STATIONARY)

        # 配對當前與前一幀的偵測（IoU matching）
        matched_pairs = self._match_detections(self._prev_detections, detections)

        if not matched_pairs:
            # 沒有 IoU 配對時，改用手邊最近的 bbox 配對
            # 確保人物在畫面中時仍能計算動作量
            if len(self._prev_detections) == 1 and len(detections) == 1:
                matched_pairs = [(self._prev_detections[0], detections[0])]
                logger.debug("使用 fallback 配對（無 IoU 重疊）")
            else:
                self._prev_detections = detections
                self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.config.method == "optical_flow" else None
                return MotionResult(score=0.0, state=MotionState.STATIONARY)

        # 計算各指標
        centroid_disp = self._calc_centroid_displacement(matched_pairs)
        area_change = self._calc_area_change(matched_pairs)
        flow_mag = 0.0

        if self.config.method == "optical_flow" and self._prev_gray is not None:
            flow_mag = self._calc_optical_flow(frame, matched_pairs)

        # 加權分數
        score = (
            self._w1 * centroid_disp
            + self._w2 * area_change * 100  # 面積變化放大
            + self._w3 * flow_mag
        )

        # 判定狀態
        state = MotionState.MOVING if score > self.config.centroid_displacement_threshold else MotionState.STATIONARY

        # 更新前一幀
        self._prev_detections = detections
        if self.config.method == "optical_flow":
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return MotionResult(
            score=score,
            state=state,
            centroid_displacement=centroid_disp,
            area_change=area_change,
            flow_magnitude=flow_mag,
        )

    def _match_detections(
        self,
        prev: List[Detection],
        curr: List[Detection],
        iou_threshold: float = 0.3,
    ) -> List[Tuple[Detection, Detection]]:
        """用 IoU 配對前後幀的偵測結果"""
        pairs = []
        used_curr = set()

        for p_det in prev:
            best_iou = 0.0
            best_idx = -1
            for j, c_det in enumerate(curr):
                if j in used_curr:
                    continue
                iou_val = iou(p_det.bbox, c_det.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:
                pairs.append((p_det, curr[best_idx]))
                used_curr.add(best_idx)

        return pairs

    def _calc_centroid_displacement(
        self, pairs: List[Tuple[Detection, Detection]]
    ) -> float:
        """計算配對偵測的平均中心位移"""
        if not pairs:
            return 0.0
        displacements = [
            euclidean_distance(p.bbox_center, c.bbox_center)
            for p, c in pairs
        ]
        return sum(displacements) / len(displacements)

    def _calc_area_change(
        self, pairs: List[Tuple[Detection, Detection]]
    ) -> float:
        """計算配對偵測的平均面積變化率"""
        if not pairs:
            return 0.0
        changes = []
        for p, c in pairs:
            if p.bbox_area > 0:
                changes.append(abs(c.bbox_area - p.bbox_area) / p.bbox_area)
            else:
                changes.append(0.0)
        return sum(changes) / len(changes)

    def _calc_optical_flow(
        self,
        frame: np.ndarray,
        pairs: List[Tuple[Detection, Detection]],
    ) -> float:
        """計算稀疏光流平均向量長度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 取最大 bbox 作為 ROI
        all_dets = [p for p, _ in pairs] + [c for _, c in pairs]
        if not all_dets:
            return 0.0

        x1 = min(d.x1 for d in all_dets)
        y1 = min(d.y1 for d in all_dets)
        x2 = max(d.x2 for d in all_dets)
        y2 = max(d.y2 for d in all_dets)

        # 特徵點偵測
        roi_prev = self._prev_gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(
            roi_prev, maxCorners=100, qualityLevel=0.01, minDistance=10
        )

        if corners is None:
            return 0.0

        # 偏移座標到原圖
        corners[:, 0, 0] += x1
        corners[:, 0, 1] += y1

        # 計算光流
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners.astype(np.float32), None
        )

        if next_pts is None or status is None:
            return 0.0

        # 計算平均位移
        valid = status.flatten() == 1
        if not valid.any():
            return 0.0

        displacements = np.sqrt(
            np.sum((next_pts[valid] - corners[valid]) ** 2, axis=2)
        )
        return float(np.mean(displacements))
