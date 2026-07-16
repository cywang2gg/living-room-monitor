"""通用工具函數"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple


def current_timestamp() -> str:
    """回傳 ISO 8601 格式時間戳（含時區）"""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    """確保目錄存在，不存在則建立"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """計算兩個 bbox 的 Intersection over Union

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU 值，0.0 ~ 1.0
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """計算兩點的歐氏距離"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
