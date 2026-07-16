"""久坐狀態機 — 核心邏輯

狀態: IDLE → STATIONARY → MOVING → ALERTED
規則:
- stationary timer 只在兩種情況重置:
  1. 持續運動 > 2 分鐘 (reset_on_moving_duration_sec)
  2. 人物離開畫面 > 10 秒 (grace_period_sec)
- 25 分鐘久坐 → 觸發警報
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from .config import StateMachineConfig

logger = logging.getLogger(__name__)


class State(Enum):
    """系統狀態"""
    IDLE = "idle"
    STATIONARY = "stationary"
    MOVING = "moving"
    ALERTED = "alerted"


@dataclass
class StateSnapshot:
    """狀態快照，供 logger 使用"""
    state: State
    person_present: bool
    person_id: str
    stationary_timer: float
    moving_timer: float
    grace_timer: float
    motion_score: float


class StateMachine:
    """久坐狀態機

    核心邏輯:
    - person_present=False → grace_timer 累計，超過 grace_period_sec → 重置所有 timer → IDLE
    - person_present=True, 新人 → 重置 timer → STATIONARY
    - motion_score > threshold → moving_timer 累計
      - moving_timer >= reset_on_moving_duration_sec → 重置 stationary_timer → MOVING
    - motion_score <= threshold → moving_timer 衰減, stationary_timer 累計 → STATIONARY
    - stationary_timer >= alert_threshold → ALERTED → fire alert
    """

    def __init__(self, config: StateMachineConfig):
        self.config = config
        self.state = State.IDLE
        self.person_present = False
        self.current_person_id: Optional[str] = None
        self.stationary_timer = 0.0
        self.moving_timer = 0.0
        self.grace_timer = 0.0
        self.last_motion_score = 0.0
        self._alert_fired = False
        self._last_alert_time = 0.0

        # 回呼
        self._on_alert: Optional[Callable[[float], None]] = None
        self._on_person_entered: Optional[Callable[[str], None]] = None
        self._on_person_left: Optional[Callable[[], None]] = None

    def set_callbacks(
        self,
        on_alert: Optional[Callable[[float], None]] = None,
        on_person_entered: Optional[Callable[[str], None]] = None,
        on_person_left: Optional[Callable[[], None]] = None,
    ):
        """設定事件回呼"""
        self._on_alert = on_alert
        self._on_person_entered = on_person_entered
        self._on_person_left = on_person_left

    def update(
        self,
        person_present: bool,
        person_id: str = "unknown",
        motion_score: float = 0.0,
        dt: float = 0.033,
        current_time: float = 0.0,
    ) -> State:
        """更新狀態機（每幀呼叫）

        Args:
            person_present: 是否有人在畫面中
            person_id: 人物 ID
            motion_score: 動作量分數
            dt: 距上幀的秒數
            current_time: 當前時間戳

        Returns:
            更新後的狀態
        """
        self.last_motion_score = motion_score

        # ── 無人在畫面 ──
        if not person_present:
            if self.current_person_id is not None:
                # 人物剛離開，開始 grace 計時
                self.grace_timer += dt
                if self.grace_timer > self.config.grace_period_sec:
                    # 真的離開了，重置所有計時器
                    logger.info(
                        "人物 %s 離開畫面超過 %.0f 秒，重置計時器",
                        self.current_person_id, self.config.grace_period_sec,
                    )
                    self._reset_all_timers()
                    if self._on_person_left:
                        self._on_person_left()
                    self.current_person_id = None
                    self._transition(State.IDLE)
            else:
                # 持續無人
                self._transition(State.IDLE)
            return self.state

        # ── 有人在畫面 ──
        self.grace_timer = 0.0  # 人物在畫面中，重置 grace

        # 新人進入
        if self.current_person_id is None or (
            person_id != "unknown" and person_id != self.current_person_id
        ):
            logger.info("人物進入: %s", person_id)
            self.current_person_id = person_id
            self._reset_all_timers()
            self._alert_fired = False
            if self._on_person_entered:
                self._on_person_entered(person_id)
            self._transition(State.STATIONARY)
            return self.state

        # ── 有持續人物在場 ──
        motion_threshold = 30.0  # 預設門檻，可由 config 覆蓋

        if motion_score > motion_threshold:
            # 正在運動
            self.moving_timer += dt
            self.stationary_timer += dt  # stationary timer 繼續累計

            if self.moving_timer >= self.config.reset_on_moving_duration_sec:
                # 持續運動超過 2 分鐘 → 重置 stationary
                logger.info(
                    "持續運動 %.0f 秒，重置久坐計時器",
                    self.moving_timer,
                )
                self.stationary_timer = 0.0
                self.moving_timer = 0.0
                self._alert_fired = False
                self._transition(State.MOVING)
            else:
                if self.state != State.ALERTED:
                    self._transition(State.MOVING)
        else:
            # 靜止 / 小範圍移動
            self.moving_timer = max(0.0, self.moving_timer - dt)  # 慢慢衰減
            self.stationary_timer += dt
            if self.state != State.ALERTED:
                self._transition(State.STATIONARY)

        # ── 檢查警報 ──
        if self.stationary_timer >= self.config.stationary_alert_threshold_sec:
            if self.state != State.ALERTED and not self._alert_fired:
                self._transition(State.ALERTED)
                self._alert_fired = True
                self._last_alert_time = current_time
                if self._on_alert:
                    self._on_alert(self.stationary_timer)
                logger.warning(
                    "久坐警報！已靜止 %.0f 秒（%.1f 分鐘）",
                    self.stationary_timer,
                    self.stationary_timer / 60.0,
                )
            elif self._alert_fired and current_time - self._last_alert_time >= self.config.alert_cooldown_sec:
                # 冷卻期過後可再次觸發
                self._alert_fired = False
                self._last_alert_time = current_time
                if self._on_alert:
                    self._on_alert(self.stationary_timer)

        return self.state

    def _transition(self, new_state: State):
        """狀態轉換"""
        if self.state != new_state:
            logger.debug("狀態轉換: %s → %s", self.state.value, new_state.value)
            self.state = new_state

    def _reset_all_timers(self):
        """重置所有計時器"""
        self.stationary_timer = 0.0
        self.moving_timer = 0.0
        self.grace_timer = 0.0

    def should_alert(self) -> bool:
        """是否應觸發警報"""
        return self.state == State.ALERTED

    def alert_message(self) -> str:
        """警報訊息"""
        minutes = self.stationary_timer / 60.0
        return f"你已經坐了 {minutes:.1f} 分鐘，請起身運動 5 分鐘！"

    def get_snapshot(self) -> StateSnapshot:
        """取得狀態快照"""
        return StateSnapshot(
            state=self.state,
            person_present=self.person_present,
            person_id=self.current_person_id or "none",
            stationary_timer=self.stationary_timer,
            moving_timer=self.moving_timer,
            grace_timer=self.grace_timer,
            motion_score=self.last_motion_score,
        )
