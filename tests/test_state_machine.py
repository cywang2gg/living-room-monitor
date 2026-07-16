"""測試: state_machine 久坐狀態機"""
import pytest

from src.state_machine import StateMachine, State
from src.config import StateMachineConfig


@pytest.fixture
def sm():
    """建立預設狀態機"""
    config = StateMachineConfig(
        stationary_alert_threshold_sec=1500.0,
        reset_on_moving_duration_sec=120.0,
        grace_period_sec=10.0,
        alert_cooldown_sec=300.0,
    )
    return StateMachine(config)


def test_initial_state_idle(sm):
    """初始狀態應為 IDLE"""
    assert sm.state == State.IDLE
    assert sm.current_person_id is None


def test_person_enters_transitions_to_stationary(sm):
    """人物進入應轉換為 STATIONARY"""
    sm.update(person_present=True, person_id="Alice", dt=0.033)
    assert sm.state == State.STATIONARY
    assert sm.current_person_id == "Alice"
    assert sm.stationary_timer == 0.0


def test_stationary_timer_increases(sm):
    """靜止狀態下 timer 應增加"""
    sm.update(person_present=True, person_id="Alice", dt=0.033)
    initial_timer = sm.stationary_timer

    sm.update(person_present=True, person_id="Alice", dt=1.0)
    assert sm.stationary_timer > initial_timer


def test_motion_does_not_reset_timer_immediately(sm):
    """短時間運動不應重置 stationary timer"""
    sm.update(person_present=True, person_id="Alice", dt=1.0)
    sm.update(person_present=True, person_id="Alice", motion_score=50.0, dt=1.0)
    sm.update(person_present=True, person_id="Alice", motion_score=50.0, dt=1.0)
    # 運動時間還不到 2 分鐘，不應重置
    assert sm.moving_timer > 0
    assert sm.state == State.MOVING or sm.state == State.STATIONARY


def test_motion_resets_after_duration(sm):
    """持續運動超過 2 分鐘應重置 stationary timer"""
    sm.update(person_present=True, person_id="Alice", dt=1.0)
    sm.stationary_timer = 100.0  # 模擬已靜止一段時間

    # 持續運動 130 秒
    for _ in range(130):
        sm.update(person_present=True, person_id="Alice", motion_score=50.0, dt=1.0)

    assert sm.stationary_timer < 100.0  # 應被重置
    assert sm.state == State.MOVING


def test_person_leaves_grace_period(sm):
    """人物離開在 grace period 內不應重置 timer"""
    sm.update(person_present=True, person_id="Alice", dt=1.0)
    sm.stationary_timer = 500.0

    # 離開 5 秒（<10s grace）
    for _ in range(5):
        sm.update(person_present=False, dt=1.0)

    assert sm.current_person_id is not None  # 尚未重置
    assert sm.grace_timer == 5.0


def test_person_leaves_after_grace_period(sm):
    """人物離開超過 grace period 應重置所有 timer"""
    sm.update(person_present=True, person_id="Alice", dt=1.0)
    sm.stationary_timer = 500.0

    # 離開 15 秒（>10s grace）
    for _ in range(15):
        sm.update(person_present=False, dt=1.0)

    assert sm.current_person_id is None
    assert sm.stationary_timer == 0.0
    assert sm.state == State.IDLE


def test_alert_at_threshold(sm):
    """stationary_timer 超過門檻應觸發警報"""
    alert_called = []

    def on_alert(seconds):
        alert_called.append(seconds)

    sm.set_callbacks(on_alert=on_alert)

    # 累計到超過 1500 秒
    for _ in range(1510):
        sm.update(person_present=True, person_id="Alice", dt=1.0)

    assert sm.state == State.ALERTED
    assert len(alert_called) > 0


def test_person_enter_and_leave_cycle(sm):
    """完整的人物進入→離開循環"""
    # 進入
    sm.update(person_present=True, person_id="Bob", dt=1.0)
    assert sm.state == State.STATIONARY

    # 靜止一段時間
    for _ in range(50):
        sm.update(person_present=True, person_id="Bob", dt=1.0)
    assert sm.stationary_timer > 0

    # 離開
    for _ in range(15):
        sm.update(person_present=False, dt=1.0)
    assert sm.state == State.IDLE

    # 另一人進入
    sm.update(person_present=True, person_id="Charlie", dt=1.0)
    assert sm.state == State.STATIONARY
    assert sm.current_person_id == "Charlie"
    assert sm.stationary_timer == 0.0  # timer 已重置