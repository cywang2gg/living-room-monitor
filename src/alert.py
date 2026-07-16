"""提醒模組 — 語音 / 桌面通知 / 文字日誌"""
import logging
import subprocess
import sys
from typing import List

from .config import AlertConfig

logger = logging.getLogger(__name__)


class Alert:
    """提醒觸發器

    支援多種提醒方式:
    - log: 寫入日誌
    - tts: 語音播放（pyttsx3 / espeak）
    - notification: 桌面通知（plyer / win10toast / notify-send）
    """

    def __init__(self, config: AlertConfig):
        self.config = config
        self._tts_engine = None
        self._notifier = None
        self._init_tts()
        self._init_notifier()

    def _init_tts(self):
        """初始化 TTS 引擎"""
        if "tts" not in self.config.methods:
            return

        try:
            if self.config.tts_engine == "pyttsx3":
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                # 設定中文語音（如果可用）
                voices = self._tts_engine.getProperty("voices")
                for voice in voices:
                    if "chinese" in voice.name.lower() or "zh" in voice.id.lower():
                        self._tts_engine.setProperty("voice", voice.id)
                        break
                logger.info("TTS 引擎初始化成功 (pyttsx3)")
        except Exception as e:
            logger.warning("TTS 初始化失敗: %s，將使用替代方案", e)
            self._tts_engine = None

    def _init_notifier(self):
        """初始化桌面通知"""
        if "notification" not in self.config.methods:
            return

        try:
            if sys.platform == "win32":
                from win10toast import ToastNotifier
                self._notifier = ToastNotifier()
                logger.info("桌面通知初始化成功 (win10toast)")
            elif sys.platform.startswith("linux"):
                # Linux 使用 notify-send，無需額外套件
                self._notifier = "notify-send"
                logger.info("桌面通知初始化成功 (notify-send)")
        except ImportError:
            logger.warning("win10toast 未安裝，桌面通知停用")
            self._notifier = None
        except Exception as e:
            logger.warning("桌面通知初始化失敗: %s", e)
            self._notifier = None

    def trigger(self, message: str):
        """觸發提醒

        Args:
            message: 提醒訊息
        """
        for method in self.config.methods:
            try:
                if method == "log":
                    self._alert_log(message)
                elif method == "tts":
                    self._alert_tts(message)
                elif method == "notification":
                    self._alert_notification(message)
                else:
                    logger.warning("未知的提醒方式: %s", method)
            except Exception as e:
                logger.error("提醒觸發失敗 (%s): %s", method, e)

    def _alert_log(self, message: str):
        """日誌提醒"""
        logger.warning("🔔 久坐提醒: %s", message)

    def _alert_tts(self, message: str):
        """語音提醒"""
        tts_message = self.config.tts_message or message

        if self._tts_engine is not None:
            try:
                self._tts_engine.say(tts_message)
                self._tts_engine.runAndWait()
                return
            except Exception as e:
                logger.debug("pyttsx3 播放失敗: %s", e)

        # 替代方案: espeak (Linux)
        if sys.platform.startswith("linux"):
            try:
                subprocess.run(
                    ["espeak", "-v", self.config.tts_language, tts_message],
                    check=False,
                    capture_output=True,
                )
                return
            except FileNotFoundError:
                pass

        logger.warning("無可用的 TTS 引擎")

    def _alert_notification(self, message: str):
        """桌面通知"""
        title = self.config.notification_title
        notif_message = self.config.notification_message or message

        if sys.platform == "win32" and self._notifier is not None:
            try:
                self._notifier.show_toast(
                    title, notif_message, duration=10, threaded=True
                )
                return
            except Exception as e:
                logger.debug("win10toast 通知失敗: %s", e)

        if sys.platform.startswith("linux") and self._notifier == "notify-send":
            try:
                subprocess.run(
                    ["notify-send", title, notif_message],
                    check=False,
                    capture_output=True,
                )
                return
            except FileNotFoundError:
                pass

        logger.warning("無可用的桌面通知方式")

    def cleanup(self):
        """清理資源"""
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
