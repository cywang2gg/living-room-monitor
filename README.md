# Living Room Monitor

跨平台（Ubuntu + Windows）客廳監控系統，使用 Webcam 偵測人物、辨識身份、追蹤久坐時間並發出提醒。

## 功能

- 🎥 Webcam 即時影像擷取（跨平台支援）
- 🧍 YOLOv8 人物偵測
- 🧑 face_recognition 身份辨識
- 🏃 動作量分析（重心位移 / 光流）
- ⏱️ 久坐狀態機：25 分鐘未活動自動提醒
- 🔔 多種提醒方式：語音、桌面通知、日誌
- 📸 警報時自動快照

## 安裝

### Ubuntu

```bash
sudo apt update
sudo apt install -y cmake build-essential libgl1-mesa-glx libglib2.0-0
sudo apt install python3-dlib  # face-recognition 依賴
pip install -r requirements.txt
```

### Windows

```powershell
# 需先安裝 CMake + Visual Studio Build Tools (dlib 編譯用)
pip install dlib
pip install -r requirements.txt
```

## 使用

1. 將已知人物照片放入 `known_faces/` 目錄（檔名即為人物 ID）
2. 編輯 `config.yaml` 調整設定
3. 執行：

```bash
python -m src.main
```

## 設定

所有設定在 `config.yaml` 中，主要項目：

| 設定 | 預設值 | 說明 |
|------|--------|------|
| `camera.device_id` | 0 | 攝影機 ID |
| `detection.confidence_threshold` | 0.5 | YOLO 信心門檻 |
| `motion.motion_duration_sec` | 120 | 持續運動秒數才算 moving |
| `state_machine.stationary_alert_threshold_sec` | 1500 | 久坐警報門檻（25 分鐘） |
| `state_machine.reset_on_moving_duration_sec` | 120 | 連續運動重置門檻（2 分鐘） |
| `state_machine.grace_period_sec` | 10 | 人物短暫離幀容忍秒數 |

## 狀態機邏輯

```
IDLE → STATIONARY → MOVING → STATIONARY → ALERTED
  ↑        ↑           ↑          ↑           |
  └────────┴───────────┴──────────┴───────────┘
```

- **stationary timer** 只在兩種情況重置：
  1. 持續運動 > 2 分鐘
  2. 人物離開畫面 > 10 秒
- 25 分鐘久坐 → 觸發警報

## 授權

MIT License
