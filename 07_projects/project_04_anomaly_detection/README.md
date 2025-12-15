# 项目 04 - 异常检测系统 🚨

## 项目概述

构建一个视频监控异常检测系统：
- 入侵检测
- 异常行为告警
- 事件记录

## 难度等级

⭐⭐⭐⭐ 挑战级

## 预计时间

7 天

## 技术栈

- OpenCV: 视频处理、背景建模
- YOLO: 目标检测
- 规则引擎: 异常判定

## 功能需求

### 基础功能
- [ ] 区域入侵检测
- [ ] 运动检测
- [ ] 异常告警

### 进阶功能
- [ ] 徘徊检测
- [ ] 遗留物检测
- [ ] 告警录像
- [ ] 邮件/消息通知

## 实现思路

### 1. 区域入侵检测
```python
# 定义监控区域（多边形）
roi_points = np.array([[100, 100], [400, 100], [400, 300], [100, 300]])

def check_intrusion(bbox, roi_points):
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    result = cv2.pointPolygonTest(roi_points, center, False)
    return result >= 0  # True 表示在区域内

# 检测
for box in result.boxes:
    if check_intrusion(box.xyxy[0], roi_points):
        trigger_alarm("入侵检测！")
```

### 2. 徘徊检测
```python
class LoiteringDetector:
    def __init__(self, threshold_time=30):
        self.tracks = {}  # track_id: {"first_seen": time, "in_roi": bool}
        self.threshold = threshold_time
    
    def check(self, track_id, in_roi, current_time):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"first_seen": current_time, "in_roi": in_roi}
        
        if in_roi:
            duration = current_time - self.tracks[track_id]["first_seen"]
            if duration > self.threshold:
                return True  # 徘徊告警
        else:
            # 离开区域，重置计时
            self.tracks[track_id]["first_seen"] = current_time
        
        return False
```

### 3. 告警系统
```python
import datetime
import smtplib

class AlertSystem:
    def __init__(self):
        self.last_alert_time = {}
        self.cooldown = 60  # 同类告警冷却时间（秒）
    
    def trigger(self, alert_type, frame, message):
        now = datetime.datetime.now()
        
        # 检查冷却
        if alert_type in self.last_alert_time:
            if (now - self.last_alert_time[alert_type]).seconds < self.cooldown:
                return
        
        self.last_alert_time[alert_type] = now
        
        # 保存截图
        filename = f"alert_{alert_type}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(f"outputs/{filename}", frame)
        
        # 发送通知
        self.send_notification(alert_type, message, filename)
    
    def send_notification(self, alert_type, message, image_path):
        # 实现邮件/消息通知
        print(f"[ALERT] {alert_type}: {message}")
```

## 目录结构

```
project_04_anomaly_detection/
├── README.md
├── main.py
├── src/
│   ├── detector.py          # 目标检测
│   ├── tracker.py           # 目标追踪
│   ├── anomaly_rules.py     # 异常规则
│   ├── alert_system.py      # 告警系统
│   └── video_recorder.py    # 录像模块
├── config/
│   ├── config.yaml          # 主配置
│   └── roi_config.yaml      # 监控区域配置
├── test_videos/
└── outputs/
    ├── alerts/              # 告警截图
    └── recordings/          # 告警录像
```

## 运行方式

```bash
conda activate yolo
python main.py --config config/config.yaml
```

## 扩展想法

- Web 管理界面
- 多摄像头支持
- 告警事件数据库
- 历史回放查询

