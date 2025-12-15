# 项目 02 - 车辆计数系统 🚗

## 项目概述

构建一个交通监控系统，实现：
- 车辆检测与分类
- 越线计数
- 流量统计

## 难度等级

⭐⭐⭐ 进阶级

## 预计时间

5 天

## 技术栈

- OpenCV: 视频处理
- YOLO: 车辆检测
- ByteTrack/BoTSORT: 目标追踪

## 功能需求

### 基础功能
- [ ] 检测车辆（car, truck, bus, motorcycle）
- [ ] 车辆追踪（分配唯一 ID）
- [ ] 越线计数

### 进阶功能
- [ ] 双向计数（进/出）
- [ ] 分车型统计
- [ ] 生成流量报告
- [ ] 速度估算

## 实现思路

### 1. 目标追踪
```python
# YOLO 内置追踪
results = model.track(
    frame,
    persist=True,    # 保持追踪状态
    tracker="bytetrack.yaml"  # 追踪器配置
)
```

### 2. 越线检测
```python
# 定义计数线
line_start = (0, 300)
line_end = (640, 300)

# 检测目标中心是否越过线
def check_crossing(prev_y, curr_y, line_y):
    if prev_y < line_y and curr_y >= line_y:
        return "down"
    elif prev_y > line_y and curr_y <= line_y:
        return "up"
    return None
```

### 3. 状态管理
```python
class VehicleTracker:
    def __init__(self):
        self.tracks = {}  # track_id: {"positions": [], "counted": False}
    
    def update(self, track_id, position):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"positions": [], "counted": False}
        self.tracks[track_id]["positions"].append(position)
```

## 目录结构

```
project_02_vehicle_counting/
├── README.md
├── main.py
├── src/
│   ├── detector.py      # 检测模块
│   ├── tracker.py       # 追踪模块
│   ├── counter.py       # 计数模块
│   └── reporter.py      # 报告生成
├── config/
│   └── config.yaml      # 配置（计数线位置等）
├── test_videos/         # 测试视频
└── outputs/             # 输出结果
```

## 测试数据

可以使用公开的交通视频数据集：
- MOT Challenge
- UA-DETRAC

## 运行方式

```bash
conda activate yolo
python main.py --video test_videos/traffic.mp4
```

## 扩展想法

- 添加车牌识别
- 违章检测（逆行、违停）
- 拥堵分析

