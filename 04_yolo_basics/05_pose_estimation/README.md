# 05_pose_estimation - 姿态估计 🏃

## 学习目标

- 理解人体姿态估计
- 使用 YOLO Pose 模型
- 处理关键点数据

## 姿态估计概念

姿态估计检测人体的关键点（关节位置），用于：
- 动作识别
- 运动分析
- 人机交互

## COCO 关键点定义

```
0: 鼻子        1: 左眼       2: 右眼
3: 左耳        4: 右耳       5: 左肩
6: 右肩        7: 左肘       8: 右肘
9: 左腕        10: 右腕      11: 左髋
12: 右髋       13: 左膝      14: 右膝
15: 左踝       16: 右踝
```

## 核心 API

### 加载姿态模型
```python
from ultralytics import YOLO

# 姿态模型以 -pose 结尾
model = YOLO("yolo11n-pose.pt")
```

### 推理
```python
results = model("image.jpg", device="mps")
result = results[0]

# 获取关键点
keypoints = result.keypoints
```

### 访问关键点数据
```python
if result.keypoints is not None:
    # 所有人的关键点
    kpts = result.keypoints.data.cpu().numpy()
    # shape: (num_persons, 17, 3) - [x, y, confidence]
    
    for person_idx, person_kpts in enumerate(kpts):
        print(f"Person {person_idx}:")
        for kpt_idx, (x, y, conf) in enumerate(person_kpts):
            if conf > 0.5:  # 只显示置信度高的点
                print(f"  Keypoint {kpt_idx}: ({x:.1f}, {y:.1f})")
```

### 绘制骨架
```python
# 自动绘制
annotated = result.plot()

# 手动绘制
skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13],  # 腿
    [12, 13],  # 髋
    [6, 12], [7, 13],  # 躯干
    [6, 7],  # 肩
    [6, 8], [7, 9],  # 上臂
    [8, 10], [9, 11],  # 下臂
    [1, 2], [0, 1], [0, 2],  # 脸
    [1, 3], [2, 4],  # 耳
    [3, 5], [4, 6]  # 耳到肩
]

for start, end in skeleton:
    pt1 = tuple(person_kpts[start-1][:2].astype(int))
    pt2 = tuple(person_kpts[end-1][:2].astype(int))
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
```

## 实际应用

### 检测举手动作
```python
for person_kpts in kpts:
    left_wrist = person_kpts[9]   # 左腕
    left_shoulder = person_kpts[5]  # 左肩
    
    # 如果手腕高于肩膀
    if left_wrist[1] < left_shoulder[1] and left_wrist[2] > 0.5:
        print("检测到举手!")
```

## 待创建文件

- `01_pose_basic.py` - 姿态估计基础
- `02_skeleton_drawing.py` - 骨架绘制
- `03_action_recognition.py` - 动作识别

