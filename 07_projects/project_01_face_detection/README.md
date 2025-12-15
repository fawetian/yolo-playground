# 项目 01 - 人脸检测系统 👤

## 项目概述

构建一个实时人脸检测系统，支持：
- 摄像头实时检测
- 图片批量处理
- 人脸计数

## 难度等级

⭐⭐ 入门级

## 预计时间

3 天

## 技术栈

- OpenCV: 视频捕获、图像处理
- YOLO: 人脸检测
- Python: 业务逻辑

## 功能需求

### 基础功能
- [ ] 实时摄像头人脸检测
- [ ] 显示检测框和置信度
- [ ] 统计画面中的人脸数量

### 进阶功能
- [ ] 人脸区域截取保存
- [ ] 人脸模糊/马赛克
- [ ] 检测结果录制

## 实现思路

### 1. 选择模型
```python
# 方案 A: 使用通用 YOLO (检测 person)
model = YOLO("yolo11n.pt")

# 方案 B: 使用人脸专用模型
# 可以在 Ultralytics HUB 或网上找到人脸检测模型
```

### 2. 检测流程
```python
# 伪代码
while True:
    frame = camera.read()
    faces = model.detect(frame, classes=['person'])
    
    for face in faces:
        draw_box(frame, face)
        face_count += 1
    
    display(frame)
```

### 3. 人脸模糊
```python
for face in faces:
    x1, y1, x2, y2 = face.bbox
    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
    frame[y1:y2, x1:x2] = blurred
```

## 目录结构

```
project_01_face_detection/
├── README.md
├── main.py              # 主程序
├── src/
│   ├── detector.py      # 检测模块
│   └── visualizer.py    # 可视化模块
├── outputs/             # 输出结果
└── test_images/         # 测试图片
```

## 运行方式

```bash
conda activate yolo
python main.py
```

## 扩展想法

- 添加人脸识别（识别是谁）
- 添加表情识别
- 生成考勤报告

