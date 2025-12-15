# 03 - OpenCV 视频处理 🎬

## 本模块内容

学习视频捕获、处理和分析，为实时目标检测做准备。

## 子目录

| 目录 | 内容 | 预计时间 |
|-----|------|---------|
| `01_video_capture/` | 视频捕获与读取 | 2 天 |
| `02_video_processing/` | 帧处理与视频保存 | 2 天 |
| `03_optical_flow/` | 光流与运动检测 | 3 天 |

## 学习目标

完成本模块后，你将掌握：

- [ ] 从摄像头和视频文件读取帧
- [ ] 处理视频帧并保存
- [ ] 使用背景减除检测运动
- [ ] 使用光流追踪运动

## macOS 注意事项

### 摄像头权限
首次使用摄像头会请求权限，需要在：
**系统设置 → 隐私与安全性 → 摄像头** 中允许 Terminal/IDE

### 视频编码
macOS 推荐使用以下编码器：
```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4
```

## 运行示例

```bash
conda activate yolo
python 01_video_capture/01_camera_capture.py
```

## 下一步

完成后进入 `04_yolo_basics/` 开始学习 YOLO！

