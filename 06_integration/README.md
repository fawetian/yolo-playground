# 06 - 综合应用 🔗

## 本模块内容

将 OpenCV 和 YOLO 结合，构建完整的视觉应用。

## 子目录

| 目录 | 内容 | 预计时间 |
|-----|------|---------|
| `01_opencv_yolo/` | OpenCV + YOLO 集成 | 2 天 |
| `02_realtime_detection/` | 实时检测系统 | 3 天 |
| `03_video_analysis/` | 视频分析 | 2 天 |

## 学习目标

完成本模块后，你将掌握：

- [ ] 在 OpenCV 中使用 YOLO 检测
- [ ] 自定义检测结果的可视化
- [ ] 构建实时检测系统
- [ ] 分析视频并生成报告

## 集成架构

```
摄像头/视频 → OpenCV 读取 → YOLO 检测 → 后处理 → 可视化/保存
```

## 运行示例

```bash
conda activate yolo

# 实时检测（需要摄像头）
python 01_opencv_yolo/realtime_detection.py
```

## macOS 注意事项

1. **摄像头权限**: 首次运行需要授权
2. **MPS 加速**: 自动使用 Apple Silicon GPU
3. **窗口显示**: 使用 `cv2.WINDOW_NORMAL` 适配 Retina

## 下一步

完成后进入 `07_projects/` 开始实战项目！

