# 07 - 实战项目 🚀

## 本模块内容

综合运用所学知识，完成完整的计算机视觉项目。

## 项目列表

| 项目 | 难度 | 预计时间 | 技术点 |
| 目录 | 脚本 | 说明 |
|-----|-----|-----|
| `project_01_face_detection` | `face_detect_app.py` | 人脸/头部检测，隐私马赛克与特效 |
| `project_02_vehicle_counting` | `roi_counter.py` | 基于 ROI 线的车辆流量统计 |
| `project_03_ocr_pipeline` | `ocr_pipeline.py` | 检测 -> 裁剪 -> OCR 文字识别 (需 Tesseract) |
| `project_04_anomaly_detection` | `simple_anomaly.py` | 基于区域规则的异常/闯入检测 |

## 运行方式

进入对应目录运行脚本即可，例如：
```bash
python project_01_face_detection/face_detect_app.py
```

## 项目结构建议

```
project_xx_xxx/
├── README.md           # 项目说明
├── requirements.txt    # 项目依赖（如有额外）
├── src/
│   ├── __init__.py
│   ├── detector.py     # 检测模块
│   ├── processor.py    # 处理模块
│   └── visualizer.py   # 可视化模块
├── config/
│   └── config.yaml     # 配置文件
├── data/               # 测试数据
├── outputs/            # 输出结果
└── main.py             # 主程序
```

## 运行项目

```bash
conda activate yolo
cd project_01_face_detection
python main.py
```

## 学习建议

1. **先理解需求**: 仔细阅读每个项目的 README
2. **分步实现**: 将项目拆解为小任务
3. **测试驱动**: 边开发边测试
4. **记录问题**: 记录遇到的问题和解决方案

## 扩展想法

完成基础项目后，可以尝试：
- 添加 Web 界面
- 部署到服务器
- 添加数据库存储
- 实现 API 接口

