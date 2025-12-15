# 07 - 实战项目 🚀

## 本模块内容

综合运用所学知识，完成完整的计算机视觉项目。

## 项目列表

| 项目 | 难度 | 预计时间 | 技术点 |
|-----|------|---------|--------|
| `project_01_face_detection/` | ⭐⭐ | 3 天 | 人脸检测、人脸识别 |
| `project_02_vehicle_counting/` | ⭐⭐⭐ | 5 天 | 目标追踪、越线计数 |
| `project_03_ocr_pipeline/` | ⭐⭐⭐ | 5 天 | 文字检测、OCR |
| `project_04_anomaly_detection/` | ⭐⭐⭐⭐ | 7 天 | 异常检测、告警系统 |

## 建议顺序

1. **入门**: 从 `project_01_face_detection` 开始
2. **进阶**: 完成 `project_02_vehicle_counting`
3. **挑战**: 尝试 `project_03_ocr_pipeline` 或 `project_04_anomaly_detection`

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

