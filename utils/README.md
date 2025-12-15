# Utils 工具函数 📚

## 模块说明

### model_loader.py

统一的模型加载工具，管理 YOLO 模型的下载和加载。

**主要功能**：
- 优先从 `models/yolo/` 目录加载模型
- 如果本地不存在，自动下载并保存
- 支持所有 YOLO 模型格式（.pt, .onnx, .engine, .mlmodel）

**使用示例**：
```python
from utils.model_loader import load_yolo_model

# 加载模型（自动管理下载）
model = load_yolo_model("yolo11n.pt")

# 列出本地已有模型
from utils.model_loader import list_local_models
models = list_local_models()
```

### helpers.py

通用辅助函数，包括：
- 图像加载和保存
- 图像信息打印
- 边界框绘制
- 设备检测（macOS MPS）

**使用示例**：
```python
from utils.helpers import load_image, show_image, get_device

img = load_image("image.jpg")
show_image(img, "My Image")
device = get_device()  # 获取最佳设备
```

