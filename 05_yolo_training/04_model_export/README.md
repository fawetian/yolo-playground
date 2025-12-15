# 04_model_export - 模型导出 📦

## 学习目标

- 将模型导出为不同格式
- 理解各格式的优缺点
- 部署模型

## 支持的导出格式

| 格式 | 后缀 | 用途 |
|-----|------|------|
| PyTorch | .pt | 原生格式 |
| ONNX | .onnx | 通用交换格式 |
| CoreML | .mlmodel | iOS/macOS |
| TensorRT | .engine | NVIDIA GPU |
| TFLite | .tflite | 移动端 |
| OpenVINO | - | Intel 硬件 |

## 导出方法

### 基本导出
```python
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")

# 导出为 ONNX
model.export(format="onnx")

# 导出为 CoreML (macOS/iOS)
model.export(format="coreml")

# 导出为 TFLite
model.export(format="tflite")
```

### 导出参数
```python
model.export(
    format="onnx",
    imgsz=640,          # 输入尺寸
    half=False,         # FP16 量化
    dynamic=True,       # 动态输入尺寸
    simplify=True,      # 简化 ONNX
    opset=12,           # ONNX opset 版本
)
```

## 使用导出的模型

### ONNX
```python
from ultralytics import YOLO

model = YOLO("best.onnx")
results = model("image.jpg")
```

### CoreML (macOS)
```python
model = YOLO("best.mlmodel")
results = model("image.jpg")
```

## macOS 部署建议

### 推荐格式
1. **CoreML** - 最适合 macOS/iOS，支持 Neural Engine
2. **ONNX** - 通用性好，可用 ONNX Runtime

### CoreML 导出
```python
model.export(
    format="coreml",
    nms=True,  # 包含 NMS 后处理
)
```

## 模型优化

### 量化
```python
# INT8 量化 (需要校准数据)
model.export(format="onnx", int8=True, data="data.yaml")

# FP16 量化
model.export(format="onnx", half=True)
```

### 模型剪枝
在训练时使用较小的模型：
```python
model = YOLO("yolo11n.pt")  # nano 版本最小
```

## 待创建文件

- `01_export_onnx.py` - ONNX 导出
- `02_export_coreml.py` - CoreML 导出
- `03_model_optimization.py` - 模型优化

