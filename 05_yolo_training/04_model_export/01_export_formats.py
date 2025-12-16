"""
模型导出
=======

学习目标:
- 将 PyTorch 模型导出为其他格式
- ONNX (通用格式)
- CoreML (Apple 设备优化)
- Benchmark (性能基准测试)
"""

from pathlib import Path
import sys
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model


def main():
    print("=" * 60)
    print("📤 模型导出")
    print("=" * 60)
    
    # ==========================================
    # 1. 准备模型
    # ==========================================
    
    # 实际项目中，这里应该加载训练好的模型 (e.g. "runs/train/exp/weights/best.pt")
    # 这里演示使用预训练模型
    print("\n📦 加载模型...")
    model = load_yolo_model("yolo11n.pt")
    
    # ==========================================
    # 2. 导出为 ONNX
    # ==========================================
    
    # ONNX 是最通用的格式，支持多种推理运行时
    print("\n🔄 导出为 ONNX 格式...")
    try:
        onnx_path = model.export(format="onnx")
        print(f"✅ 导出成功: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
    
    # ==========================================
    # 3. 导出为 CoreML (macOS)
    # ==========================================
    
    # CoreML 专用于 Apple 设备 (iOS, macOS)
    print("\n🍎 导出为 CoreML 格式...")
    try:
        # nms=True 在模型中包含 NMS 后处理，简化 iOS 开发
        coreml_path = model.export(format="coreml", nms=True)
        print(f"✅ 导出成功: {coreml_path}")
    except Exception as e:
        print(f"❌ CoreML 导出失败: {e}")
        print("  可能需要安装核心依赖: pip install coremltools")
    
    # ==========================================
    # 4. 导出格式对比
    # ==========================================
    
    print("\n📝 常见导出格式:")
    print("""
    | 格式      | 参数 (format) | 适用场景          |
    |----------|--------------|------------------|
    | PyTorch  | -            | 训练、这Python 推理 |
    | ONNX     | 'onnx'       | 跨平台、C++ 部署   |
    | CoreML   | 'coreml'     | iOS, macOS apps  |
    | TFLite   | 'tflite'     | Android Edge     |
    | TensorRT | 'engine'     | Nvidia GPU 加速   |
    """)


if __name__ == "__main__":
    main()
