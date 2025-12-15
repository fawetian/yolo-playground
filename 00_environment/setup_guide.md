# macOS 环境搭建指南 🍎

> 专为 macOS 系统优化，支持 Intel 和 Apple Silicon (M1/M2/M3)

## 1. 系统要求

### 硬件
- **Intel Mac**: 任意 Intel 处理器
- **Apple Silicon**: M1 / M2 / M3 系列（推荐，性能更好）
- **内存**: >= 8GB RAM（推荐 16GB）
- **存储**: >= 20GB 可用空间

### 软件
- **macOS**: 12.0+ (Monterey 或更高)
- **Python**: 3.10 或 3.11（**推荐 3.11**）
- **Xcode Command Line Tools**: 必需

---

## 2. 前置准备

### 2.1 安装 Xcode Command Line Tools

```bash
xcode-select --install
```

### 2.2 安装 Homebrew（如果没有）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2.3 安装 Python（推荐使用 pyenv）

```bash
# 安装 pyenv
brew install pyenv

# 添加到 shell 配置 (~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# 重新加载
source ~/.zshrc

# 安装 Python 3.11
pyenv install 3.11.7
pyenv global 3.11.7

# 验证
python --version
```

### 2.4 安装 FFmpeg（视频处理需要）

```bash
brew install ffmpeg
```

---

## 3. 环境安装 (使用 Conda)

本项目统一使用 **conda 环境**，环境名为 `yolo`。

### 安装 Miniforge

```bash
# 安装 Miniforge (Apple Silicon 优化版 Conda)
brew install miniforge

# 初始化 (首次安装后需要)
conda init zsh
source ~/.zshrc
```

### 创建 yolo 环境

```bash
# 创建名为 yolo 的环境
conda create -n yolo python=3.11 -y

# 激活环境
conda activate yolo

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r 00_environment/requirements.txt
```

### 日常使用

```bash
# 每次使用前激活环境
conda activate yolo

# 查看当前环境
conda info --envs

# 退出环境
conda deactivate

# 删除环境 (如需重建)
conda remove -n yolo --all
```

---

## 4. 验证安装

创建测试脚本 `test_env.py`:

```python
#!/usr/bin/env python3
"""macOS 环境验证脚本"""

import platform

def main():
    print("=" * 50)
    print("🍎 macOS 环境验证")
    print("=" * 50)
    
    # 系统信息
    print(f"\n📱 系统信息:")
    print(f"  macOS 版本: {platform.mac_ver()[0]}")
    print(f"  处理器: {platform.processor()}")
    
    # 判断是否为 Apple Silicon
    is_arm = platform.processor() == 'arm'
    chip_type = "Apple Silicon (M系列)" if is_arm else "Intel"
    print(f"  芯片类型: {chip_type}")
    
    # 1. 检查 OpenCV
    try:
        import cv2
        print(f"\n✅ OpenCV: {cv2.__version__}")
        
        # 测试摄像头访问权限提示
        print("  💡 首次使用摄像头时，系统会请求权限")
    except ImportError as e:
        print(f"\n❌ OpenCV 未安装: {e}")
    
    # 2. 检查 NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 未安装: {e}")
    
    # 3. 检查 PyTorch 和 MPS
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 检查 MPS (Metal Performance Shaders) 支持
        if torch.backends.mps.is_available():
            print(f"  🚀 MPS 加速: ✅ 可用 (Apple Silicon GPU)")
            # 测试 MPS
            try:
                x = torch.ones(1, device="mps")
                print(f"  🔧 MPS 测试: ✅ 正常工作")
            except Exception as e:
                print(f"  ⚠️ MPS 测试失败: {e}")
        else:
            print(f"  ⚠️ MPS 加速: 不可用")
            if not is_arm:
                print(f"     (Intel Mac 不支持 MPS，将使用 CPU)")
            else:
                print(f"     (请检查 macOS 版本是否 >= 12.3)")
        
        # CPU 后备
        print(f"  💻 CPU 计算: ✅ 始终可用")
        
    except ImportError as e:
        print(f"❌ PyTorch 未安装: {e}")
    
    # 4. 检查 Ultralytics (YOLO)
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Ultralytics 未安装: {e}")
    
    # 5. 检查 Matplotlib
    try:
        import matplotlib
        # macOS 后端设置
        matplotlib.use('TkAgg')  # 或 'MacOSX'
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib 未安装: {e}")
    
    # 6. 检查 FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version.split(' ')[2]}")
        else:
            print("⚠️ FFmpeg: 未安装 (brew install ffmpeg)")
    except FileNotFoundError:
        print("⚠️ FFmpeg: 未安装 (brew install ffmpeg)")
    
    print("\n" + "=" * 50)
    print("验证完成!")
    print("=" * 50)
    
    # 设备推荐
    if is_arm:
        print("\n💡 推荐配置:")
        print("  YOLO 训练/推理时使用 device='mps' 获得 GPU 加速")
    else:
        print("\n💡 Intel Mac 提示:")
        print("  将使用 CPU 进行计算，速度较慢但功能正常")

if __name__ == "__main__":
    main()
```

运行验证:

```bash
python test_env.py
```

---

## 5. Apple Silicon MPS 加速 🚀

M1/M2/M3 芯片可以使用 **MPS (Metal Performance Shaders)** 进行 GPU 加速。

### 在 YOLO 中使用 MPS

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 使用 MPS 进行推理
results = model("image.jpg", device="mps")

# 使用 MPS 进行训练
model.train(data="data.yaml", device="mps", epochs=100)
```

### 在 PyTorch 中使用 MPS

```python
import torch

# 检查 MPS 可用性
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 加速")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 将张量移动到 MPS
x = torch.randn(3, 3).to(device)
```

### MPS vs CPU 性能对比

| 任务 | CPU | MPS | 加速比 |
|-----|-----|-----|--------|
| YOLO11n 推理 | ~100ms | ~20ms | 5x |
| YOLO11n 训练 | ~10min/epoch | ~2min/epoch | 5x |

---

## 6. macOS 特有注意事项

### 6.1 摄像头权限

首次使用 OpenCV 访问摄像头时，系统会弹出权限请求：

1. 点击 "允许"
2. 如果错过了，去 **系统设置 → 隐私与安全性 → 摄像头** 手动开启

```python
import cv2

# 首次运行会请求权限
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头访问被拒绝，请在系统设置中开启权限")
```

### 6.2 OpenCV 窗口显示

macOS 上 OpenCV 窗口可能有些问题，推荐配置：

```python
import cv2

# 使用这个可以让窗口更稳定
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.imshow("window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)  # macOS 上需要额外的 waitKey
```

### 6.3 Matplotlib 后端

如果 Matplotlib 图像不显示：

```python
import matplotlib
matplotlib.use('TkAgg')  # 在 import pyplot 之前设置
import matplotlib.pyplot as plt
```

或者在 `~/.matplotlib/matplotlibrc` 中添加：
```
backend: TkAgg
```

### 6.4 高 DPI 显示（Retina）

Retina 屏幕上图像可能显示很大：

```python
import cv2

# 缩小窗口
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 800, 600)
cv2.imshow("window", image)
```

---

## 7. 常见问题

### Q: OpenCV 安装后 import 报错

```bash
# 卸载重装
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Q: MPS 报错 "MPS backend out of memory"

```python
# 减小批次大小
model.train(data="data.yaml", device="mps", batch=8)  # 从 16 减到 8

# 或者回退到 CPU
model.train(data="data.yaml", device="cpu")
```

### Q: YOLO 下载模型很慢

```bash
# 使用代理或手动下载
# 下载地址: https://github.com/ultralytics/assets/releases

# 放到项目目录后直接使用
model = YOLO("./yolo11n.pt")
```

### Q: cv2.imshow 窗口无响应

```python
# 在主线程中运行，并确保有 waitKey
cv2.imshow("window", image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)  # 重要！macOS 需要这个
```

---

## 8. 下一步

环境配置完成后：

```bash
# 运行第一个 OpenCV 示例
python 01_opencv_basics/01_image_io/01_read_image.py

# 运行第一个 YOLO 示例
python 04_yolo_basics/01_intro/01_yolo_quickstart.py
```

祝学习愉快！🎉
