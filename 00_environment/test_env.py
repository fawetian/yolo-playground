#!/usr/bin/env python3
"""
macOS 环境验证脚本
运行: python 00_environment/test_env.py
"""

import platform


def main():
    print("=" * 50)
    print("🍎 macOS 环境验证")
    print("=" * 50)
    
    # 系统信息
    print(f"\n📱 系统信息:")
    print(f"  macOS 版本: {platform.mac_ver()[0]}")
    print(f"  处理器架构: {platform.processor()}")
    
    # 判断是否为 Apple Silicon
    is_arm = platform.processor() == 'arm'
    chip_type = "Apple Silicon (M系列) 🚀" if is_arm else "Intel"
    print(f"  芯片类型: {chip_type}")
    
    # 1. 检查 OpenCV
    print(f"\n📦 依赖检查:")
    try:
        import cv2
        print(f"  ✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  ❌ OpenCV 未安装: {e}")
    
    # 2. 检查 NumPy
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy 未安装: {e}")
    
    # 3. 检查 PyTorch 和 MPS
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        
        # 检查 MPS 支持
        print(f"\n⚡ GPU 加速:")
        if torch.backends.mps.is_available():
            print(f"  ✅ MPS 可用 (Apple Silicon GPU)")
            # 测试 MPS
            try:
                x = torch.ones(1, device="mps")
                print(f"  ✅ MPS 测试通过")
            except Exception as e:
                print(f"  ⚠️ MPS 测试失败: {e}")
        else:
            print(f"  ⚠️ MPS 不可用")
            if not is_arm:
                print(f"     (Intel Mac 不支持 MPS，将使用 CPU)")
            else:
                print(f"     (请检查 macOS 版本是否 >= 12.3)")
        
        print(f"  ✅ CPU 计算始终可用")
        
    except ImportError as e:
        print(f"  ❌ PyTorch 未安装: {e}")
    
    # 4. 检查 Ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  ✅ Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"  ❌ Ultralytics 未安装: {e}")
    
    # 5. 检查 Matplotlib
    try:
        import matplotlib
        print(f"  ✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ Matplotlib 未安装: {e}")
    
    # 6. 检查 FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0].split(' ')[2]
            print(f"  ✅ FFmpeg: {version}")
        else:
            print("  ⚠️ FFmpeg: 未正常工作")
    except FileNotFoundError:
        print("  ⚠️ FFmpeg: 未安装")
        print("     安装命令: brew install ffmpeg")
    
    print("\n" + "=" * 50)
    print("验证完成!")
    print("=" * 50)
    
    # 推荐配置
    print("\n💡 推荐配置:")
    if is_arm:
        print("  • YOLO 使用 device='mps' 获得 GPU 加速")
        print("  • 训练时 batch 建议 8-16 (MPS 内存有限)")
    else:
        print("  • Intel Mac 将使用 CPU，速度较慢但功能正常")
        print("  • 建议使用较小的模型 (yolo11n)")
    
    print("\n🚀 开始学习:")
    print("  python 01_opencv_basics/01_image_io/01_read_image.py")


if __name__ == "__main__":
    main()

