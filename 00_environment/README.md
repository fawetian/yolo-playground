# 00 - 环境配置 🛠️

## 本模块内容

配置 YOLO 和 OpenCV 学习所需的开发环境。

## 文件说明

| 文件 | 说明 |
|-----|------|
| `requirements.txt` | Python 依赖列表 |
| `setup_guide.md` | 详细的环境搭建指南 |
| `test_env.py` | 环境验证脚本 |

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n yolo python=3.11 -y

# 2. 激活环境
conda activate yolo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python test_env.py
```

## 学习目标

- [ ] 安装 Miniforge/Conda
- [ ] 创建并激活 `yolo` 环境
- [ ] 安装所有依赖
- [ ] 验证 OpenCV 和 YOLO 正常工作
- [ ] (Apple Silicon) 验证 MPS 加速可用

## 下一步

环境配置完成后，进入 `01_opencv_basics/` 开始学习！

