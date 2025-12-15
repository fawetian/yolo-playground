"""
模型加载工具
统一管理 YOLO 模型的下载和加载
优先从本地 models/yolo/ 目录加载，如果没有则下载
"""

from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import shutil


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 模型存储目录
MODELS_DIR = PROJECT_ROOT / "models" / "yolo"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_yolo_model(model_name: str, download_if_missing: bool = True) -> YOLO:
    """
    加载 YOLO 模型，优先从本地 models/yolo/ 目录加载
    
    Args:
        model_name: 模型名称，如 "yolo11n.pt", "yolo11s.pt" 等
        download_if_missing: 如果本地不存在，是否自动下载
    
    Returns:
        YOLO 模型对象
    
    Examples:
        >>> model = load_yolo_model("yolo11n.pt")
        >>> model = load_yolo_model("yolo11m-seg.pt")
    """
    # 确保模型名称有 .pt 后缀
    if not model_name.endswith((".pt", ".onnx", ".engine", ".mlmodel")):
        model_name = f"{model_name}.pt"
    
    # 本地模型路径
    local_model_path = MODELS_DIR / model_name
    
    # 如果本地存在，直接加载
    if local_model_path.exists():
        LOGGER.info(f"📦 从本地加载模型: {local_model_path}")
        return YOLO(str(local_model_path))
    
    # 本地不存在，尝试加载（会自动下载）
    if download_if_missing:
        LOGGER.info(f"📥 模型不存在，将从网络下载: {model_name}")
        LOGGER.info(f"   下载后将保存到: {MODELS_DIR}")
        
        # 加载模型（会自动下载到默认位置）
        model = YOLO(model_name)
        
        # 尝试将下载的模型复制到我们的目录
        # Ultralytics 默认下载到 ~/.ultralytics/weights/ 或当前目录
        try:
            # 方法1: 从 ckpt_path 获取
            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                source_path = Path(model.ckpt_path)
                if source_path.exists() and source_path != local_model_path:
                    LOGGER.info(f"📋 复制模型到本地目录...")
                    shutil.copy2(source_path, local_model_path)
                    LOGGER.info(f"✅ 模型已保存到: {local_model_path}")
                    return model
            
            # 方法2: 从 model_name 查找（可能在当前目录或默认位置）
            possible_paths = [
                Path(model_name),  # 当前目录
                Path.home() / ".ultralytics" / "weights" / model_name,  # 默认位置
            ]
            
            for source_path in possible_paths:
                if source_path.exists() and source_path != local_model_path:
                    LOGGER.info(f"📋 复制模型到本地目录: {source_path}")
                    shutil.copy2(source_path, local_model_path)
                    LOGGER.info(f"✅ 模型已保存到: {local_model_path}")
                    return model
            
            # 方法3: 如果模型已加载，尝试保存
            if hasattr(model, 'model') and model.model is not None:
                LOGGER.info(f"📋 保存模型到本地目录...")
                model.save(str(local_model_path))
                LOGGER.info(f"✅ 模型已保存到: {local_model_path}")
                
        except Exception as e:
            LOGGER.warning(f"⚠️ 无法保存模型到本地目录: {e}")
            LOGGER.info(f"   模型已下载，下次运行将尝试从默认位置加载")
        
        return model
    else:
        raise FileNotFoundError(
            f"模型 {model_name} 不存在于 {MODELS_DIR}，"
            f"且 download_if_missing=False，无法下载"
        )


def list_local_models() -> list:
    """
    列出本地已下载的模型
    
    Returns:
        模型文件列表
    """
    models = list(MODELS_DIR.glob("*.pt"))
    models.extend(MODELS_DIR.glob("*.onnx"))
    models.extend(MODELS_DIR.glob("*.engine"))
    models.extend(MODELS_DIR.glob("*.mlmodel"))
    return sorted(models)


def get_model_path(model_name: str) -> Path:
    """
    获取模型的完整路径（不加载）
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型路径
    """
    if not model_name.endswith((".pt", ".onnx", ".engine", ".mlmodel")):
        model_name = f"{model_name}.pt"
    return MODELS_DIR / model_name


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("🧪 模型加载工具测试")
    print("=" * 60)
    
    print(f"\n📁 模型目录: {MODELS_DIR}")
    
    print("\n📋 本地已有模型:")
    local_models = list_local_models()
    if local_models:
        for model in local_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model.name} ({size_mb:.1f} MB)")
    else:
        print("  (暂无)")
    
    print("\n💡 使用示例:")
    print("  from utils.model_loader import load_yolo_model")
    print("  model = load_yolo_model('yolo11n.pt')")

