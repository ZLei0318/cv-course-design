"""模型工厂注册表：train/evaluate 统一入口，加新模型只改这里"""
from src.segformer import build_segformer
from src.unet import build_unet

FACTORIES = {
    "unet": build_unet,
    "segformer": build_segformer,
}
