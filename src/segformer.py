"""SegFormer（MiT-B2，ImageNet 预训练）——三代对比的第二代：Transformer 通用分割

- HF transformers 官方实现，编码器带 ImageNet 预训练（对比手写 UNet 检验"预训练的价值"）
- SAR 双通道适配：复制 vv 补成三通道（保持预训练 patch embedding 的输入维度）
- HF 语义分割头输出为 1/4 分辨率，双线性上采样回原尺寸
- 二分类：num_labels=1，解码头随机初始化（ignore_mismatched_sizes）
国内下载：HF_ENDPOINT=https://hf-mirror.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/mit-b2"


class SegFormerWrapper(nn.Module):
    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            model_id, num_labels=1, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        if x.size(1) == 2:  # (vv, vh) → (vv, vh, vv)
            x = torch.cat([x, x[:, :1]], dim=1)
        logits = self.net(pixel_values=x).logits  # (B, 1, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_segformer() -> SegFormerWrapper:
    return SegFormerWrapper()
