"""SAM LoRA 微调（三代对比 · 第三幕下半场：少量数据高效微调救活大模型）

设计：
- 冻结 SAM-ViT-B 全部权重，peft LoRA 只注入注意力 q/v 投影（编码器+解码器）
- 训练目标与 UNet/SegFormer 完全一致：masked BCE + Dice（同一把尺子量三代）
- 交互式提示协议：每张图随机取 1 个真值水体像素作点提示（与零样本评估同协议，
  评估时同样用 5 个随机水点取最好，前后可比）
- 用法：HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m src.train_sam --epochs 100
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import SamModel, SamProcessor

from src.dataset import Sen1FloodsDataset

OUT = Path(__file__).resolve().parents[1] / "outputs"
MODEL_ID = "facebook/sam-vit-base"


class SamPromptDataset(Dataset):
    """包装 Sen1FloodsDataset：额外返回一个随机真值水点作为提示"""

    def __init__(self, split: str):
        self.ds = Sen1FloodsDataset(split, augment=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, water, valid, name = self.ds[i]
        ys, xs = torch.where(water[0] > 0)
        if len(ys) == 0:  # 无水样本退化到中心点（损失几乎全为背景，影响可忽略）
            point = torch.tensor([256, 256])
        else:
            k = int(torch.randint(len(ys), (1,)))
            point = torch.tensor([int(xs[k]), int(ys[k])])
        return img, water, valid, point, name


def masked_bce_dice(logits, water, valid):
    v = valid.unsqueeze(1).float()
    bce = F.binary_cross_entropy_with_logits(logits, water, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits)
    inter = ((p * water) * v).sum()
    denom = ((p + water) * v).sum()
    return bce + 1 - (2 * inter + 1) / (denom + 1)


def forward_pred_masks(model, processor, imgs, points, device):
    """SAM 前向：返回上采样到原图尺寸的 logits (B,1,H,W)"""
    rgbs = [np.stack([im[0], im[1], im[0]], axis=-1) for im in imgs]
    inputs = processor(rgbs, input_points=[[[int(p[0]), int(p[1])]] for p in points],
                       return_tensors="pt").to(device)
    out = model(**inputs, multimask_output=False)
    pm = out.pred_masks  # 前缀维度组合不定，统一 reshape 成 (n,1,h,w)
    pm = pm.reshape(-1, 1, pm.shape[-2], pm.shape[-1])
    return F.interpolate(pm.float(), size=imgs[0].shape[-2:], mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="sam_lora")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda"

    base = SamModel.from_pretrained(MODEL_ID)
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "v_proj"])  # 编码器+解码器注意力投影
    model = get_peft_model(base, lcfg).to(device)
    model.print_trainable_parameters()

    processor = SamProcessor.from_pretrained(MODEL_ID)
    train_ld = DataLoader(SamPromptDataset("train"), batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for imgs, waters, valids, points, _ in train_ld:
            imgs = [im for im in imgs]
            waters, valids = waters.to(device), valids.to(device)
            logits = forward_pred_masks(model, processor, imgs, points, device)
            loss = masked_bce_dice(logits, waters, valids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        history.append({"epoch": ep, "loss": tot / nb})
        pd.DataFrame(history).to_csv(OUT / f"{args.tag}_history.csv", index=False)
        if ep % 5 == 0 or ep == 1:
            print(f"[{args.tag}] ep{ep:03d} loss {tot/nb:.4f} | {time.time()-t0:.1f}s", flush=True)

    model.save_pretrained(OUT / "checkpoints" / f"{args.tag}_adapter")
    print(f"LoRA 适配器已保存 -> outputs/checkpoints/{args.tag}_adapter")


if __name__ == "__main__":
    main()
