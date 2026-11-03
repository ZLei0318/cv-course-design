"""训练循环（模型无关：--model 选择工厂，后续 SegFormer 复用同一套）

- 损失：masked BCE + masked Dice（双掩膜内计算，类别不平衡稳健）
- 指标：洪水类 IoU —— micro（全 valid 像素聚合，与论文口径可比）+ macro（逐 chip 平均）
- 混合精度 bf16（H20 原生支持），余弦退火学习率
- 产物：outputs/checkpoints/{tag}_best.pth（按 valid macro IoU 存优）、outputs/{tag}_history.csv

用法：.venv/bin/python -m src.train --model unet --tag unet --epochs 80
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import Sen1FloodsDataset
from src.models import FACTORIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CKPT_DIR = OUT / "checkpoints"


def masked_bce_dice(logits, water, valid):
    """只在双掩膜内计算 BCE + Dice"""
    v = valid.unsqueeze(1).float()  # (B,1,H,W)
    bce = F.binary_cross_entropy_with_logits(logits, water, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits)
    inter = ((p * water) * v).sum()
    denom = ((p + water) * v).sum()
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 micro IoU/P/R 与 macro IoU（跳过无有效水像素的 chip）"""
    tp = fp = fn = 0
    ious = []
    model.eval()
    for img, water, valid, _ in loader:
        img, water, valid = img.to(device), water.to(device), valid.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (torch.sigmoid(model(img)) > 0.5).float()
        for b in range(img.size(0)):
            v = valid[b]
            if v.sum() == 0:
                continue
            pb, wb = pred[b, 0][v], water[b, 0][v]
            t = (pb * wb).sum()
            ious.append((t / (pb.sum() + wb.sum() - t + 1e-6)).item())
            tp += t
            fp += pb.sum() - t
            fn += wb.sum() - t
    micro_iou = (tp / (tp + fp + fn)).item()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    return micro_iou, float(torch.tensor(ious).mean()), precision, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet", choices=FACTORIES)
    ap.add_argument("--tag", default="unet")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = Sen1FloodsDataset("train", augment=True)
    valid_ds = Sen1FloodsDataset("valid", augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=False, pin_memory=True)
    valid_ld = DataLoader(valid_ds, batch_size=32, num_workers=args.workers, pin_memory=True)
    print(f"[data] train {len(train_ds)} / valid {len(valid_ds)} chips")

    model = FACTORIES[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model}, 参数量 {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_macro = [], -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, nb = time.time(), 0.0, 0
        for img, water, valid, _ in train_ld:
            img, water, valid = img.to(device), water.to(device), valid.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(img)
                loss = masked_bce_dice(logits.float(), water, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        sched.step()

        micro, macro, prec, rec = evaluate(model, valid_ld, device)
        mark = ""
        if macro > best_macro:
            best_macro = macro
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "macro_iou": macro, "micro_iou": micro},
                       CKPT_DIR / f"{args.tag}_best.pth")
            mark = " *best*"
        history.append({"epoch": ep, "loss": total_loss / nb, "lr": sched.get_last_lr()[0],
                        "val_micro_iou": micro, "val_macro_iou": macro,
                        "val_precision": prec, "val_recall": rec})
        pd.DataFrame(history).to_csv(OUT / f"{args.tag}_history.csv", index=False)
        print(f"[{args.tag}] ep{ep:03d} loss {total_loss/nb:.4f} | val microIoU {micro:.4f} "
              f"macroIoU {macro:.4f} P {prec:.4f} R {rec:.4f} | {time.time()-t0:.1f}s{mark}", flush=True)

    print(f"[done] best macro IoU = {best_macro:.4f}, 权重 -> {CKPT_DIR}/{args.tag}_best.pth")


if __name__ == "__main__":
    main()
