"""正式评估：加载最优 checkpoint，在指定 split 上计算洪水类指标

输出（追加写入 outputs/{tag}_eval.csv，一行一个 split）：
- microIoU / macroIoU / Precision / Recall / Dice
用法：.venv/bin/python -m src.evaluate --tag unet_v3 --splits valid test bolivia
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import Sen1FloodsDataset
from src.unet import build_unet

CKPT_DIR = Path(__file__).resolve().parents[1] / "outputs/checkpoints"
OUT = Path(__file__).resolve().parents[1] / "outputs"


@torch.no_grad()
def eval_split(model, split: str, device: str):
    ds = Sen1FloodsDataset(split, augment=False)
    ld = DataLoader(ds, batch_size=32, num_workers=5, pin_memory=True)
    tp = fp = fn = tn = 0
    ious = []
    chip_rows = []
    model.eval()
    for img, water, valid, names in ld:
        img, water, valid = img.to(device), water.to(device), valid.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (torch.sigmoid(model(img)) > 0.5).float()
        for b, name in enumerate(names):
            v = valid[b]
            if v.sum() == 0:
                continue
            pb, wb = pred[b, 0][v], water[b, 0][v]
            t = float((pb * wb).sum())
            p_sum, w_sum = float(pb.sum()), float(wb.sum())
            iou = t / (p_sum + w_sum - t + 1e-6)
            ious.append(iou)
            chip_rows.append({"split": split, "chip": name, "iou": iou,
                              "water_pct": 100 * w_sum / float(v.sum())})
            tp += t; fp += p_sum - t; fn += w_sum - t; tn += float(((1 - pb) * (1 - wb)).sum())
    micro = tp / (tp + fp + fn)
    return {
        "split": split, "chips": len(ious),
        "micro_iou": micro,
        "macro_iou": float(torch.tensor(ious).mean()),
        "precision": tp / (tp + fp + 1e-6),
        "recall": tp / (tp + fn + 1e-6),
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-6),
    }, chip_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test", "bolivia"])
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(CKPT_DIR / f"{args.tag}_best.pth", map_location=device, weights_only=True)
    model = build_unet().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"加载 {args.tag}（epoch {ckpt['epoch']}, valid macroIoU {ckpt['macro_iou']:.4f}）")

    rows, chips = [], []
    for split in args.splits:
        r, c = eval_split(model, split, device)
        rows.append(r); chips += c
        print(f"[{split:8s}] chips {r['chips']:3d} | microIoU {r['micro_iou']:.4f} macroIoU {r['macro_iou']:.4f} "
              f"P {r['precision']:.4f} R {r['recall']:.4f} Dice {r['dice']:.4f}")

    pd.DataFrame(rows).to_csv(OUT / f"{args.tag}_eval.csv", index=False)
    pd.DataFrame(chips).to_csv(OUT / f"{args.tag}_eval_per_chip.csv", index=False)
    print(f"已写入 outputs/{args.tag}_eval.csv 与 _per_chip.csv")


if __name__ == "__main__":
    main()
