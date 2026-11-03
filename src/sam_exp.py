"""SAM 零样本实验（三代对比 · 第三幕上半场：视觉大模型不训练直接用）

两个视角回答"SAM 能不能直接干水利遥感的活"：
1) AMG 自动掩码生成（无任何提示）：看 SAM 在 SAR 洪水图上"看见"什么 —— 定性，存图
2) 点提示分割（用真值水体像素当提示点）：量化 SAM 眼中的"这个点所在的对象"与水体真值的
   重合度 —— 定量，与 UNet/SegFormer 同为 valid-mask 内 IoU 口径

用法：HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m src.sam_exp --tag sam_zeroshot
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import SamAutomaticMaskGenerator, SamModel, SamProcessor

from src.dataset import Sen1FloodsDataset

OUT = Path(__file__).resolve().parents[1] / "outputs"
MODEL_ID = "facebook/sam-vit-base"


def to_rgb(img2ch: np.ndarray) -> np.ndarray:
    """(2,H,W) [0,1] → (H,W,3) RGB：与 SegFormer 相同的 vv,vh,vv 适配"""
    return np.stack([img2ch[0], img2ch[1], img2ch[0]], axis=-1)


def masked_iou(pred: np.ndarray, water: np.ndarray, valid: np.ndarray) -> float:
    p, w = pred[valid], water[valid]
    inter = (p & w).sum()
    return float(inter / (p.sum() + w.sum() - inter + 1e-6))


@torch.no_grad()
def point_prompted_iou(model, processor, img, water, valid, k_points=5):
    """随机取 k 个真值水点作提示，返回该 chip 的最好/平均 IoU 与空掩码率"""
    ys, xs = np.where(water[0] > 0)
    n = min(k_points, len(ys))
    if n == 0:
        return None
    idx = np.random.choice(len(ys), n, replace=False)
    ious, empties = [], 0
    for i in idx:
        inputs = processor(to_rgb(img), input_points=[[[int(xs[i]), int(ys[i])]]], return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        masks = processor.post_process_masks(
            outputs, inputs["original_sizes"].tolist(), inputs["reshaped_input_sizes"].tolist())
        m = masks[0]  # (num_points?, 1, H, W) logits
        m = np.asarray(m.sigmoid().cpu())[0, 0] > 0.5
        if m.sum() == 0:
            empties += 1
            continue
        ious.append(masked_iou(m, water[0] > 0, valid.numpy()))
    if not ious:
        return {"best": 0.0, "mean": 0.0, "empty_rate": 1.0}
    return {"best": max(ious), "mean": float(np.mean(ious)), "empty_rate": empties / n}


@torch.no_grad()
def amg_qualitative(model, ds, names, out_dir: Path, limit=3):
    """自动掩码生成：无提示零样本，SAM 自己找'对象'"""
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = SamAutomaticMaskGenerator(model)
    rows = []
    for i in names[:limit]:
        img, water, valid, name = ds[i]
        rgb01 = to_rgb(img.numpy())
        masks = gen((rgb01 * 255).astype(np.uint8))
        rows.append({"chip": name, "amg_masks": len(masks),
                     "amg_area_pct": 100 * sum(int(m["segmentation"].sum()) for m in masks) / rgb01.size * 3})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        axes[0].imshow(rgb01[..., 0], cmap="gray"); axes[0].set_title("SAR (vv)")
        ov = rgb01.copy()
        for m in masks:
            ov[m["segmentation"]] = [0.9, 0.3, 0.3]
        axes[1].imshow(ov); axes[1].set_title(f"SAM AMG: {len(masks)} masks (red)")
        gt = np.zeros((*rgb01.shape[:2], 3)); gt[water[0] > 0] = [0, 0.4, 1]
        axes[2].imshow(rgb01[..., 0], cmap="gray"); axes[2].imshow(gt, alpha=0.5)
        axes[2].set_title("Ground truth water (blue)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout(); plt.savefig(out_dir / f"{name.replace('.tif','')}_amg.png", dpi=110); plt.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sam_zeroshot")
    ap.add_argument("--splits", nargs="+", default=["test", "bolivia"])
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--amg-limit", type=int, default=3)
    args = ap.parse_args()

    np.random.seed(42)
    model = SamModel.from_pretrained(MODEL_ID).to("cuda").eval()
    processor = SamProcessor.from_pretrained(MODEL_ID)
    print(f"SAM 零样本模型: {MODEL_ID} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.0f}M 参数，全部冻结)")

    # 1) AMG 定性（test 前 N 张）
    ds_test = Sen1FloodsDataset("test", augment=False)
    amg_rows = amg_qualitative(model, ds_test, list(range(len(ds_test))), OUT / "figures/sam_amg", args.amg_limit)
    for r in amg_rows:
        print(f"[AMG] {r['chip']}: 生成 {r['amg_masks']} 个掩码, 覆盖 {r['amg_area_pct']:.0f}% 像素")

    # 2) 点提示定量
    all_rows = []
    for split in args.splits:
        ds = Sen1FloodsDataset(split, augment=False)
        for i in range(len(ds)):
            img, water, valid, name = ds[i]
            if (water[0] > 0).sum() == 0:
                continue
            r = point_prompted_iou(model, processor, img.numpy(), water.numpy(), valid, args.points)
            r.update({"split": split, "chip": name})
            all_rows.append(r)
        sub = [r for r in all_rows if r["split"] == split]
        df = pd.DataFrame(sub)
        print(f"[{split:8s}] chips {len(df):3d} | 点提示 best-IoU 均值 {df.best.mean():.4f} "
              f"mean-IoU {df['mean'].mean():.4f} 空掩码率 {df.empty_rate.mean()*100:.0f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / f"{args.tag}_eval.csv", index=False)
    print(f"明细已写入 outputs/{args.tag}_eval.csv")


if __name__ == "__main__":
    main()
