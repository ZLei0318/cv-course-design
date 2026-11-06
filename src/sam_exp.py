"""SAM 零样本实验

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
from transformers import SamModel, SamProcessor

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


def masks_to_original(outputs, orig_hw=(512, 512)):
    """绕开 transformers 5.x 有问题的 post_process_masks：直接把 pred_masks 双线性回原尺寸

    输入是正方形 512 时，官方后处理等价于此操作。pred_masks 可能是
    (B, point_batch, C, h, w) 的任意前缀，统一 reshape 后插值。
    返回 (n_masks, 512, 512) 的概率图 numpy。
    """
    pm = outputs.pred_masks
    pm2 = pm.reshape(-1, 1, pm.shape[-2], pm.shape[-1])
    up = torch.nn.functional.interpolate(pm2.float(), size=orig_hw, mode="bilinear", align_corners=False)
    return up.sigmoid().cpu().numpy()[:, 0]


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
        probs = masks_to_original(outputs)  # (1, 512, 512)
        m = probs[0] > 0.5
        if m.sum() == 0:
            empties += 1
            continue
        ious.append(masked_iou(m, water[0] > 0, valid.numpy()))
    if not ious:
        return {"best": 0.0, "mean": 0.0, "empty_rate": 1.0}
    return {"best": max(ious), "mean": float(np.mean(ious)), "empty_rate": empties / n}


@torch.no_grad()
def grid_zero_shot(model, processor, ds, indices, out_dir: Path, limit=3, stride=64):
    """无真值提示的零样本：网格点逐个前向扫描，掩码并集=SAM 看见的前景

    （原计划用 SamAutomaticMaskGenerator，transformers 5.x 已移除；多点一次前向的
    形状后处理也不支持，故逐点前向——语义等价，仅慢一点）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in indices[:limit]:
        img, water, valid, name = ds[i]
        rgb01 = to_rgb(img.numpy())
        union = np.zeros(rgb01.shape[:2], dtype=bool)
        n_pts = 0
        for y in range(stride // 2, 512, stride):
            for x in range(stride // 2, 512, stride):
                inputs = processor(rgb01, input_points=[[[x, y]]], return_tensors="pt").to("cuda")
                outputs = model(**inputs, multimask_output=False)
                m = masks_to_original(outputs)[0] > 0.5
                union |= m
                n_pts += 1
        print(f"[grid] {name}: {n_pts} 点扫描完成", flush=True)
        iou = masked_iou(union, water[0].numpy() > 0, valid.numpy())
        rows.append({"chip": name, "points": n_pts,
                     "coverage_pct": 100 * union.mean(), "iou_vs_water": iou})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        axes[0].imshow(rgb01[..., 0], cmap="gray"); axes[0].set_title("SAR (vv)")
        ov = rgb01.copy(); ov[union] = [0.9, 0.3, 0.3]
        axes[1].imshow(ov); axes[1].set_title(f"SAM grid zero-shot union ({n_pts} pts)")
        gt = np.zeros((*rgb01.shape[:2], 3)); gt[water[0] > 0] = [0, 0.4, 1]
        axes[2].imshow(rgb01[..., 0], cmap="gray"); axes[2].imshow(gt, alpha=0.5)
        axes[2].set_title("Ground truth water (blue)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout(); plt.savefig(out_dir / f"{name.replace('.tif','')}_grid.png", dpi=110); plt.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sam_zeroshot")
    ap.add_argument("--splits", nargs="+", default=["test", "bolivia"])
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--amg-limit", type=int, default=3)
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；给出则评估微调后模型")
    args = ap.parse_args()

    np.random.seed(42)
    model = SamModel.from_pretrained(MODEL_ID)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已加载 LoRA 适配器: {args.adapter}")
    model = model.to("cuda").eval()
    processor = SamProcessor.from_pretrained(MODEL_ID)
    print(f"SAM 模型: {MODEL_ID} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.0f}M 参数，全部冻结)"
          + ("（零样本）" if not args.adapter else "（LoRA 微调后）"))

    # 1) 网格点扫描零样本（仅零样本基线时跑）
    if not args.adapter:
        ds_test = Sen1FloodsDataset("test", augment=False)
        grid_rows = grid_zero_shot(model, processor, ds_test, list(range(len(ds_test))),
                                   OUT / "figures/sam_grid", args.amg_limit)
        for r in grid_rows:
            print(f"[grid] {r['chip']}: 覆盖 {r['coverage_pct']:.0f}% 像素, 与水体IoU {r['iou_vs_water']:.4f}")

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
            if (i + 1) % 20 == 0:
                print(f"[heartbeat] {split} {i+1}/{len(ds)} 已评估", flush=True)
        sub = [r for r in all_rows if r["split"] == split]
        df = pd.DataFrame(sub)
        print(f"[{split:8s}] chips {len(df):3d} | 点提示 best-IoU 均值 {df.best.mean():.4f} "
              f"mean-IoU {df['mean'].mean():.4f} 空掩码率 {df.empty_rate.mean()*100:.0f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / f"{args.tag}_eval.csv", index=False)
    print(f"明细已写入 outputs/{args.tag}_eval.csv")


if __name__ == "__main__":
    main()
