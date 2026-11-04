"""淹没斑块计数与面积统计（阶段 3：把分割结果变成灾情评估数字）

流程：UNet 预测水体 → 双掩膜内二值化 → 连通域分析 → 面积过滤 → 斑块计数 + 面积换算
口径：Sen1Floods11 为 Sentinel-1 10m 分辨率 → 每像素 100㎡（1 km² = 10,000 px）
      预测与真值用完全相同的后处理（同一 min_area 噪声下限），保证可比
产物：outputs/counting_eval.csv（逐 chip）+ outputs/figures/counting/*.png（专题图）

用法：.venv/bin/python -m src.counting --tag unet_v3 --model unet
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import Sen1FloodsDataset
from src.models import FACTORIES

CKPT_DIR = Path(__file__).resolve().parents[1] / "outputs/checkpoints"
OUT = Path(__file__).resolve().parents[1] / "outputs"
PX_AREA_M2 = 100.0  # 10m × 10m


def count_components(binary: np.ndarray, min_area_px: int):
    """连通域分析：返回 (斑块数, 各斑块面积px列表, 带编号标签图)"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]  # 0 号是背景
    kept = [a for a in areas if a >= min_area_px]
    return len(kept), sorted(kept, reverse=True), labels


@torch.no_grad()
def predict_mask(model, ds, device):
    """返回逐 chip 的 (预测二值图, 真值二值图, 有效掩膜)"""
    ld = DataLoader(ds, batch_size=32, num_workers=5, pin_memory=True)
    results = []
    model.eval()
    for img, water, valid, names in ld:
        img = img.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (torch.sigmoid(model(img)) > 0.5).float().cpu().numpy()[:, 0]
        for b, name in enumerate(names):
            v = valid[b].numpy()
            results.append((name, pred[b] * v, (water[b, 0].numpy() > 0.5) * v, v))
    return results


def flood_map_figure(rgb01, gt_binary, pred_binary, labels, n_kept, name, out_dir):
    """灾情专题图：SAR 底图 + 预测淹没填充 + 真值轮廓 + 斑块编号"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    axes[0].imshow(rgb01[..., 0], cmap="gray")
    gt = np.zeros((*gt_binary.shape, 4)); gt[gt_binary > 0] = [0, 0.4, 1, 0.55]
    pred = np.zeros((*gt_binary.shape, 4)); pred[pred_binary > 0] = [1, 0.25, 0.15, 0.5]
    axes[0].imshow(gt); axes[0].set_title("Ground truth water")
    axes[1].imshow(rgb01[..., 0], cmap="gray"); axes[1].imshow(pred)
    axes[1].contour(gt_binary, colors="cyan", linewidths=0.5)
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lab)
        if len(ys) >= 10:
            axes[1].text(xs.mean(), ys.mean(), str(lab), color="yellow",
                         fontsize=8, ha="center", weight="bold")
    axes[1].set_title(f"Predicted inundation ({n_kept} patches)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.replace('.tif', '')}_count.png", dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unet_v3")
    ap.add_argument("--model", default="unet", choices=sorted(FACTORIES))
    ap.add_argument("--splits", nargs="+", default=["valid", "test", "bolivia"])
    ap.add_argument("--min-area-px", type=int, default=10, help="噪声下限：10px = 1000㎡")
    ap.add_argument("--fig-chips", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(CKPT_DIR / f"{args.tag}_best.pth", map_location=device, weights_only=True)
    model = FACTORIES[args.model]().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"模型 {args.tag}（epoch {ckpt['epoch']}）加载完毕，min_area={args.min_area_px}px")

    fig_dir = OUT / "figures/counting"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, figs_done = [], 0
    for split in args.splits:
        ds = Sen1FloodsDataset(split, augment=False)
        for name, pred, gt, valid in predict_mask(model, ds, device):
            n_p, areas_p, labels = count_components(pred, args.min_area_px)
            n_g, areas_g, _ = count_components(gt, args.min_area_px)
            a_p = sum(areas_p) * PX_AREA_M2 / 1e6  # km²
            a_g = sum(areas_g) * PX_AREA_M2 / 1e6
            rows.append({
                "split": split, "chip": name,
                "n_pred": n_p, "n_gt": n_g,
                "count_err": n_p - n_g,
                "count_rel_err": (n_p - n_g) / n_g if n_g > 0 else np.nan,
                "area_pred_km2": a_p, "area_gt_km2": a_g,
                "area_rel_err": (a_p - a_g) / a_g if a_g > 1e-6 else np.nan,
            })
            if figs_done < args.fig_chips and n_g >= 2:  # 挑有多个斑块的图更有代表性
                i = [j for j, s in enumerate(ds.samples) if s[0] == name][0]
                img, _, _, _ = ds[i]
                rgb01 = torch.stack([img[0], img[1], img[0]], dim=-1).numpy()
                flood_map_figure(rgb01, gt, pred, labels, n_p, name, fig_dir)
                figs_done += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "counting_eval.csv", index=False)

    print("\n===== 淹没斑块计数与面积统计 =====")
    for split, g in df.groupby("split"):
        r = np.corrcoef(g.n_pred, g.n_gt)[0, 1] if g.n_gt.nunique() > 1 else np.nan
        print(f"[{split:8s}] chips {len(g):3d} | 斑块数 MAE {g.count_err.abs().mean():5.1f} "
              f"| 计数相关系数 r={r:.3f} | 面积 MAPE {100*g.area_rel_err.abs().mean():5.1f}% "
              f"| 预测总面积 {g.area_pred_km2.sum():7.2f} km² vs 真值 {g.area_gt_km2.sum():7.2f} km²")
    print(f"\n逐 chip 明细 -> outputs/counting_eval.csv；专题图 {figs_done} 张 -> outputs/figures/counting/")


if __name__ == "__main__":
    main()
