#!/usr/bin/env python3
"""SCI 出版级图表统一生成器（nature-figure 规范）

Figure contracts (one claim per figure):
  Fig.1 Dataset heterogeneity & dual-mask necessity  [image plate + quant]
      a vv  b vh  c dual valid mask  d GT overlay | e water-fraction hist  f mismatch hist
  Fig.2 Methodology: three generations on one stack  [schematic-led composite]
  Fig.3 Supervised training dynamics & data-centric ablation  [quantitative grid]
      a loss  b valid micro-IoU  c UNet ablation  d LoRA v1-vs-v2 loss (moved to Fig.6)
  Fig.4 Qualitative three-generation comparison  [image plate]
      rows: hardest/best test chip; cols: SAR, GT, UNet, SegFormer, SAM+LoRA
  Fig.5 Main results & counting reliability  [quantitative grid]
      a grouped IoU bars  b per-chip IoU dist  c patch-count scatter  d area scatter
  Fig.6 Foundation-model boundary & LoRA rescue  [quantitative + image]
      a LoRA v1/v2 loss  b zero-shot vs LoRA dumbbell  c Bolivia prompt qualitative

Usage: .venv/bin/python scripts/make_sci_figures.py  (GPU needed for Fig.4/Fig.6c)
Outputs: 报告/figures_sci/fig{1..6}_*.pdf + .png(300dpi) + QA json
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/ctgu53/.claude/skills/nature-figure/scripts")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "legend.handletextpad": 0.8,
    "figure.dpi": 110, "savefig.dpi": 300,
})

C_UNET, C_SEGF, C_SAMZ, C_SAML, C_ACC, C_WATER = (
    "#0072B2", "#009E73", "#8C8C8C", "#E69F00", "#D55E00", (0.0, 0.4, 1.0))
OUT = ROOT / "报告" / "figures_sci"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = dict(fontsize=9, fontweight="bold")


def panel_label(ax, letter, dx=-0.14, dy=1.12):
    ax.text(dx, dy, letter, transform=ax.transAxes, ha="left", va="bottom", **PANEL)


FIG_VERSION = "v1.0"  # 版本规范：改图升版本号，一图一文件夹


def export(fig, name, panels=True):
    """契约出口：先对齐门，后导出 PDF+PNG（一图一夹 + 版本号）"""
    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    name = f"{name}/{name}_{FIG_VERSION}"
    if panels:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT / f"{name}.alignment.json"),
            overlay_svg=str(OUT / f"{name}.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[export] {name}")


def panel_label_img(ax, letter):
    ax.text(-0.13, 1.12, letter, transform=ax.transAxes, ha="left", va="bottom", **PANEL)


# ---------------------------------------------------------------- Fig.1
def fig1_dataset():
    from src.dataset import Sen1FloodsDataset
    ds = Sen1FloodsDataset("test", augment=False)
    img, water, valid, name = ds[4]
    stats = pd.read_csv(ROOT / "outputs/data_summary.csv")

    fig = plt.figure(figsize=(7.09, 5.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.30)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    for ax, data, ttl in [
        (axs[0], img[0], f"{name.split('_')[0]} chip: VV (normalized)"),
        (axs[1], img[1], "VH (normalized)"),
        (axs[2], valid.numpy().astype(float), "Dual valid mask (white = valid)"),
    ]:
        ax.imshow(data, cmap="gray", interpolation="nearest")
        ax.set_title(ttl, pad=4)
    axs[3].imshow(img[0], cmap="gray")
    ov = np.zeros((*valid.shape, 4))
    ov[water[0] > 0] = [*C_WATER, 0.55]
    axs[3].imshow(ov, interpolation="nearest")
    axs[3].contour(valid.numpy(), levels=[0.5], colors="#D55E00", linewidths=0.5)
    axs[3].set_title("Reference water (blue)", pad=4)
    for ax in axs[:4]:
        ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6)

    order = ["train", "valid", "test", "bolivia"]
    cols = ["#4C4C4C", "#7A7A7A", "#A8A8A8", C_ACC]
    for x, sp in enumerate(order):
        axs[4].hist(stats[stats.split == sp].water_pct, bins=np.arange(0, 60, 2.5),
                    color=cols[x], alpha=0.75, label=sp)
    axs[4].set_xlabel("Water fraction per chip (%)")
    axs[4].set_ylabel("Chips"); axs[4].legend(ncol=2, handlelength=1.2)
    axs[4].set_title("Water is sparse and skewed", pad=4)

    for x, sp in enumerate(order):
        axs[5].hist(stats[stats.split == sp].mismatch_pct, bins=np.arange(0, 40, 1.5),
                    color=cols[x], alpha=0.75)
    axs[5].set_xlabel("NaN ∖ label-invalid mismatch (%)")
    axs[5].set_ylabel("Chips")
    axs[5].set_title("NaN and label-invalid regions disagree", pad=4)
    axs[5].axvline(stats[stats.split != "bolivia"].mismatch_pct.mean(),
                   color=C_ACC, lw=0.9, ls="--")
    for i, ax in enumerate(axs):
        panel_label_img(ax, "abcdef"[i] if i < 4 else "ef"[i - 4])
    export(fig, "fig1_dataset")


# ---------------------------------------------------------------- Fig.2
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(7.09, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec="#333333", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.8, color="#333333"))

    box(1, 26, 16, 12, "Sen1Floods11\n446 hand-labelled chips\nSAR VV/VH · 10 m/px", "#E8EDF3")
    box(21, 26, 18, 12, "Pre-processing\nper-channel dB clip\nnorm · dual valid mask", "#E8F2EA")
    box(43.5, 26, 14, 12, "Official splits\ntrain 252 / val 89\ntest 90 / Bolivia 15", "#E8EDF3")
    arrow(17, 32, 21, 32); arrow(39, 32, 43.5, 32)

    box(3, 8, 19, 11, "Gen-1 · UNet (hand-written)\n31 M · from scratch", "#D9E8F5")
    box(25.5, 8, 19, 11, "Gen-2 · SegFormer-B2\n27 M · ImageNet pre-trained", "#DCF0E6")
    box(48, 8, 21, 11, "Gen-3 · SAM ViT-B\n94 M · zero-shot + LoRA (1.0%)", "#FBEBD5")
    for xm in (12.5, 35, 58.5):
        arrow(50, 26, xm, 19.5)

    box(72.5, 26, 17, 12, "masked BCE + Dice\nidentical loss & metric", "#F3E8EC")
    box(73, 8, 17, 11, "Connected components\npatches ≥ 1000 m²\narea at 100 m² per px", "#EDE8F5")
    box(84, -3.5, 15, 11, "Inundation statistics\npatch count · km²", "#E8EDF3")
    arrow(89.5, 26, 84, 19.5); arrow(81.5, 8, 89, 8); arrow(58.5, 13.5, 73, 13.5)
    ax.text(50, 38.6, "Three generations of segmentation paradigms on one data / loss / metric stack",
            ha="center", fontsize=7.5, style="italic")
    export(fig, "fig2_pipeline", panels=False)


# ---------------------------------------------------------------- Fig.3
def fig3_training():
    H = {t: pd.read_csv(ROOT / "outputs" / f"{t}_history.csv")
         for t in ["unet", "unet_v2", "unet_v3", "segformer_b2"]}
    fig, axs = plt.subplots(1, 4, figsize=(7.09, 1.95))

    for t, c in zip(["unet", "unet_v2", "unet_v3", "segformer_b2"],
                    ["#9EC3E3", "#5E93C8", C_UNET, C_SEGF]):
        axs[0].plot(H[t].epoch, H[t].loss, color=c, lw=0.9,
                    label={"unet": "v1 (80 ep)", "unet_v2": "v2 (250 ep)",
                           "unet_v3": "v3 (+ch-norm)", "segformer_b2": "SegFormer"}[t])
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Training loss")
    axs[0].legend(handlelength=1.2, labelspacing=0.25)

    for t, c in zip(["unet", "unet_v2", "unet_v3", "segformer_b2"],
                    ["#9EC3E3", "#5E93C8", C_UNET, C_SEGF]):
        axs[1].plot(H[t].epoch, H[t].val_micro_iou, color=c, lw=0.9)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation micro-IoU")
    axs[1].set_ylim(0.3, 0.72)

    best = [H[t].loc[H[t].val_macro_iou.idxmax()].val_micro_iou for t in ["unet", "unet_v2", "unet_v3"]]
    bars = axs[2].bar(["v1", "v2", "v3"], best, width=0.62,
                      color=["#9EC3E3", "#5E93C8", C_UNET])
    for b, v in zip(bars, best):
        axs[2].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", fontsize=6)
    axs[2].set_ylabel("Best valid micro-IoU"); axs[2].set_ylim(0.60, 0.69)
    axs[2].set_xlabel("UNet iterations")
    axs[2].set_title("Longer epochs vs data fix", pad=4)

    per = pd.read_csv(ROOT / "outputs" / "counting_eval.csv")
    for sp, c in zip(["valid", "test", "bolivia"], ["#7A7A7A", "#4C4C4C", C_ACC]):
        g = per[per.split == sp]
        axs[3].scatter(g.n_gt, g.n_pred, s=5, alpha=0.55, color=c, label=sp)
    lim = [0, per[["n_gt", "n_pred"]].max().max()]
    axs[3].plot(lim, lim, color="#222222", lw=0.7, ls="--")
    axs[3].set_xlabel("Reference patch count"); axs[3].set_ylabel("Predicted")
    axs[3].legend(handlelength=1.0, labelspacing=0.25)
    for i, ax in enumerate(axs):
        panel_label(ax, "abcd"[i], dx=-0.24 if i in (0, 1) else -0.16)
    fig.subplots_adjust(wspace=0.42)
    export(fig, "fig3_training")


# ---------------------------------------------------------------- Fig.4
def fig4_qualitative():
    import torch  # noqa
    from peft import PeftModel
    from transformers import SamModel, SamProcessor
    from src.dataset import Sen1FloodsDataset
    from src.models import FACTORIES

    dev = "cuda"
    def load(tag, key):
        ck = torch.load(ROOT / "outputs/checkpoints" / f"{tag}_best.pth",
                        map_location=dev, weights_only=True)
        m = FACTORIES[key]().to(dev).eval(); m.load_state_dict(ck["model"]); return m

    unet = load("unet_v3", "unet"); segf = load("segformer_b2", "segformer")
    sam = PeftModel.from_pretrained(SamModel.from_pretrained("facebook/sam-vit-base"),
                                    str(ROOT / "outputs/checkpoints/sam_lora_v2_adapter")).to(dev).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    ds = Sen1FloodsDataset("test", augment=False)
    per = pd.read_csv(ROOT / "outputs/unet_v3_eval_per_chip.csv")
    tp = per[per.split == "test"].sort_values("iou").reset_index(drop=True)
    tp2 = tp[tp.iou > 0.05].reset_index(drop=True)
    picks = [tp2.chip[0], tp.chip.iloc[-1]]

    fig, axs = plt.subplots(2, 5, figsize=(7.09, 3.15))
    for r, name in enumerate(picks):
        i = [j for j, s in enumerate(ds.samples) if s[0] == name][0]
        img, water, valid, _ = ds[i]
        v = valid.numpy(); rgb = torch.stack([img[0], img[1], img[0]], -1).numpy()
        gt = (water[0].numpy() > 0.5) * v
        x = img.unsqueeze(0).to(dev)
        with torch.no_grad():
            pu = ((torch.sigmoid(unet(x))[0, 0].cpu().numpy() > 0.5) * v)
            ps = ((torch.sigmoid(segf(x))[0, 0].cpu().numpy() > 0.5) * v)
            ys, xs_ = np.where(gt > 0) if gt.any() else (np.array([256]), np.array([256]))
            inputs = proc(rgb, input_points=[[[int(xs_.mean()), int(ys.mean())]]],
                          return_tensors="pt").to(dev)
            out = sam(**inputs, multimask_output=False)
            pm = out.pred_masks.reshape(-1, 1, out.pred_masks.shape[-2], out.pred_masks.shape[-1])
            pk = torch.nn.functional.interpolate(pm.float(), size=(512, 512), mode="bilinear",
                                                 align_corners=False).sigmoid()[0, 0].cpu().numpy()
            pk = (pk > 0.5) * v
        row = [rgb[..., 0], gt, pu, ps, pk]
        ttl = ["SAR (VV)", "Reference", "UNet", "SegFormer-B2", "SAM + LoRA"]
        for c, (m, t) in enumerate(zip(row, ttl)):
            axs[r][c].imshow(rgb[..., 0], cmap="gray")
            if c > 0:
                ov = np.zeros((*v.shape, 4))
                ov[m > 0] = [*(C_WATER if c == 1 else (0.85, 0.33, 0.1)), 0.5]
                axs[r][c].imshow(ov, interpolation="nearest")
                axs[r][c].contour(gt, levels=[0.5], colors="#00E5FF", linewidths=0.45)
            axs[r][c].set_aspect("auto")
            axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
            if r == 0:
                axs[r][c].set_title(t, pad=3)
            for s in axs[r][c].spines.values():
                s.set_visible(True); s.set_linewidth(0.5)
        axs[r][0].set_ylabel(["hardest chip", "best chip"][r], fontsize=6.5)
        axs[r][2].plot(xs_.mean(), ys.mean(), "+", color="#FFD700", ms=7, mew=0.9)
    fig.subplots_adjust(wspace=0.06, hspace=0.14)
    for i in range(5):
        panel_label(axs[0][i], "abcde"[i], dx=-0.10, dy=1.12)
    export(fig, "fig4_qualitative")


# ---------------------------------------------------------------- Fig.5
def fig5_results():
    ev = {}
    for tag in ["unet_v3", "segformer_b2"]:
        e = pd.read_csv(ROOT / "outputs" / f"{tag}_eval.csv").set_index("split")
        ev[tag] = e
    z = pd.read_csv(ROOT / "outputs/sam_zeroshot_eval.csv")
    l = pd.read_csv(ROOT / "outputs/sam_lora_v2_eval.csv")
    models = [("UNet", C_UNET, ev["unet_v3"]),
              ("SegFormer", C_SEGF, ev["segformer_b2"]),
              ("SAM-0", C_SAMZ, z), ("SAM+LoRA", C_SAML, l)]

    fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.0))
    x = np.arange(4); w = 0.38
    te = [ev["unet_v3"].loc["test", "micro_iou"], ev["segformer_b2"].loc["test", "micro_iou"],
          z[z.split == "test"].best.mean(), l[l.split == "test"].best.mean()]
    bo = [ev["unet_v3"].loc["bolivia", "micro_iou"], ev["segformer_b2"].loc["bolivia", "micro_iou"],
          z[z.split == "bolivia"].best.mean(), l[l.split == "bolivia"].best.mean()]
    axs[0].bar(x - w / 2, te, w, color=[m[1] for m in models], label="Test (in-dist.)")
    axs[0].bar(x + w / 2, bo, w, color=[m[1] for m in models], alpha=0.45, label="Bolivia (OOD)")
    for xi, v in zip(x - w / 2, te):
        axs[0].text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
    for xi, v in zip(x + w / 2, bo):
        axs[0].text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
    axs[0].set_xticks(x); axs[0].set_xticklabels(["UNet", "SegF.", "SAM-0", "SAM+L"], rotation=35, ha="right")
    axs[0].set_ylabel("Water micro-IoU"); axs[0].set_ylim(0, 0.93)
    axs[0].legend(handlelength=1.0, labelspacing=0.25, loc="upper right")

    data, labels, colors = [], [], []
    for nice, c, src in [("UNet", C_UNET, "unet_v3"), ("SegF.", C_SEGF, "segformer_b2")]:
        pc = pd.read_csv(ROOT / "outputs" / f"{src}_eval_per_chip.csv")
        pc = pc[pc.split == "test"].iou.values
        data.append(pc); labels.append(nice); colors.append(c)
    data.append(z[z.split == "test"].best.values); labels.append("SAM-0"); colors.append(C_SAMZ)
    data.append(l[l.split == "test"].best.values); labels.append("SAM+L"); colors.append(C_SAML)
    bp = axs[1].boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#222222", lw=0.8),
                        whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_linewidth(0.7)
    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip(data, colors)):
        axs[1].scatter(rng.normal(i + 1, 0.06, len(d)), d, s=3, color=c, alpha=0.5, lw=0)
    axs[1].set_xticklabels(labels, rotation=35, ha="right")
    axs[1].set_ylabel("Per-chip IoU (test)"); axs[1].set_ylim(0, 1.0)

    per = pd.read_csv(ROOT / "outputs/counting_eval.csv")
    g = per[per.split == "test"]
    ga = g[(g.area_gt_km2 > 0) & (g.area_pred_km2 > 0)]
    axs[2].scatter(g.n_gt, g.n_pred, s=6, alpha=0.6, color=C_UNET)
    lim = [0, max(g.n_gt.max(), g.n_pred.max()) * 1.05]
    axs[2].plot(lim, lim, color="#222222", lw=0.7, ls="--")
    r = np.corrcoef(g.n_gt, g.n_pred)[0, 1]
    axs[2].set_xlabel("Reference patches"); axs[2].set_ylabel("Predicted")
    axs[2].set_title(f"Patch count, r = {r:.2f}", pad=4)

    axs[3].scatter(ga.area_gt_km2, ga.area_pred_km2, s=6, alpha=0.6, color=C_UNET)
    axs[3].set_xscale("log"); axs[3].set_yscale("log")
    ticks = [0.01, 0.1, 1, 10]
    tl = ["0.01", "0.1", "1", "10"]
    axs[3].set_xticks(ticks); axs[3].set_xticklabels(tl)
    axs[3].set_yticks(ticks); axs[3].set_yticklabels(tl)
    lim = [8e-3, max(ga.area_gt_km2.max(), ga.area_pred_km2.max()) * 1.4]
    axs[3].plot(lim, lim, color="#222222", lw=0.7, ls="--")
    bias = 100 * (g.area_pred_km2.sum() / g.area_gt_km2.sum() - 1)  # 总量仍按全部样本
    axs[3].set_xlabel("Reference area (km²)"); axs[3].set_ylabel("Predicted (km²)")
    axs[3].set_title(f"Total-area bias {bias:+.1f}%", pad=4)
    for i, ax in enumerate(axs):
        panel_label(ax, "abcd"[i], dx=-0.24 if i in (0, 1, 2, 3) else -0.16)
    fig.subplots_adjust(wspace=0.46)
    export(fig, "fig5_results")


# ---------------------------------------------------------------- Fig.6
def fig6_sam():
    h1 = pd.read_csv(ROOT / "outputs/sam_lora_history.csv")
    h2 = pd.read_csv(ROOT / "outputs/sam_lora_v2_history.csv")
    z = pd.read_csv(ROOT / "outputs/sam_zeroshot_eval.csv")
    l = pd.read_csv(ROOT / "outputs/sam_lora_v2_eval.csv")

    fig = plt.figure(figsize=(7.09, 2.05))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.42)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))

    ax0.plot(h1.epoch, h1.loss, color=C_SAMZ, lw=0.9, label="v1: 47 k params (mis-targeted)")
    ax0.plot(h2.epoch, h2.loss, color=C_SAML, lw=0.9, label="v2: 0.98 M params (fixed)")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Training loss")
    ax0.legend(handlelength=1.2, labelspacing=0.25)

    for sp, off, mkr in [("test", 0, "o"), ("bolivia", 0.22, "s")]:
        zb = z[z.split == sp].best.mean(); lb = l[l.split == sp].best.mean()
        ax1.plot([0, 1], [zb, lb], color=C_ACC if sp == "test" else "#7A7A7A",
                 lw=1.0, marker=mkr, ms=3.5,
                 label=f"{sp}: {zb:.2f} → {lb:.2f}")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Zero-shot", "+ LoRA"])
    ax1.set_ylabel("Best point-prompted IoU"); ax1.set_xlim(-0.35, 1.35)
    ax1.set_ylim(0, 0.42)
    ax1.legend(handlelength=1.0, labelspacing=0.25)
    ax1.set_title("1.0% parameters, 2.0× gain", pad=4)

    import torch
    from peft import PeftModel
    from transformers import SamModel, SamProcessor
    from src.dataset import Sen1FloodsDataset
    sam = PeftModel.from_pretrained(SamModel.from_pretrained("facebook/sam-vit-base"),
                                    str(ROOT / "outputs/checkpoints/sam_lora_v2_adapter")).to("cuda").eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    ds = Sen1FloodsDataset("bolivia", augment=False)
    img, water, valid, name = ds[2]
    v = valid.numpy(); rgb = torch.stack([img[0], img[1], img[0]], -1).numpy()
    ys, xs = np.where((water[0] > 0).numpy() & v)
    if len(ys) == 0:
        ys, xs = np.array([256]), np.array([256])
    inputs = proc(rgb, input_points=[[[int(xs.mean()), int(ys.mean())]]],
                  return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = sam(**inputs, multimask_output=False)
        pm = out.pred_masks.reshape(-1, 1, out.pred_masks.shape[-2], out.pred_masks.shape[-1])
        pk = torch.nn.functional.interpolate(pm.float(), size=(512, 512), mode="bilinear",
                                             align_corners=False).sigmoid()[0, 0].cpu().numpy()
    pk = (pk > 0.5) * v; gt = (water[0].numpy() > 0.5) * v
    ax2.imshow(rgb[..., 0], cmap="gray")
    ov = np.zeros((*v.shape, 4)); ov[pk > 0] = (0.85, 0.33, 0.1, 0.5)
    ax2.imshow(ov, interpolation="nearest")
    ax2.contour(gt, levels=[0.5], colors="#00E5FF", linewidths=0.5)
    ax2.plot(xs.mean(), ys.mean(), "+", color="#FFD700", ms=9, mew=1.0)
    ax2.set_aspect("auto"); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"SAM+LoRA on {name.split('_')[0]} (cyan = reference)", pad=4)
    for s in ax2.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    panel_label(ax0, "a", dx=-0.22); panel_label(ax1, "b", dx=-0.22)
    panel_label(ax2, "c", dx=-0.16)
    export(fig, "fig6_sam")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    fig1_dataset(); fig2_pipeline(); fig3_training()
    fig5_results(); fig6_sam(); fig4_qualitative()
    print("[done] 全部 SCI 图表 ->", OUT)
