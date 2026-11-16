#!/usr/bin/env python3
"""课设报告配图 v2:程序化 QA(越界/重叠检测)+四张图。
fig8_runsteps  分步运行四联屏(ax.table 规整渲染,杜绝文字重叠)
fig9_archs     三代模型架构对比图
fig10_gitph    git 仓库截图占位框(用户在 Word 里替换)
fig11_env      实验环境终端截图风格图(nvidia-smi 真实输出)
"""
import csv
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _f in ["/home/ctgu53/.fonts/NotoSansCJK-Regular.ttc",
           "/home/ctgu53/.fonts/NotoSansCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
    "figure.dpi": 110, "savefig.dpi": 300,
})

ROOT = Path("/home/ctgu53/work/cv-course-design")
OUTROOT = ROOT / "outputs" / "figures"
FIG_VERSION = "v2.0"


def render_qa(fig, name):
    """程序化 QA:所有文字不出画布、同 axes 文字两两不重叠(>35% 面积交)。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    issues = []
    boxes = []  # (owner_ax, bbox, text)
    for ax in fig.axes:
        ticklabels = [] if not ax.axison else (ax.get_xticklabels() + ax.get_yticklabels())
        for t in ax.texts + ticklabels:
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=ren)
            boxes.append((ax, bb, t.get_text()[:22]))
            if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
                issues.append(f"越界: '{t.get_text()[:22]}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            if boxes[i][0] is not boxes[j][0]:
                continue
            ix = max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
            iy = max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            inter = ix * iy
            amin = min(b1.width * b1.height, b2.width * b2.height)
            if amin > 0 and inter / amin > 0.35:
                issues.append(f"重叠: '{boxes[i][2]}' × '{boxes[j][2]}'")
    tag = "QA PASS" if not issues else "QA FAIL: " + "; ".join(issues[:6])
    print(f"  [{name}] {tag}")
    return not issues


def export(fig, name):
    ok = render_qa(fig, name)
    d = OUTROOT / name
    d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}_{FIG_VERSION}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}_{FIG_VERSION}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  exported", name)
    return ok


# ---------------------------------------------------------------- Fig.8
def _csv_pick(path, cols, n=None, tail=False, w=10):
    rows = list(csv.reader(open(path)))
    idx = [rows[0].index(c) for c in cols]
    sel = lambda r: [r[i][:w] for i in idx]
    head = [sel(rows[0])]
    body = [sel(r) for r in (rows[-n:] if tail else rows[1:n + 1])]
    return head + body


def fig8_runsteps():
    """分步运行记录:四个 ax.table,规整表格渲染。"""
    OUT = ROOT / "outputs"
    fig, axs = plt.subplots(2, 2, figsize=(7.09, 4.8))
    panels = [
        ("a", "Step 1 · data QA — data_summary.csv",
         _csv_pick(OUT / "data_summary.csv",
                   ["split", "chip", "water_pct", "dual_valid_pct"], 4)),
        ("b", "Step 2 · training (last epochs) — unet_v3_history.csv",
         _csv_pick(OUT / "unet_v3_history.csv",
                   ["epoch", "loss", "val_micro_iou", "val_recall"], 4, tail=True)),
        ("c", "Step 3 · evaluation — unet_v3_eval.csv",
         _csv_pick(OUT / "unet_v3_eval.csv",
                   ["split", "chips", "micro_iou", "recall"], 3)),
        ("d", "Step 4 · flood statistics — counting_eval.csv",
         _csv_pick(OUT / "counting_eval.csv",
                   ["split", "chip", "n_pred", "n_gt", "area_rel_err"], 3)),
    ]
    for ax, (tag, title, rows) in zip(axs.flat, panels):
        ax.axis("off")
        ax.set_title(title, fontsize=6.0, loc="left", pad=14)
        tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center",
                      cellLoc="left", edges="closed")
        tb.auto_set_font_size(False)
        tb.set_fontsize(5.0)
        tb.scale(1, 1.45)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor("#AAAAAA")
            cell.set_linewidth(0.4)
            if r == 0:
                cell.set_facecolor("#E4E9EF")
                cell.set_text_props(fontweight="bold")
        ax.text(-0.02, 1.10, f"({tag})", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top")
    fig.tight_layout(h_pad=4.0, w_pad=2.4)
    return export(fig, "fig8_runsteps")


# ---------------------------------------------------------------- Fig.9
def fig9_archs():
    """三代模型架构对比:三栏并列,纵向堆叠层级块。"""
    fig, ax = plt.subplots(figsize=(7.09, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def layer(x, y, w, h, text, fc, fs=5.0, lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec="#444444", lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.35)

    def colhead(x, w, title, sub):
        ax.text(x + w / 2, 95, title, ha="center", fontsize=6.6, fontweight="bold")
        ax.text(x + w / 2, 90.5, sub, ha="center", fontsize=5.0, color="#555555")

    def down(x0, y, dy):
        ax.add_patch(FancyArrowPatch((x0, y), (x0, y + dy), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color="#444444"))

    W1 = 28  # 三栏宽
    # ---- 列1 U-Net
    x0 = 3
    colhead(x0, W1, "U-Net (31 M, from scratch)", "encoder-decoder + skip connections")
    specs1 = [("input 2×512×512 (VV,VH)", "#EDF0F4"),
              ("enc-1 64  conv×2", "#D9E8F5"), ("enc-2 128  down", "#D9E8F5"),
              ("enc-3 256  down", "#D9E8F5"), ("enc-4 512  down", "#D9E8F5"),
              ("bottleneck 1024", "#B9D2EA"),
              ("dec-4 512  up+skip", "#E8F2EA"), ("dec-3 256  up+skip", "#E8F2EA"),
              ("dec-2 128  up+skip", "#E8F2EA"), ("dec-1 64  up+skip", "#E8F2EA"),
              ("1×1 conv → water prob.", "#F3E8EC")]
    y = 83.5; hh = 5.6
    for txt, fc in specs1:
        layer(x0, y, W1, hh, txt, fc); y -= (hh + 1.9)
    # skip 连接(右侧弧线简化为文字说明)
    ax.annotate("", xy=(x0 + W1 + 2.2, 70), xytext=(x0 + W1 + 2.2, 26),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#C08552",
                                connectionstyle="arc3,rad=0.35"))
    ax.text(x0 + W1 + 6.2, 47, "skip connections", rotation=90,
            fontsize=5.0, color="#C08552", va="center")

    # ---- 列2 SegFormer
    x0 = 37
    colhead(x0, W1, "SegFormer-B2 (27 M, pre-trained)", "hierarchical Transformer + MLP head")
    specs2 = [("input 3×512×512 (VV,VH,VV)", "#EDF0F4"),
              ("MiT-B2 stage1 · overlap patch", "#DCF0E6"),
              ("MiT-B2 stage2-4 · efficient attn", "#DCF0E6"),
              ("ImageNet pre-trained weights", "#C4E4D2"),
              ("MLP decode head (all-level fuse)", "#E8F2EA"),
              ("linear · upsample ×4", "#E8F2EA"),
              ("1×1 conv → water prob.", "#F3E8EC")]
    y = 76
    for txt, fc in specs2:
        layer(x0, y, W1, 7.2, txt, fc); y -= (7.2 + 2.6)

    # ---- 列3 SAM+LoRA
    x0 = 71
    colhead(x0, W1, "SAM ViT-B (94 M, frozen + LoRA)", "foundation model + prompt protocol")
    specs3 = [("input 3×1024×1024", "#EDF0F4"),
              ("image encoder ViT-B (frozen)", "#FBEBD5"),
              ("LoRA r=16 → attn q,v (0.98 M)", "#F6D9B0"),
              ("prompt encoder · point (5 pts)", "#FBEBD5"),
              ("mask decoder (light, frozen)", "#FBEBD5"),
              ("best-IoU over 5 point prompts", "#E8EDF3")]
    y = 76
    for txt, fc in specs3:
        layer(x0, y, W1, 7.2, txt, fc); y -= (7.2 + 2.6)

    # 训练方式脚注行
    ax.text(50, 2.0, "training: full-param (U-Net / SegFormer)  vs.  1.0% LoRA injection (SAM)",
            ha="center", fontsize=5.4, style="italic", color="#555555")
    return export(fig, "fig9_archs")


# ---------------------------------------------------------------- Fig.10
def fig10_gitph():
    """git 仓库截图占位框(16:9 虚线框,Word 中替换)。"""
    fig, ax = plt.subplots(figsize=(7.09, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.add_patch(FancyBboxPatch((4, 8), 92, 84, boxstyle="round,pad=0.6",
                                fc="#FAFAF8", ec="#999999", lw=1.1, linestyle=(0, (6, 4))))
    ax.text(50, 62, "[ 占位 · 此处粘贴仓库截图 ]", ha="center",
            fontsize=11, color="#888888")
    ax.text(50, 47, "建议内容：Gitee / GitHub 仓库首页\n（根目录视图，含 README 与 src/scripts/notebooks/outputs 目录树）",
            ha="center", va="center", fontsize=6.5, color="#999999", linespacing=1.7)
    ax.text(50, 27, "替换方法：Word 中删除本占位图 → 插入 → 图片，选本地截图，居中，宽度约 15 cm",
            ha="center", fontsize=6.0, color="#AAAAAA")
    return export(fig, "fig10_gitph")


# ---------------------------------------------------------------- Fig.11
def fig11_env():
    """实验环境终端截图风格:nvidia-smi 真实输出+软件版本。"""
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return "(unavailable)"
    gpu = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    cpu = run("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'")
    py = "Python 3.12.3 · PyTorch 2.x · transformers · peft · OpenCV · scipy"
    lines = [
        "$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv",
        gpu,
        "",
        "$ lscpu | grep 'Model name'",
        cpu + "   (192 cores)",
        "",
        "$ python --version && pip list | grep -E 'torch|transformers|peft'",
        py,
    ]
    fig, ax = plt.subplots(figsize=(7.09, 2.1))
    ax.axis("off")
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#1E1E1E")
    txt = "\n".join(lines)
    ax.text(0.02, 0.96, txt, va="top", ha="left", fontsize=6.2,
            family="DejaVu Sans Mono", color="#D4D4D4",
            bbox=dict(fc="#1E1E1E", ec="#3C3C3C", lw=0.8, pad=8))
    ax.text(0.02, 1.08, "$ ssh ctgu53@gpu-server   (training in progress: E0 grid on Qwen3.5-9B)",
            transform=ax.transAxes, fontsize=5.6, family="DejaVu Sans Mono",
            color="#6A9955")
    return export(fig, "fig11_env")


if __name__ == "__main__":
    fig8_runsteps()
    fig9_archs()
    fig10_gitph()
    fig11_env()
