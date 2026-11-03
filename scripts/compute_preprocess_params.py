#!/usr/bin/env python3
"""计算分通道预处理参数：train 集每个 chip 的 vv/vh 1%/99% 分位数，取中位数存盘

背景：SAR 的 vv 与 vh 极化回波强度天然差 5~10dB，共用一套裁剪范围会把 vh 压扁
（调优第 2 轮修复项）。用 .venv/bin/python 运行。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

DATA = Path.home() / "datasets/sen1floods11"
OUT = Path(__file__).resolve().parents[1] / "outputs/preprocess_params.json"

df = pd.read_csv(DATA / "splits/flood_handlabeled/flood_train_data.csv",
                 header=None, names=["s1", "label"])
vv_p, vh_p = [], []
for s1f, labf in zip(df.s1, df.label):
    im = tifffile.imread(DATA / "data/flood_events/HandLabeled/S1Hand" / s1f).astype(np.float32)
    lab = tifffile.imread(DATA / "data/flood_events/HandLabeled/LabelHand" / labf)
    m = (lab != -1) & ~np.isnan(im).any(axis=0)  # 双条件有效掩膜内取分位数
    if m.any():
        vv_p.append(np.nanpercentile(im[0][m], [1, 99]))
        vh_p.append(np.nanpercentile(im[1][m], [1, 99]))

vv, vh = np.median(vv_p, axis=0), np.median(vh_p, axis=0)
params = {
    "clip_p1_vv": float(vv[0]), "clip_p99_vv": float(vv[1]),
    "clip_p1_vh": float(vh[0]), "clip_p99_vh": float(vh[1]),
}
OUT.write_text(json.dumps(params, indent=2))
print(f"train chips: {len(vv_p)}")
print(f"vv 裁剪范围: [{vv[0]:.2f}, {vv[1]:.2f}] dB")
print(f"vh 裁剪范围: [{vh[0]:.2f}, {vh[1]:.2f}] dB")
print(f"已写入 {OUT}")
