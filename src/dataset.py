"""Sen1Floods11 数据管道：读取、预处理、双条件有效掩膜、数据增强

预处理方案（由 notebooks/01_data_exploration.ipynb 全量统计定稿）：
- dB 裁剪到 [p1, p99]（train 集分位数中位数，存于 outputs/preprocess_params.json）后归一化 [0,1]
- NaN 像素归一化前置 0
- 有效掩膜 = (标注 != -1) & 影像非 NaN —— 两类无效互不完全包含，必须同时排除
返回 (img, water, valid, chip名)：
- img:   (2, H, W) float32，已归一化
- water: (1, H, W) float32，1=水（训练目标）
- valid: (H, W) bool，损失/指标只在这些像素上计算
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

DATA = Path.home() / "datasets/sen1floods11"


class Sen1FloodsDataset(Dataset):
    def __init__(self, split: str = "train", augment: bool = False, params_path: Path | None = None):
        assert split in {"train", "valid", "test", "bolivia"}
        self.img_dir = DATA / "data/flood_events/HandLabeled/S1Hand"
        self.lab_dir = DATA / "data/flood_events/HandLabeled/LabelHand"
        split_file = DATA / f"splits/flood_handlabeled/flood_{split}_data.csv"
        df = pd.read_csv(split_file, header=None, names=["s1", "label"])
        self.samples = list(zip(df.s1, df.label))
        self.split = split
        self.augment = augment

        if params_path is None:
            params_path = Path(__file__).resolve().parents[1] / "outputs/preprocess_params.json"
        params = json.loads(Path(params_path).read_text())
        # 分通道裁剪范围（vv/vh 回波强度差 5~10dB，不能共用一套，见调优记录）
        self.clip_lo = np.array([params["clip_p1_vv"], params["clip_p1_vh"]], dtype=np.float32)[:, None, None]
        self.clip_hi = np.array([params["clip_p99_vv"], params["clip_p99_vh"]], dtype=np.float32)[:, None, None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s1f, labf = self.samples[i]
        img = tifffile.imread(self.img_dir / s1f).astype(np.float32)  # (2, H, W) dB 值
        lab = tifffile.imread(self.lab_dir / labf)  # (H, W) int16: -1/0/1

        nan_mask = np.isnan(img).any(axis=0)  # (H, W) 任一通道缺测即视为无效
        img = np.clip(img, self.clip_lo, self.clip_hi)
        img = (img - self.clip_lo) / (self.clip_hi - self.clip_lo)
        img = np.nan_to_num(img, nan=0.0)

        water = (lab == 1).astype(np.float32)
        valid = (lab != -1) & ~nan_mask  # ★ 双条件有效掩膜

        if self.augment:  # 图像与两个掩膜必须同步变换
            if np.random.rand() < 0.5:  # 水平翻转
                img, water, valid = np.flip(img, 2), np.flip(water, 1), np.flip(valid, 1)
            if np.random.rand() < 0.5:  # 垂直翻转
                img, water, valid = np.flip(img, 1), np.flip(water, 0), np.flip(valid, 0)
            k = np.random.randint(4)  # 旋转 90°×k
            img = np.rot90(img, k, axes=(1, 2))
            water = np.rot90(water, k, axes=(0, 1))
            valid = np.rot90(valid, k, axes=(0, 1))
            img, water, valid = np.ascontiguousarray(img), np.ascontiguousarray(water), np.ascontiguousarray(valid)

        return (
            torch.from_numpy(img),
            torch.from_numpy(water).unsqueeze(0),
            torch.from_numpy(valid),
            s1f,
        )
