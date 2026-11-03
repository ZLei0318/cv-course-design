"""纯 PyTorch 手写 UNet（课程要求：从零实现，不调用分割库）

经典结构：4 层下采样编码器 → 瓶颈 → 转置卷积上采样 + 跳跃连接解码器。
输入为 SAR 双通道 (vv, vh)，输出单通道 logits（水/非水二分类，训练时配 BCE）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 => BN => ReLU) × 2，UNet 的基本积木"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """最大池化下采样 + 双卷积"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """转置卷积上采样 + 编码器跳跃拼接 + 双卷积"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin, cout)  # cin = 上采样后的 cin/2 + 跳跃的 cin/2

    def forward(self, x, skip):
        x = self.up(x)
        # 输入尺寸非 16 整数倍时的兜底对齐
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """输入 (B, 2, H, W) → 输出 (B, 1, H, W) logits"""

    def __init__(self, in_ch: int = 2, base: int = 64, out_ch: int = 1):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)      # 512×512, b 通道
        self.enc2 = Down(b, b * 2)            # 256
        self.enc3 = Down(b * 2, b * 4)        # 128
        self.enc4 = Down(b * 4, b * 8)        # 64
        self.bottleneck = Down(b * 8, b * 16)  # 32
        self.dec4 = Up(b * 16, b * 8)
        self.dec3 = Up(b * 8, b * 4)
        self.dec2 = Up(b * 4, b * 2)
        self.dec1 = Up(b * 2, b)
        self.head = nn.Conv2d(b, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bt = self.bottleneck(e4)
        d4 = self.dec4(bt, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)


def build_unet() -> UNet:
    """模型工厂：train.py 通过统一入口构建"""
    return UNet(in_ch=2, base=64)


if __name__ == "__main__":
    # 自检：前向传播形状
    m = build_unet()
    n_params = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 2, 512, 512))
    print(f"UNet 参数量 {n_params/1e6:.1f}M, 输入 (2,2,512,512) → 输出 {tuple(y.shape)}")
