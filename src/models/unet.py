import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [InstanceNorm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Using bilinear upsampling followed by a conv layer to reduce channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # x1 is from upsampling, x2 is from skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, base_c=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (5 levels total: Input -> L1 -> L2 -> L3 -> L4 -> Bottle)
        # L1
        self.inc = DoubleConv(n_channels, base_c)
        # L2
        self.down1 = Down(base_c, base_c * 2)
        # L3
        self.down2 = Down(base_c * 2, base_c * 4)
        # L4
        self.down3 = Down(base_c * 4, base_c * 8)
        # L5 (Bottleneck)
        self.down4 = Down(base_c * 8, base_c * 16)
        
        # Decoder (4 upsampling steps)
        # Up 1: 1024 + 512 -> 512
        self.up1 = Up(base_c * 16 + base_c * 8, base_c * 8)
        # Up 2: 512 + 256 -> 256
        self.up2 = Up(base_c * 8 + base_c * 4, base_c * 4)
        # Up 3: 256 + 128 -> 128
        self.up3 = Up(base_c * 4 + base_c * 2, base_c * 2)
        # Up 4: 128 + 64 -> 64
        self.up4 = Up(base_c * 2 + base_c, base_c)
        
        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=19)
    dummy_input = torch.randn(1, 3, 512, 1024)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 19, 512, 1024), "Output shape mismatch!"
    print("U-Net Sanity Check Passed!")
