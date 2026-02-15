import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super(DeepLabV3Plus, self).__init__()
        
        # 1. Backbone: ResNet-50 with Dilated Convolutions
        # We need "layer 1" (low level) and "layer 4" (high level)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        
        # 2. ASPP (Atrous Spatial Pyramid Pooling)
        # Input: 2048 channels (from ResNet Layer 4) -> Output: 256 channels
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # 3. Decoder
        # Low-level feature projection (from Layer 1 - 256 channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Fusion: Concat(ASPP_Upsampled [256] + LowLevel [48]) = 304 channels
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        low_level_feat = self.backbone.layer1(x) # 256 ch, 1/4 res
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # 2048 ch, 1/8 res (due to dilation)
        
        # ASPP
        x = self.aspp(x) # 256 ch
        
        # Decoder: Upsample ASPP features to match low-level features (1/4 res)
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # Project low-level features
        low_level = self.low_level_conv(low_level_feat)
        
        # Concat & Refine
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder_conv(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Final Upsample to original size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]): # Atrous rates different from original paper
        super(ASPP, self).__init__()
        modules = []
        # 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 3x3 Atrous Convs
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Image Pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels), # Safe for BS=1
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                feat = conv(x)
                feat = F.interpolate(feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            else:
                feat = conv(x)
            res.append(feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

if __name__ == "__main__":
    try:
        model = DeepLabV3Plus(num_classes=19)
        model.eval()
        dummy = torch.randn(1, 3, 512, 1024)
        print(f"Input shape: {dummy.shape}")
        out = model(dummy)
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 19, 512, 1024)
        print("DeepLabV3+ Sanity Check Passed")
    except Exception as e:
        import traceback
        traceback.print_exc()
