import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super(DeepLabV3Plus, self).__init__()
        
        # Load Pretrained DeepLabV3 (Standard) to get Backbone + ASPP
        # We will extract the backbone and classifier(ASPP) from it.
        # Alternatively, we can load just ResNet, but loading deeplab gives us a pre-trained ASPP head too? 
        # Actually, torchvision's deeplab is trained on COCO (21 classes). 
        # Using ResNet50 backbone directly is cleaner for "Fine-Tuning" methodology (ImageNet -> Cityscapes).
        
        # Let's use the torchvision DeepLabv3_ResNet50 model as a base container for backbone + ASPP
        # But we need access to low-level features (Layer 1) for V3+.
        # Standard torchvision DeepLab model returns a dict {'out': ...}.
        # It's better to implement the class wrapping the backbone components.

        # 1. Backbone: ResNet-50
        # We need "layer 1" (low level) and "layer 4" (high level)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        # We load a standard resnet to easily access layers
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=weights)
        
        # Eliminate FC and AvgPool to turn into FCN
        del self.backbone.avgpool
        del self.backbone.fc
        
        # IMPORTANT: DeepLab usually uses Dilated Convolutions (atrous) in the backbone
        # ResNet layer 3 and 4 should have dilation.
        # Torchvision's `deeplabv3_resnet50` does this automatically. 
        # Let's reproduce that: replace strides with dilation.
        self.backbone.layer3[0].conv2.dilation = (2, 2)
        self.backbone.layer3[0].conv2.padding = (2, 2)
        self.backbone.layer3[0].downsample[0].stride = (1, 1) # Don't downsample spacially
        
        self.backbone.layer4[0].conv2.dilation = (4, 4)
        self.backbone.layer4[0].conv2.padding = (4, 4)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        # But doing this patching manually is error-prone. 
        # Better: Use `torchvision.models.resnet50(replace_stride_with_dilation=[False, True, True])`
        from torchvision.models import resnet50
        self.backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        
        # 2. ASPP (Atrous Spatial Pyramid Pooling)
        # Input: 2048 channels (from ResNet Layer 4)
        # Output: 256 channels
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # 3. Decoder
        # Low-level feature projection (from Layer 1 - 256 channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        # Concat(ASPP_Upsampled [256] + LowLevel [48]) = 304 channels
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
        
        # Forward Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        low_level_feat = self.backbone.layer1(x) # 256 ch, 1/4 res
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # 2048 ch, 1/16 res (due to dilation)
        
        # ASPP (Encoder Head)
        x = self.aspp(x) # 256 ch
        
        # Decoder
        # Upsample ASPP features to matches low-level features (1/4 res)
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # Project low-level features
        low_level = self.low_level_conv(low_level_feat)
        
        # Concat
        x = torch.cat([x, low_level], dim=1)
        
        # Refine
        x = self.decoder_conv(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Final Upsample to original size (4x upsample from 1/4 to 1/1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
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
            nn.BatchNorm2d(out_channels),
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
            # Image Pooling needs upsampling back
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
        model.eval() # Switch to eval mode to avoid BN error with BS=1 and 1x1 pooling
        # Use a slightly larger input to ensure no 1x1 issues
        dummy = torch.randn(1, 3, 512, 1024)
        print(f"Input shape: {dummy.shape}")
        out = model(dummy)
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 19, 512, 1024)
        print("DeepLabV3+ Sanity Check Passed")
    except Exception as e:
        import traceback
        traceback.print_exc()
