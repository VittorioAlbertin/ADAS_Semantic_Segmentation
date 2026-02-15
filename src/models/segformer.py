import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super(SegFormer, self).__init__()
        
        # Use mit-b0 (Lightest)
        # We load the pretrained encoder/model but initialize the head for our num_classes
        model_name = "nvidia/mit-b0" if pretrained_backbone else "nvidia/mit-b0" 
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
    def forward(self, x):
        # SegFormer outputs logits of half the resolution (H/4, W/4)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits # (B, num_classes, H/4, W/4)
        
        # Upsample to full input resolution
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return logits

if __name__ == "__main__":
    try:
        model = SegFormer(num_classes=19)
        model.eval()
        dummy = torch.randn(1, 3, 512, 1024)
        print(f"Input shape: {dummy.shape}")
        out = model(dummy)
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 19, 512, 1024)
        print("SegFormer Sanity Check Passed")
    except Exception as e:
        import traceback
        traceback.print_exc()
