import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, num_classes=19, pretrained_backbone=True):
        super(SegFormer, self).__init__()
        
        # Use mit-b0 (Lightest) - Pretrained on ImageNet
        # "nvidia/mit-b0" is just the encoder. 
        # "nvidia/segformer-b0-finetuned-ade-512-512" is a full model.
        # We want to fine-tune, so starting from a pre-trained semantic segmentation model (on ADE20k) 
        # is often better than just ImageNet encoder, as the decoder is initialized too.
        # However, to be strict "ImageNet -> Cityscapes", we should ideally use just the encoder.
        # But `SegformerForSemanticSegmentation` handles both.
        # Let's use `nvidia/mit-b0` and let the head be random initialized?
        # Actually `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")` will warn that decoder is random.
        # VALID approach for "Fine-Tuning".
        
        model_name = "nvidia/mit-b0" if pretrained_backbone else "nvidia/mit-b0" 
        
        # We need to tell it the number of classes. 
        # Providing num_labels=19 and ignore_mismatched_sizes=True allows loading encoder weights 
        # while reshaping/initializing the head for 19 classes.
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            reshape=True, 
        )
        
    def forward(self, x):
        # SegFormer expects output size to match input, but internally it outputs H/4.
        # The library's `forward` usually returns a `SemanticSegmenterOutput`.
        # The `logits` in that output are upsampled to input size automatically?
        # CHECK DOCS: SegFormer model outputs logits of shape (batch, num_labels, height/4, width/4).
        # We need to upsample 4x.
        
        outputs = self.model(pixel_values=x)
        logits = outputs.logits # (B, 19, H/4, W/4)
        
        # Upsample to full input resolution
        # Note: 'x' is the input tensor
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
