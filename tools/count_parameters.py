import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model, UNet, DeepLabV3Plus, SegFormer
from src.config import NUM_CLASSES

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def freeze_backbone(model, model_name):
    if model_name == "deeplab":
        # Freeze ResNet backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif model_name == "segformer":
        # Freeze MiT encoder
        for param in model.model.segformer.parameters():
            param.requires_grad = False
    elif model_name == "unet":
        print(f"  Note: UNet has no pre-trained backbone to freeze.")
        
def format_params(num):
    return f"{num / 1e6:.2f}M"

def main():
    models_to_test = ['unet', 'deeplab', 'segformer']
    
    print("="*90)
    print(f"{'Model':<15} {'Scenario':<20} {'Total Params':<15} {'Trainable Params':<20} {'% Trainable':<10}")
    print("="*90)
    
    for model_name in models_to_test:
        # 1. Unfrozen (Default)
        model = get_model(model_name, num_classes=NUM_CLASSES)
        total, trainable = count_params(model)
        print(f"{model_name:<15} {'Unfrozen':<20} {format_params(total):<15} {format_params(trainable):<20} {trainable/total*100:>9.1f}%")
        
        # 2. Frozen Backbone
        freeze_backbone(model, model_name)
        total_frozen, trainable_frozen = count_params(model)
        
        if trainable_frozen != trainable:
             print(f"{model_name:<15} {'Frozen Backbone':<20} {format_params(total_frozen):<15} {format_params(trainable_frozen):<20} {trainable_frozen/total_frozen*100:>9.1f}%")
        else:
             print(f"{model_name:<15} {'Frozen Backbone':<20} {'N/A (No backbone)':<35}")
             
        print("-" * 90)

if __name__ == "__main__":
    main()
