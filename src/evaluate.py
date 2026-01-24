import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.dataset import CityscapesDataset
from src.config import DATASET_ROOT, NUM_CLASSES, IGNORE_INDEX, DEVICE, FULL_SIZE
from src.models import get_model
from src.utils import get_cityscapes_colormap, denormalize, decode_segmap

# Mapping trainId to class name
CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def fast_hist(a, b, n):
    """
    Return confusion matrix between label 'a' and prediction 'b'
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def validate(model, val_loader, device, num_classes, max_samples=None, save_dir=None, save_num=0):
    """
    Runs validation with optional visualization and sample limiting.
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    
    total_samples = len(val_loader)
    if max_samples:
        total_samples = min(total_samples, max_samples)
    
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(val_loader, desc="Validating", leave=False, total=total_samples)):
            image = image.to(device)
            # label is (B, H, W)
            
            # Forward (AMP for memory safety)
            with torch.amp.autocast('cuda', enabled=True):
                output = model(image)
            
            pred = output.argmax(dim=1).cpu().numpy() # (B, H, W)
            label = label.numpy()
            
            label_flat = label.flatten()
            pred_flat = pred.flatten()
            
            # Remove ignore_index
            mask = (label_flat != IGNORE_INDEX)
            hist += fast_hist(label_flat[mask], pred_flat[mask], num_classes)
            
            # Visualization
            if save_dir and i < save_num:
                # Save comparison
                img_vis = denormalize(image[0])
                gt_vis = decode_segmap(label[0])
                pred_vis = decode_segmap(pred[0])
                
                # Combine: Input | GT | Pred
                # img_vis is float 0..1, others are uint8 0..255
                img_vis = (img_vis * 255).astype(np.uint8)
                
                combined = np.hstack([img_vis, gt_vis, pred_vis])
                Image.fromarray(combined).save(os.path.join(save_dir, f"val_{i}.png"))
            
            # Break if max_samples reached
            if max_samples and (i + 1) >= max_samples:
                break
            
    # Metrics
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    
    return {
        "Pixel Acc": acc,
        "Mean Acc": acc_cls,
        "mIoU": mean_iu,
        "Class IoU": iu
    }

def evaluate(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Model: {args.model} on Device: {device}")
    
    # Load Model
    model = get_model(args.model, num_classes=NUM_CLASSES).to(device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("WARNING: No checkpoint loaded! Evaluating random weights.")
    
    # Validation Dataset
    val_set = CityscapesDataset(root=DATASET_ROOT, split='val', mode='fine', transform=None)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    save_dir = os.path.join("results", args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Call validate with all arguments
    metrics = validate(model, val_loader, device, NUM_CLASSES, 
                      max_samples=args.max_samples,
                      save_dir=save_dir,
                      save_num=args.save_num)
    
    print("\n" + "="*40)
    print(f"Evaluation Results [{args.model}]")
    print("="*40)
    print(f"Pixel Accuracy:     {metrics['Pixel Acc']:.4f}")
    print(f"Mean Per-Class Acc: {metrics['Mean Acc']:.4f}")
    print(f"Mean IoU:           {metrics['mIoU']:.4f}")
    print("-" * 40)
    print("Per-Class IoU:")
    for name, score in zip(CLASSES, metrics['Class IoU']):
        print(f"{name:15s}: {score:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["unet", "deeplab", "segformer"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--save_num", type=int, default=5, help="Number of qualitative images to save")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--max_samples", type=int, default=None, help="Stop after N samples (for debugging)")
    args = parser.parse_args()
    
    evaluate(args)
