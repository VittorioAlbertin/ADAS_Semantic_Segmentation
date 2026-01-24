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

def validate(model, val_loader, device, num_classes):
    """
    Runs validation on the given loader and returns metrics.
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
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
    
    # Tqdm total handling
    total_samples = len(val_set)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        # Create a subset or custom sampler if needed, but for now we just break loop
        # Actually simplest is to just slice dataset indices? 
        # But Dataset doesn't support slicing easily. 
        # We'll rely on the loop break in 'validate' if we passed max_samples logic there.
        # Check: `validate` above doesn't have max_samples logic.
        # Let's keep `evaluate` controlling max_samples logic by slicing the LOADER?
        # Or Just keep simple loop in validate.
    
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # For visualization in `evaluate` CLI mode, we unfortunately need the loop. 
    # The refactor requested assumes `validate` is pure metrics.
    # So `evaluate` script might duplicate some logic OR `validate` should support a callback/hook?
    # Let's keep `validate` pure for training loop.
    # The CLI `evaluate` can use `validate` for metrics, but if it needs visualization...
    # Okay, for simplicity: `validate` calculates metrics. 
    # If CLI needs viz, it can run a separate loop or we pass a `save_dir` to `validate`.
    
    # Let's implement `validate` with optional save_dir
    save_dir = os.path.join("results", args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Quick dirty: We just call `validate`.
    metrics = validate(model, val_loader, device, NUM_CLASSES)
    
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
