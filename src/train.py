import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from src.dataset import CityscapesDataset
from src.config import DATASET_ROOT, CROP_SIZE, FULL_SIZE, BATCH_SIZE, NUM_CLASSES, IGNORE_INDEX, DEVICE
from src.models import UNet, DeepLabV3Plus, SegFormer, get_model
from src.evaluate import validate

def train(args):
    # Setup Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    print("Initializing Datasets...")
    # Reduce workers to save RAM (Host Allocation Error fix)
    num_workers = 2 
    train_set = CityscapesDataset(root=DATASET_ROOT, split='train', mode='fine', transform=None)
    val_set = CityscapesDataset(root=DATASET_ROOT, split='val', mode='fine', transform=None)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True) # Val always BS=1

    # Model, Optimizer, Loss
    print(f"Initializing Model: {args.model}")
    model = get_model(args.model, num_classes=NUM_CLASSES).to(device)
    
    # Disable cuDNN benchmark to save VRAM/RAM stability
    torch.backends.cudnn.benchmark = False
    
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW is standard now
    
    # Load Checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            # Handle if file is full checkpoint or just state_dict
            # We saved just state_dict in previous version.
            # To resume epoch, we need to have saved it. 
            # Since we didn't, we just load weights.
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint '{args.resume}'")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # Training Loop
    best_iou = 0.0
    ACCUM_STEPS = 4  # Simulate Batch Size = 4 * 1 = 4
    
    print("Starting Training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device) # Shape [B, H, W]
            
            # Forward (AMP)
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images) # [B, 19, H, W]
                loss = criterion(outputs, masks)
                loss = loss / ACCUM_STEPS # Normalize loss for gradient accumulation
            
            # Backward
            scaler.scale(loss).backward()
            
            # Step
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            loss_val = loss.item() * ACCUM_STEPS # Back to real loss for logging
            epoch_loss += loss_val
            progress_bar.set_postfix({'loss': loss_val})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            print("Running Validation...")
            torch.cuda.empty_cache() # Clear VRAM
            metrics = validate(model, val_loader, device, NUM_CLASSES)
            print(f"Val mIoU: {metrics['mIoU']:.4f} | Pixel Acc: {metrics['Pixel Acc']:.4f}")
            
            # Save Latest
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("checkpoints", f"{args.model}_latest.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save Best
            if metrics['mIoU'] > best_iou:
                best_iou = metrics['mIoU']
                best_path = os.path.join("checkpoints", f"{args.model}_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"New Best mIoU! Saved to {best_path}")
            
            model.train() # Switch back to train mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "deeplab", "segformer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args)
