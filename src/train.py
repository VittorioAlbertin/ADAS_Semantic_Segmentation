import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import csv
import math
import matplotlib.pyplot as plt

from src.dataset import CityscapesDataset
from src.config import DATASET_ROOT, CROP_SIZE, FULL_SIZE, BATCH_SIZE, NUM_CLASSES, IGNORE_INDEX, DEVICE, CLASS_WEIGHTS
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
    
    if args.weighted_loss:
        print("Using Weighted Cross Entropy Loss")
        weights = torch.tensor(CLASS_WEIGHTS).to(device).float()
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    else:
        print("Using Standard Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW is standard now
    
    # Freeze Backbone logic
    if args.freeze_backbone:
        print("INFO: Freezing Backbone parameters...")
        if args.model == "deeplab":
            # Freeze ResNet backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif args.model == "segformer":
            # Freeze MiT encoder. 
            # SegFormer structure: model.model -> SegFormerForSemanticSegmentation -> .segformer (encoder)
            for param in model.model.segformer.parameters():
                param.requires_grad = False
        elif args.model == "unet":
            print("WARNING: U-Net is trained from scratch. --freeze_backbone has no pre-trained backbone to freeze. Ignoring.")
            
    # Count Trainable Params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")
    
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
            
            save_dir = os.path.join("results", args.model)
            os.makedirs(save_dir, exist_ok=True)
            
            metrics = validate(model, val_loader, device, NUM_CLASSES, save_dir=save_dir, save_num=5)
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
            
            # --- Logging & Plotting ---
            log_path = os.path.join("results", args.model, "log.csv")
            file_exists = os.path.isfile(log_path)
            
            # Key classes to track: Road (Common), Wall (Rare), Traffic Sign (Safety/Rare), Car (Common)
            # Indices: Road=0, Wall=3, Traffic Sign=7, Car=13
            iou_road = metrics['Class IoU'][0]
            iou_wall = metrics['Class IoU'][3]
            iou_sign = metrics['Class IoU'][7]
            iou_car = metrics['Class IoU'][13]

            # Write key metrics to CSV
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['epoch', 'train_loss', 'val_miou', 'val_pixel_acc', 'val_mean_acc', 'iou_road', 'iou_wall', 'iou_sign', 'iou_car'])
                writer.writerow([epoch+1, avg_loss, metrics['mIoU'], metrics['Pixel Acc'], metrics['Mean Acc'], iou_road, iou_wall, iou_sign, iou_car])
            
            # Update Plot
            try:
                # Read back log file to plot full history
                epochs, losses, mious = [], [], []
                road_ious, wall_ious = [], []
                
                def safe_float(v):
                    if v is None or v == "": return float('nan')
                    try: return float(v)
                    except ValueError: return float('nan')

                with open(log_path, mode='r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'epoch' not in row or not row['epoch']: continue
                        epochs.append(int(row['epoch']))
                        losses.append(safe_float(row.get('train_loss')))
                        mious.append(safe_float(row.get('val_miou')))
                        
                        # Handle missing columns from older runs
                        if 'iou_road' in row:
                            road_ious.append(safe_float(row['iou_road']))
                            wall_ious.append(safe_float(row.get('iou_wall')))
                        else:
                            road_ious.append(float('nan'))
                            wall_ious.append(float('nan'))
                
                plt.figure(figsize=(12, 5))
                
                # Loss Plot
                plt.subplot(1, 2, 1)
                plt.plot(epochs, losses, label='Train Loss', color='red', marker='o')
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.grid(True)
                
                # Metrics Plot (mIoU vs Specific Classes)
                plt.subplot(1, 2, 2)
                plt.plot(epochs, mious, label='mIoU (Avg)', color='black', linewidth=2, linestyle='--')
                
                # Only plot class indices if we have valid data
                valid_road = [x for x in road_ious if not math.isnan(x)]
                if valid_road:
                     plt.plot(epochs, road_ious, label='Road (Common)', color='green', marker='.')
                     plt.plot(epochs, wall_ious, label='Wall (Rare)', color='orange', marker='.')
                
                plt.title('Class-Specific IoU Trends')
                plt.xlabel('Epoch')
                plt.ylabel('IoU')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join("results", args.model, "training_plot.png"))
                plt.close()
                print(f"Updated training plot at results/{args.model}/training_plot.png")
            except Exception as e:
                print(f"Warning: Failed to plot metrics: {e}")
            
            model.train() # Switch back to train mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "deeplab", "segformer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--weighted_loss", action="store_true", help="Use class weighted loss")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the backbone/encoder parameters")
    args = parser.parse_args()
    
    train(args)
