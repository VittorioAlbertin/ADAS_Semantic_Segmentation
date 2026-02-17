import os
import sys
import csv
import math
import matplotlib.pyplot as plt

def plot_training_log():
    # Use unet/log.csv by default or arg
    log_path = r"results/.../log.csv"
    output_path = r"results/.../training_plot.png"
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    print(f"Reading {log_path}...")
    
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
    
    print(f"Plotting {len(epochs)} epochs...")
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Train Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
    plt.savefig(output_path)
    plt.close()
    print(f"Updated training plot at {output_path}")

if __name__ == "__main__":
    plot_training_log()
