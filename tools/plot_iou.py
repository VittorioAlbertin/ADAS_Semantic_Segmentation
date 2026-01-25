import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_WEIGHTS, CLASS_COUNTS

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def plot_iou_from_csv():
    csv_path = r"results/unet/evaluation_metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read CSV
    iou_dict = {}
    print(f"Reading {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        reading_classes = False
        for row in reader:
            if not row: continue
            if row[0] == 'Class':
                reading_classes = True
                continue
            
            if reading_classes:
                iou_dict[row[0]] = float(row[1])
    
    # Map counts to classes
    # CLASS_COUNTS is parallel to CLASSES list in config/dataset
    # Let's verify mapping
    scores = []
    counts = []
    names = []
    
    for i, name in enumerate(CLASSES):
        if name in iou_dict:
            names.append(name)
            scores.append(iou_dict[name])
            counts.append(CLASS_COUNTS[i])
        else:
            print(f"Warning: Class {name} not found in CSV.")
            
    # Sort by Count (Descending)
    data = list(zip(counts, names, scores))
    data.sort(key=lambda x: x[0], reverse=True)
    
    sorted_counts, sorted_names, sorted_scores = zip(*data)
    
    # Plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(sorted_names))
    plt.bar(x, sorted_scores, color='skyblue', edgecolor='navy')
    plt.xticks(x, sorted_names, rotation=45, ha='right')
    plt.ylabel('IoU Score')
    plt.title('Per-Class IoU (Sorted by Dataset Frequency)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(sorted_scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
    plt.tight_layout()
    plot_path = r"results/unet/iou_per_class_sorted.png"
    plt.savefig(plot_path)
    print(f"Sorted graph saved to {plot_path}")

if __name__ == "__main__":
    plot_iou_from_csv()
