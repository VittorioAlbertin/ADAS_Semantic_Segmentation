import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CityscapesDataset
from src.config import DATASET_ROOT, NUM_CLASSES, IGNORE_INDEX

def calculate_weights():
    print("Initializing Dataset...")
    dataset = CityscapesDataset(root=DATASET_ROOT, split='train', mode='fine', transform=None, crop=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Pixel counts per class
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    print("Computing class distribution...")
    for _, label in tqdm(loader):
        label = label.numpy().flatten()
        # Remove ignore index
        label = label[label != IGNORE_INDEX]
        
        # Bincount
        # Ensure we only count valid classes
        lbl_counts = np.bincount(label, minlength=NUM_CLASSES)
        if len(lbl_counts) > NUM_CLASSES:
             # Just in case something weird happens, truncate
             lbl_counts = lbl_counts[:NUM_CLASSES]
        
        counts += lbl_counts
    
    # Calculate Weights (ENet Scheme)
    # Class probability = count / total
    # Weight = 1 / ln(1.02 + prob)
    
    total_pixels = counts.sum()
    probs = counts / total_pixels
    
    # ENet weights
    weights = 1.0 / np.log(1.02 + probs)
    
    print("\n" + "="*40)
    print("Class Weights (Copy to src/config.py)")
    print("="*40)
    print("CLASS_WEIGHTS = [")
    for w in weights:
        print(f"    {w:.4f},")
    print("]")
    print("="*40)
    
    # Also print counts for debug
    print("Class Counts:")
    for i, c in enumerate(counts):
        print(f"Class {i}: {c} pixels ({probs[i]*100:.2f}%)")

if __name__ == "__main__":
    calculate_weights()
