import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from src.dataset import CityscapesDataset
from src.config import DATASET_ROOT, BATCH_SIZE
from src.utils import visualize_batch

def main():
    print("Initializing Cityscapes Dataset...")
    train_dataset = CityscapesDataset(root=DATASET_ROOT, split='train', mode='fine')
    val_dataset = CityscapesDataset(root=DATASET_ROOT, split='val', mode='fine')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\nFetching one batch from Training set...")
    images, masks = next(iter(train_loader))
    print(f"Image Batch Shape: {images.shape}")
    print(f"Mask Batch Shape: {masks.shape}")
    
    print("\nVisualizing batch...")
    visualize_batch(images, masks, output_path="sanity_check_train.png")
    
    print("\nFetching one batch from Validation set...")
    images_val, masks_val = next(iter(val_loader))
    print(f"Val Image Batch Shape: {images_val.shape}")
    print(f"Val Mask Batch Shape: {masks_val.shape}")
    
    # Optional: verify ignore index
    unique_labels = masks.unique()
    print(f"\nUnique labels in batch: {unique_labels}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
