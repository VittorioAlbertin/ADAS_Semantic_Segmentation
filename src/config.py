
DATASET_ROOT = r"C:\Users\v01al\Documents\GitHub\ADAS_Semantic_Segmentation\datasets\cityscapes"

# Model Parameters
NUM_CLASSES = 19
IGNORE_INDEX = 255

# Data resolution
CROP_SIZE = (512, 1024)  # (H, W)
FULL_SIZE = (1024, 2048) # (H, W)

# Training Constraints
BATCH_SIZE = 1 # Hardware constraint
DEVICE = "cuda"

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
