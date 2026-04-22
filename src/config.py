import os
import torch

DATASET_ROOT = r"C:\Users\v01al\Documents\GitHub\ADAS_Semantic_Segmentation\datasets\cityscapes"

# Model Parameters
NUM_CLASSES = 19
IGNORE_INDEX = 255

# Data resolution
CROP_SIZE = (512, 1024)  # (H, W)
FULL_SIZE = (1024, 2048) # (H, W)

# Training Constraints
BATCH_SIZE = 1 # Hardware constraint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class Weights (Inverse Log Frequency - ENet Scheme)
# Calculated on Cityscapes "fine" training set
CLASS_WEIGHTS = [
    2.7122,  # road
    13.2552, # sidewalk
    5.5872,  # building
    37.3778, # wall
    34.6692, # fence
    30.6163, # pole
    46.5850, # traffic light
    40.4255, # traffic sign
    6.9759,  # vegetation
    30.8575, # terrain
    24.2824, # sky
    25.8544, # person
    44.9643, # rider
    9.1379,  # car
    43.5033, # truck
    43.7164, # bus
    43.9003, # train
    47.2353, # motorcycle
    39.0508, # bicycle
]

# Class Counts (Calculated from training set)
# Used for sorting visualization
CLASS_COUNTS = [
    625527504, # road
    85722638,  # sidewalk
    258515533, # building
    10450701,  # wall
    13607425,  # fence
    19391126,  # pole
    2494369,   # traffic light
    7410874,   # traffic sign
    197022220, # vegetation
    19003732,  # terrain
    32376137,  # sky
    28548212,  # person
    3655973,   # rider
    140490917, # car
    4778142,   # truck
    4609747,   # bus
    4465719,   # train
    2050967,   # motorcycle
    8722501,   # bicycle
]
# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Category definitions (Cityscapes standard)
CATEGORIES = [
    'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'
]

# Map logical class ID (0-18) to category ID (0-6)
CLASS_ID_TO_CATEGORY_ID = [
    0, 0,       # road, sidewalk
    1, 1, 1,    # building, wall, fence
    2, 2, 2,    # pole, traffic light, traffic sign
    3, 3,       # vegetation, terrain
    4,          # sky
    5, 5,       # person, rider
    6, 6, 6, 6, 6, 6 # car, truck, bus, train, motorcycle, bicycle
]
