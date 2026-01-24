import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import MEAN, STD, IGNORE_INDEX

def get_cityscapes_colormap():
    """
    Returns the Cityscapes colormap for the 19 training classes.
    """
    # 19 colors for the 19 classes + 1 color for ignore_index (black)
    colors = [
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [70, 70, 70],    # building
        [102, 102, 156], # wall
        [190, 153, 153], # fence
        [153, 153, 153], # pole
        [250, 170, 30],  # traffic light
        [220, 220, 0],   # traffic sign
        [107, 142, 35],  # vegetation
        [152, 251, 152], # terrain
        [70, 130, 180],  # sky
        [220, 20, 60],   # person
        [255, 0, 0],     # rider
        [0, 0, 142],     # car
        [0, 0, 70],      # truck
        [0, 60, 100],    # bus
        [0, 80, 100],    # train
        [0, 0, 230],     # motorcycle
        [119, 11, 32],   # bicycle
        [0, 0, 0],       # ignore
    ]
    return np.array(colors, dtype=np.uint8)

def decode_segmap(mask):
    """
    Decodes a label mask (H, W) into an RGB image (H, W, 3).
    """
    mask = mask.cpu().numpy().astype(np.uint8) if isinstance(mask, torch.Tensor) else mask.astype(np.uint8)
    colormap = get_cityscapes_colormap()
    
    # Map ignore index (255) to 19 (the black color at the end of colormap)
    mask[mask == IGNORE_INDEX] = 19
    
    rgb = colormap[mask]
    return rgb

def denormalize(tensor):
    """
    Denormalizes an image tensor (C, H, W) for visualization.
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    
    # Remove batch dim if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Reverse normalization: x * std + mean
    tensor = tensor.cpu() * std + mean
    return tensor.permute(1, 2, 0).clamp(0, 1).numpy()

def visualize_batch(images, masks, output_path="sanity_check_data.png"):
    """
    Visualizes a batch of images and their corresponding masks.
    """
    batch_size = images.shape[0]
    fig, axs = plt.subplots(batch_size, 2, figsize=(15, 5 * batch_size))
    
    if batch_size == 1:
        axs = [axs]
        
    for i in range(batch_size):
        img_vis = denormalize(images[i])
        mask_vis = decode_segmap(masks[i])
        
        axs[i][0].imshow(img_vis)
        axs[i][0].set_title("Input Image")
        axs[i][0].axis("off")
        
        axs[i][1].imshow(mask_vis)
        axs[i][1].set_title("Ground Truth")
        axs[i][1].axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")
