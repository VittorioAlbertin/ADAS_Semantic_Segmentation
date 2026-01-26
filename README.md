# Cityscapes Semantic Segmentation

## Project Overview
This project implements a comparative study of three semantic segmentation architectures on the Cityscapes dataset. The focus is on methodological rigor under constrained resources (Single 8GB GPU). The project uses **PyTorch** and enforces **Automatic Mixed Precision (AMP)** and **Gradient Accumulation** to handle high-resolution data efficiently.

### Models Implemented
1.  **U-Net (From Scratch)**: Custom 5-level encoder-decoder with Instance Normalization.
2.  **DeepLabV3+ (Fine-Tuned)**: ResNet-50 backbone (ImageNet pretrained) with a custom DeepLabV3+ decoder head.
3.  **SegFormer (Fine-Tuned)**: Transformer-based architecture using the MixTransformer (MiT-B0) backbone.

## System Requirements
*   **OS**: Windows 11 (tested) / Linux compatible.
*   **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM) or equivalent.
*   **Python**: 3.10+
*   **CUDA**: 12.x / 13.x

## Installation
1.  **Clone the Repository**:
    ```powershell
    git clone [<repo_url>](https://github.com/VittorioAlbertin/ADAS_Semantic_Segmentation.git)
    cd ADAS_Semantic_Segmentation
    ```

2.  **Create Virtual Environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install PyTorch**:
    *   This project requires PyTorch 2.x with CUDA support.
    *   Command used:
    ```powershell
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```
    *(Adjust `--index-url` based on your specific CUDA version)*

4.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
    *Dependencies include: `numpy`, `matplotlib`, `pillow`, `opencv-python`, `tqdm`, `transformers`.*

## Dataset Setup
The project expects the **Cityscapes** dataset structure at `datasets/cityscapes/` (configured in `src/config.py`).
```text
datasets/
└── cityscapes/
    ├── gtFine/             # Ground Truth (train/val/test)
    │   └── train/
    │       └── aachen/
    │           └── *_gtFine_labelIds.png
    ├── leftImg8bit/        # Input Images (train/val/test)
    │   └── train/
    │       └── aachen/
    │           └── *_leftImg8bit.png
    ├── license.txt
    └── README
```
**Important**: The `gtFine` labels must include `labelIds.png`. The `trainId` mapping (19 classes) is handled internally by `src.dataset.CityscapesDataset`.

## Usage

### 1. Data Pipeline Verification
Before training, verify the dataset loading, cropping, and label mapping:
```powershell
python debug/check_data.py
```
*   **Output**: `debug/sanity_check_train.png` showing Input Image and Colorized Ground Truth.
*   **Checks**: Verifies 512x1024 training crops and correct class mapping.

### 2. Training
The main training script is `src/train.py`. It supports all three models.

**Arguments**:
*   `--model`: [`unet`, `deeplab`, `segformer`]
*   `--epochs`: Number of epochs (Default: 10)
*   `--lr`: Learning rate (Default: 1e-4)
*   `--val_interval`: Run validation every N epochs (Default: 1).
*   `--resume`: Path to a checkpoint `.pth` file to resume training from.
*   `--weighted_loss`: Use Class-Weighted Cross Entropy (Addresses imbalance).

**Examples**:
```powershell
# Train U-Net from scratch (Validate every epoch)
python -m src.train --model unet --epochs 50 --val_interval 1

# Resume training from latest checkpoint
python -m src.train --model unet --epochs 50 --resume checkpoints/unet_latest.pth

# Train SegFormer with custom LR
python -m src.train --model segformer --epochs 50 --lr 6e-5

# Train with Weighted Loss (Recommended)
python -m src.train --model unet --epochs 50 --weighted_loss
```

**Features**:
*   **Auto-Save Best**: The script automatically tracks mIoU and saves the best model to `checkpoints/<model>_best.pth`.
*   **Latest**: Always saves the current state to `checkpoints/<model>_latest.pth`.
*   **Logs**: Training loss and validation metrics (mIoU, Pixel Acc) printed to console.

### 3. Evaluation
Evaluate trained models on the **full-resolution (1024x2048)** validation set.

```powershell
# Standard Evaluation (Full validation set)
python -m src.evaluate --model unet --checkpoint checkpoints/unet_best.pth --save_num 5

# Quick Visualization (Test only 5 images and stop)
python -m src.evaluate --model unet --checkpoint checkpoints/unet_best.pth --max_samples 5 --save_num 5 --device cpu
```
*   **Metrics**: Calculates Global Pixel Accuracy, Mean Class Accuracy, and **mIoU**.
*   **Visualization**: Saves side-by-side comparisons (Image | GT | Pred) to `results/<model>/`.

## Technical Implementation Details
*   **Resolution Strategy**:
    *   **Training**: Random Crops of 512x1024 to fit into 8GB VRAM with `BatchSize=1`.
    *   **Inference**: Full 1024x2048 input.
*   **Mixed Precision**: Training uses `torch.amp.autocast` to reduce VRAM usage.
*   **Gradient Accumulation**: Accumulates gradients over 4 steps to simulate an effective batch size of 4.
*   **Normalization**:
    *   **U-Net**: Uses `InstanceNorm2d` layers to be agnostic to small batch sizes.
    *   **DeepLab/SegFormer**: Use standard ImageNet normalization.
