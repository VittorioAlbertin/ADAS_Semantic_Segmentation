# Cityscapes Semantic Segmentation

## Project Overview
This project implements a **comparative study** of three **semantic segmentation** architectures on the **Cityscapes** dataset: **U-Net**, **DeepLabV3+**, and **SegFormer**. 

**Focus**: Methodological rigor under strict hardware constraints (Single 8GB GPU).

> [!NOTE]
> For a detailed analysis of architectures, training strategies, and experimental results, please read the **[Project Report](docs/project_report.md)**.

## System Requirements
*   **OS**: Windows 11 / Linux
*   **GPU**: 8GB VRAM (min)
*   **Python**: 3.10+
*   **CUDA**: 12.x / 13.x

## Installation
1.  **Clone & Environment**:
    ```powershell
    git clone https://github.com/VittorioAlbertin/ADAS_Semantic_Segmentation.git
    cd ADAS_Semantic_Segmentation
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies**:
    ```powershell
    # Install PyTorch (Adjust for your CUDA version)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    pip install -r requirements.txt
    ```

## Dataset
Ensure **Cityscapes** is at `datasets/cityscapes/` with `gtFine` and `leftImg8bit`.

## Usage

### Training
**Arguments**:
*   `--model`: Model architecture [`unet`, `deeplab`, `segformer`].
*   `--epochs`: Number of training epochs (Default: 10).
*   `--lr`: Learning rate (Default: 1e-4).
*   `--val_interval`: Validation frequency in epochs (Default: 1).
*   `--resume`: Path to `.pth` checkpoint to resume from.
*   `--weighted_loss`: Use Class-Weighted Cross Entropy (Recommended for imbalance).
*   `--freeze_backbone`: Freezes encoder weights (Transfer Learning Phase 1).
*   `--full_scale`: Train on full 1024x2048 images (High VRAM usage).
*   `--experiment_name`: Custom name for results folder (Default: model name).

```powershell
# Train DeepLabV3+ (Frozen Backbone)
python -m src.train --model deeplab --epochs 10 --freeze_backbone --weighted_loss

# Resume Training (Unfrozen)
python -m src.train --model deeplab --epochs 20 --resume checkpoints/deeplab_latest.pth --weighted_loss
```
See `src/train.py --help` for full arguments including `--full_scale`.

### Evaluation
**Arguments**:
*   `--model`: Model architecture to evaluate.
*   `--checkpoint`: Path to specific `.pth` checkpoint.
*   `--save_num`: Number of qualitative result images to save (Default: 5).
*   `--device`: Force device usage [`cpu`, `cuda`] (Default: Auto).
*   `--max_samples`: Limit number of validation samples (for debugging).

```powershell
python -m src.evaluate --model deeplab --checkpoint checkpoints/deeplab_best.pth --save_num 5
```
