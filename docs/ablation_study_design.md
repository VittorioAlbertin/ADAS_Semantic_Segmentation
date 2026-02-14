# Scientific Ablation Study Design

## Objective
To rigorously evaluate the impact of **Loss Functions**, **Transfer Learning Strategies**, and **Input Resolution** on semantic segmentation performance across three architectures: U-Net, DeepLabV3+, and SegFormer.

## Variables
1.  **Model**: U-Net (Custom), DeepLabV3+ (ResNet-50), SegFormer (MiT-B0).
2.  **Loss Function**:
    *   `Unweighted`: Standard Cross-Entropy.
    *   `Weighted`: Class-Weighted Cross-Entropy (Inverse Log Frequency).
    *   `Mixed`: Weighted (Epochs 0-10) -> Unweighted (Epochs 10-20).
3.  **Backbone Strategy**:
    *   `Scratch`: Random initialization (U-Net).
    *   `Frozen`: Pretrained backbone frozen, only head trained.
    *   `Unfrozen`: Full model fine-tuning.
    *   `Two-Stage`: Frozen (Epochs 0-10) -> Unfrozen (Epochs 10-20).
4.  **Resolution**:
    *   `Cropped`: Random 512x1024 crops (Standard).
    *   `Full`: Full 1024x2048 input (New).

---

## Study 1: Loss Function Efficacy (Focus: U-Net)
**Hypothesis**: Class/Frequency imbalance requires weighted loss for convergence, but unweighted loss yields better final precision once capabilities are established.

| Experiment ID | Configuration | Command | Status |
| :--- | :--- | :--- | :--- |
| **UNET-A** (Baseline) | Unweighted (10 epochs) | `python -m src.train --model unet --epochs 10` | *Done (10 ep)* - `checkpoints/unet_tenth_unweighted_best.pth`. Deemed sufficient. |
| **UNET-B** | Weighted (20 epochs) | `python -m src.train --model unet --epochs 20 --weighted_loss` | **TODO** |
| **UNET-C** (Mixed) | Weighted (10) -> Resume Unweighted (10) | *Run UNET-B (10 ep)* -> `python -m src.train --model unet --epochs 10 --resume ...` | *Done* - `checkpoints/unet_best.pth` |

---

## Study 2: Transfer Learning Strategy (Focus: DeepLabV3+)
**Hypothesis**: Freezing the backbone is critical for distinct encoder-decoder architectures to prevent catastrophic forgetting of ImageNet features during early training.

| Experiment ID | Configuration | Command | Status |
| :--- | :--- | :--- | :--- |
| **DL-A** (Naive) | Unfrozen from start (20 epochs, Weighted) | `python -m src.train --model deeplab --epochs 20 --weighted_loss` | **TODO** |
| **DL-B** (Frozen) | Frozen only (20 epochs, Weighted) | `python -m src.train --model deeplab --epochs 20 --weighted_loss --freeze_backbone` | *Done (10 ep)* - `checkpoints/deeplab_tenth_weighted_frozen.pth` |
| **DL-C** (Two-Stage) | Frozen (10) -> Unfrozen (10) (Weighted) | *Run DL-B (10 ep)* -> `python -m src.train --model deeplab --epochs 10 --weighted_loss --resume ...` | *Done* - `checkpoints/deeplab_best.pth` |

---

## Study 3: Resolution Impact (Focus: SegFormer & DeepLab)
**Hypothesis**: Global context from full-scale images significantly improves mIoU for Transformer-based architectures, more so than CNNs.

*Note: Models trained with best known loss/backbone strategy from previous studies.*

| Experiment ID | Configuration | Command | Status |
| :--- | :--- | :--- | :--- |
| **SF-A** (Baseline) | Cropped, Two-Stage Weighted | *Standard SegFormer training flow* | *Done* - `checkpoints/segformer_10plus10.pth` |
| **SF-B** (Full) | Full Scale, Two-Stage Weighted | `python -m src.train --model segformer --full_scale --epochs 10 --freeze_backbone --weighted_loss` -> Resume Unfrozen (10 ep) | **WIP** |
| **DL-D** (Full) | Full Scale, Two-Stage Weighted | `python -m src.train --model deeplab --full_scale --epochs 10 --freeze_backbone --weighted_loss` -> Resume Unfrozen (10 ep) | **TODO** |

---

## Execution scripts

To ensure reproducibility, use the generated automation scripts (to be created) or run the commands sequentially.
