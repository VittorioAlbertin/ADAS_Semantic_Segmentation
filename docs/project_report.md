# Comparative Study of Semantic Segmentation Architectures for ADAS

**Author**: [Your Name]
**Date**: January 2026
**Project**: ADAS Semantic Segmentation on Cityscapes

---

## Abstract

This project presents a comparative analysis of three distinct semantic segmentation architectures, **U-Net**, **DeepLabV3+**, and **SegFormer**, applied to the **Cityscapes** dataset for Autonomous Driving Assistance Systems (ADAS). The study focuses on implementing and optimizing these models under strict hardware constraints (Single 8GB GPU). By employing techniques such as Automatic Mixed Precision (AMP), Gradient Accumulation, and Class-Weighted Loss functions, we demonstrate effective training strategies for high-resolution segmentation. We evaluate the models based on Mean Intersection over Union (mIoU), considering both global performance and class-specific accuracy to address the severe class imbalance inherent in automotive datasets.

---

## 1. Introduction

Semantic Segmentation is a fundamental task in computer vision where every pixel in an image is classified into a predefined category. In the context of **Autonomous Driving**, this translates to distinguishing drivable surfaces (Road) from obstacles (Cars, Pedestrians) and infrastructure (Traffic Signs, Poles).

This project utilizes the **Cityscapes** dataset, a benchmark suite containing high-quality pixel-level annotations of urban street scenes. The dataset features 19 semantic classes and high-resolution images ($1024 \times 2048$), presenting significant challenges in terms of computational resources and model complexity.

---

## 2. Motivation

The primary motivations for this study are threefold:

1.  **Architectural Evolution**: To compare the performance of a classic Fully Convolutional Network (U-Net), a state-of-the-art CNN with dilated convolutions (DeepLabV3+), and a modern Transformer-based architecture (SegFormer).
2.  **Resource-Constrained Engineering**: To demonstrate that modern deep learning research usually performed on massive clusters can be scaled down to consumer hardware (8GB VRAM) without compromising methodological rigor.
3.  **Class Imbalance**: To address the "long-tail" distribution problem in urban scenes, where safety-critical classes like *Traffic Lights* and *Pedestrians* are significantly rarer than *Road* or *Building*.

---

## 3. Methodology: Model Architectures

We implemented three models from scratch or using fine-tuning frameworks, each representing a different paradigm in segmentation history.

### 3.1 U-Net (The Baseline)
*   **Paradigm**: Encoder-Decoder with Skip Connections.
*   **Implementation**: A custom 5-level U-Net built from scratch in PyTorch.
*   **Key Modification**: We replaced standard Batch Normalization with **Instance Normalization**.
    *   *Reasoning*: Due to memory constraints, we are forced to train with a physical batch size of 1 per GPU. at Batch Size=1, Batch Normalization statistics are too noisy to yield stable convergence. Instance Normalization is batch-agnostic.

### 3.2 DeepLabV3+ (The Context Specialist)
*   **Paradigm**: CNN with Atrous Spatial Pyramid Pooling (ASPP).
*   **Backbone**: **ResNet-50** (Pretrained on ImageNet).
*   **Mechanism**:
    *   **Dilated (Atrous) Convolutions**: allow the network to expand its receptive field without downsampling resolution, preserving spatial detail.
    *   **ASPP**: Captures context at multiple scales (rates 6, 12, 18), allowing the model to understand large objects (Bus) and small details (Traffic Light) simultaneously.
*   **Decoder**: The "Plus" decoder fuses low-level features from the backbone to refine object boundaries.

### 3.3 SegFormer (The Transformer)
*   **Paradigm**: Hierarchical Transformer (Encoder) + MLP (Decoder).
*   **Backbone**: **MiT-B0** (MixTransformer, Lightest Variant).
*   **Mechanism**:
    *   Unlike CNNs where the receptive field grows slowly, Transformers utilize **Self-Attention** to have a global receptive field from the very first layer.
    *   It eschews complex Positional Encodings for 3x3 Conv "Overlapped Patch Merging".
*   **Efficiency**: The MLP decoder aggregates features from all 4 scales ($1/4, 1/8, 1/16, 1/32$) and fuses them, proving highly efficient for the 8GB VRAM budget.

---

## 4. Training Process

### 4.1 Data Pipeline & Constraints
Training on $1024 \times 2048$ images on a single 8GB GPU is non-trivial. We adopted the following strategy:
*   **Training Resolution**: Random Crops of **$512 \times 1024$**.
*   **Validation Resolution**: Full **$1024 \times 2048$** (via sliding window or mapped inference).
*   **Automatic Mixed Precision (AMP)**: We use `float16` for activations to reduce VRAM usage by ~40%.
*   **Gradient Accumulation**: To stabilize training with Batch Size 1, we accumulate gradients over 4 steps before updating weights, simulating an **Effective Batch Size of 4**.

### 4.2 Loss Function Evolution
We evolved the loss function to improve performace:
1.  **Standard Cross Entropy**:
    *   *Result*: The model optimized for the most frequent classes (Road, Building). Rare classes (Wall, Motorcycle) had near-zero IoU.
2.  **Weighted Cross Entropy**:
    *   We calculated the class distribution of the training set.
    *   We applied **Inverse Log Frequency weights** (ENet scheme): $W_c = \frac{1}{\ln(1.02 + f_c)}$.
    *   *Result*: The penalty for misclassifying rare classes is increased (e.g., *Motorcycle* has a weight of ~47 vs. *Road* ~2.7), forcing the model to learn features for scarce objects.

---

## 5. Results

### 5.1 Metrics Table
| Model | Global Pixel Acc | Mean Class Acc | mIoU |
| :--- | :--- | :--- | :--- |
| U-Net (Baseline) | 0.8998 | 0.5491 | 0.4612 |
| DeepLabV3+ | **0.9263** | **0.7224** | **0.6324** |
| SegFormer | 0.9075 | 0.6457 | 0.5389 |

### 5.2 Per-Class IoU Analysis

#### U-Net
![U-Net IoU Per Class](../results/unet/iou_per_class.png)

#### DeepLabV3+
![DeepLabV3+ IoU Per Class](../results/deeplab/iou_per_class.png)

#### SegFormer
![SegFormer IoU Per Class](../results/segformer/iou_per_class.png)

**Analysis**:
*   **Strengths**: All models perform well on frequent classes like *Road*, *Building*, and *Vegetation*. DeepLabV3+ shows superior performance on *Car* (0.8758) and *Sky* (0.9277).
*   **Weaknesses**: U-Net struggles significantly with rare classes, achieving near-zero IoU for *Wall*, *Train*, and *Motorcycle*. DeepLabV3+ and SegFormer improved on these, but *Train* and *Truck* remain challenging.
*   **Impact of Weights**:
    *   **U-Net**: Unweighted training collapsed on rare classes. Introducing weighted loss improved mIoU from 0.3273 to 0.4010, recovering some performance on *Pole* and *Traffic Sign*.
    *   **SegFormer**: While it showed good intermediate performance (mIoU ~0.57), the final unweighted fine-tuning stage increased Global Pixel Accuracy but degraded Mean Class Accuracy (0.7691 -> 0.6457) and mIoU, indicating it started to overfit to the majority classes/texture at the expense of semantic structure for rare objects.
    *   **DeepLabV3+**: Proved the most robust, effectively leveraging the weighted loss to maintain high accuracy across the board.

### 5.3 Qualitative Comparison
We compare the qualitative outputs on a validation sample (Validation ID: 10).

#### U-Net
![U-Net Result](../results/unet/val_10.png)

#### DeepLabV3+
![DeepLabV3+ Result](../results/deeplab/val_10.png)

#### SegFormer
![SegFormer Result](../results/segformer/val_10.png)

---

## 6. Conclusions

This study successfully implemented a robust pipeline for semantic segmentation under constrained resources.
*   **Architectural Findings**: **DeepLabV3+ > SegFormer > U-Net**. DeepLabV3+ proved to be the most effective architecture for this dataset and constraint profile, offering the best trade-off between spatial detail and global context. SegFormer, while promising and efficient, showed instability during the final fine-tuning phase. U-Net served as a solid baseline but lacked the capacity to handle the complex, multi-scale nature of Cityscapes effectively without more advanced modules.
*   **Engineering**: Usage of InstanceNorm and Gradient Accumulation was critical for convergence at low batch sizes.
*   **Balancing**: Weighted Loss is essential for Cityscapes to prevent the model from ignoring safety-critical but rare classes. The extraction of "Context" via ASPP (DeepLab) or Self-Attention (SegFormer) proved more valuable than simple skip connections (U-Net) for correctly classifying difficult regions.

---
