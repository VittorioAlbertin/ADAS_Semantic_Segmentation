# Comparative Study of Semantic Segmentation Architectures for ADAS

**Project**: ADAS Semantic Segmentation on Cityscapes
**Date**: January 2026

---

## Abstract

This project presents a comparative analysis of three distinct semantic segmentation architectures—**U-Net**, **DeepLabV3+**, and **SegFormer**—applied to the **Cityscapes** dataset for Autonomous Driving Assistance Systems (ADAS). The study demonstrates effective training strategies for high-resolution segmentation under strict hardware constraints (Single 8GB GPU). By employing techniques such as Automatic Mixed Precision (AMP), Gradient Accumulation, and Class-Weighted Loss functions, we evaluate the models based on Mean Intersection over Union (mIoU), addressing the severe class imbalance inherent in automotive datasets.

---

## 1. Introduction

Semantic Segmentation is a fundamental task in computer vision where every pixel in an image is classified into a predefined category. In **Autonomous Driving**, this distinguishes drivable surfaces (Road) from obstacles (Cars, Pedestrians) and infrastructure (Traffic Signs, Poles).

This project utilizes the **Cityscapes** dataset, a benchmark suite containing high-quality pixel-level annotations of urban street scenes. The dataset features 19 semantic classes and high-resolution images ($1024 \times 2048$), presenting significant challenges in terms of computational resources and model complexity.

---

## 2. Motivation

1.  **Architectural Evolution**: Comparing a classic Fully Convolutional Network (U-Net), a state-of-the-art CNN with dilated convolutions (DeepLabV3+), and a modern Transformer-based architecture (SegFormer).
2.  **Resource-Constrained Engineering**: Demonstrating that deep learning research usually performed on massive clusters can be scaled down to consumer hardware (8GB VRAM) without compromising methodological rigor.
3.  **Class Imbalance**: Addressing the "long-tail" distribution problem in urban scenes, where safety-critical classes like *Traffic Lights* and *Pedestrians* are significantly rarer than *Road* or *Building*.

---

## 3. Methodology: Model Architectures

### 3.1 U-Net (The Baseline)
*   **Paradigm**: Encoder-Decoder with Skip Connections.
*   **Implementation**: A custom 5-level U-Net built from scratch.
*   **Key Modification**: Replaced Batch Normalization with **Instance Normalization** to handle the physical batch size of 1.

### 3.2 DeepLabV3+ (The Context Specialist)
*   **Paradigm**: CNN with Atrous Spatial Pyramid Pooling (ASPP).
*   **Backbone**: **ResNet-50** (Pretrained on ImageNet).
*   **Mechanism**:
    *   **Dilated Convolutions**: Expand receptive field without downsampling.
    *   **ASPP**: Captures context at multiple scales (rates 6, 12, 18).
*   **Decoder**: Fuses low-level features from the backbone to refine object boundaries.

### 3.3 SegFormer (The Transformer)
*   **Paradigm**: Hierarchical Transformer (Encoder) + MLP (Decoder).
*   **Backbone**: **MiT-B0** (MixTransformer, Lightest Variant).
*   **Mechanism**: Uses **Self-Attention** for a global receptive field from the first layer and **Overlapped Patch Merging** instead of positional encodings.
*   **Efficiency**: The MLP decoder aggregates features from all 4 scales, proving highly efficient for the 8GB VRAM budget.

### 3.4 Architectural Adaptations for Resource Constraints

To adapt these foundational models to the physical limitations of consumer hardware (specifically a rigid batch_size=1 limitation for high-resolution tensors), several non-standard architectural modifications were necessary.

#### 3.4.1 U-Net Adaptation
**Why it couldn't work normally**: The original 2015 U-Net architecture relies on unpadded convolutions, which consistently shrink the spatial dimensions of feature maps at every layer, necessitating complex cropping mechanisms during the skip connections. Furthermore, utilizing standard Batch Normalization (which became standard practice post-2015) mathematically fails when constrained to a batch size of 1, as calculating variance across a single batch element yields zero, leading to unstable gradients.

**How we changed it**: The architecture was modernized by enforcing padding=1 across all convolutional layers to preserve spatial dimensions exactly. Crucially, standard Batch Normalization was entirely replaced with Instance Normalization (nn.InstanceNorm2d). Finally, the learnable transposed convolutions in the expanding path were replaced with deterministic bilinear upsampling.

**What is achieved**: Instance Normalization computes statistics across the spatial dimensions of each individual channel independently, granting the model completely stable training dynamics completely independent of the batch dimension. Padded convolutions allow the encoder feature maps to concatenate perfectly with the decoder without arbitrary cropping, and bilinear upsampling suppresses the grid-like "checkerboard artifacts" common in transposed convolutions, yielding smoother semantic boundaries.

#### 3.4.2 DeepLabV3+ Adaptation
**Why it couldn't work normally**: The DeepLabV3+ architecture relies on an Atrous Spatial Pyramid Pooling (ASPP) module to capture multi-scale context. A critical component of this ASPP is the Image Pooling branch, which performs global average pooling to encode global context. Under our strict hardware constraint of a physical batch size of 1, this global pooling reduces the spatial dimensions to $1 \times 1$. If a single-element batch of shape (1, C, 1, 1) is passed through standard Batch Normalization, the calculated variance is exactly zero, which immediately crashes the training process with a ZeroDivisionError. Additionally, the architecture requires modifying the standard ResNet backbone to utilize atrous convolutions to maintain a dense output stride of 16 or 8. However, manually rewriting custom dilated residual blocks (the "Multi-Grid" method) breaks compatibility with standard pre-trained weights. Finally, the paper advocates for replacing standard convolutions with depthwise separable convolutions within the ASPP and decoder to reduce parameters, which introduces unnecessary engineering complexity when standard operations are already highly optimized.

**How we changed it**: To prevent the global pooling crash, we explicitly isolated the Image Pooling branch within the ASPP module and replaced its standard BatchNorm2d layer with Group Normalization (nn.GroupNorm). Second, instead of manually writing custom dilated residual blocks to achieve the required output stride, we leveraged PyTorch's native replace_stride_with_dilation=[False, True, True] API. Finally, we bypassed depthwise separable convolutions, opting to use standard, highly optimized $3\times3$ convolutions within the ASPP and decoder refinement stages.

**What is achieved**: The GroupNorm adaptation elegantly bypasses the $1 \times 1$ pooling crash by calculating statistics across channel groups rather than the batch dimension, allowing the ASPP to capture global context stably even with a batch size of 1. Utilizing the native PyTorch dilation flag dynamically altered the spatial downsampling of the ResNet-50 architecture under the hood. This seamlessly achieved the required dense output stride while preserving 100% compatibility with ResNet50_Weights.IMAGENET1K_V1, drastically accelerating convergence. Ultimately, the model retains the full boundary-recovery precision of the V3+ decoder structure while remaining robust to our extreme hardware constraints.

#### 3.4.3 SegFormer Adaptation
**Why it couldn't work normally**: Standard Vision Transformers (ViTs) rely on fixed positional encodings. Because our training pipeline utilizes $512 \times 1024$ random crops but evaluates on full $1024 \times 2048$ images, rigid positional encodings would break during inference. Furthermore, standard ViTs produce a single-resolution feature map, which is insufficient for dense pixel-level prediction tasks, and pairing a Transformer with a heavy CNN context head (like ASPP) would easily exceed the 8GB VRAM limit.

**How we changed it**: We utilized the MiT-B0 (Mix Transformer) backbone. This specific architecture discards fixed positional encodings entirely, replacing them with a Mix-FFN (a $3 \times 3$ convolution inside the feed-forward network that implicitly learns spatial location via zero-padding). We also bypassed complex decoders by utilizing SegFormer's purely All-MLP decoder, which simply unifies the channel dimensions of the multi-scale features and bilinearly upsamples them.

**What is achieved**: Because Transformers natively utilize Layer Normalization (which normalizes across the channel dimension per pixel), this model is fundamentally immune to the batch_size=1 limitation right out of the box. The Mix-FFN enables the model to accept dynamic, resolution-independent inputs, seamlessly transitioning from cropped training to full-scale inference. Finally, because the Transformer's self-attention mechanism naturally provides a massive effective receptive field from the earliest layers, the ultra-lightweight All-MLP decoder achieves global context awareness without the immense computational overhead of an ASPP module.

---

## 4. Technical Implementation Details (Hardware Constraints)

Training on $1024 \times 2048$ images on a single 8GB GPU required specific optimizations:

1.  **Resolution Strategy**:
    *   **Training**: Random Crops of **$512 \times 1024$**.
    *   **Inference**: Full **$1024 \times 2048$** (Direct inference).
    *   *Note on Full Scale Training*: We implemented a mode to train on full $1024 \times 2048$ images. However, with an unfrozen backbone, training time escalated to **~6 hours per epoch**, making it infeasible for this comparative study. All results presented use the cropped training strategy for fairness and reproducibility.

2.  **Memory Optimization**:
    *   **Automatic Mixed Precision (AMP)**: Uses `float16` for activations to reduce VRAM usage by ~30-40%.
    *   **Gradient Accumulation**: Accumulates gradients over 4 steps to simulate an **Effective Batch Size of 4** (since Physical Batch Size is limited to 1).

3.  **Normalization**:
    *   **U-Net**: Uses `InstanceNorm2d`.
    *   **DeepLab/SegFormer**: Use standard ImageNet normalization statistics.

---

## 5. Training Process & Strategy Analysis

### 5.1 Loss Function Evolution
To address class imbalance, we evolved the loss function:
*   **Unweighted Cross Entropy**: Resulted in models predicting only frequent classes (Road, Building).
*   **Weighted Cross Entropy**: applied **Inverse Log Frequency weights** (ENet scheme). This forced the model to learn rare classes like *Pole* and *Traffic Light*.
*   **Mixed Strategy**: Weighted Loss (Epochs 0-10) to learn features $\rightarrow$ Unweighted Loss (Epochs 10-20) to refine precision.

### 5.2 Transfer Learning Strategy
*   **DeepLabV3+**: Required freezing the ResNet-50 backbone for the first 10 epochs. Unfreezing immediately caused "catastrophic forgetting" of ImageNet features due to unstable gradients.
*   **SegFormer**: Similarly benefited from a frozen stage. However, the final unweighted fine-tuning stage (Phase 3) caused performance degradation, suggesting Transformers are more sensitive to class imbalance than CNNs.

---

## 6. Evaluation & Results

### 6.1 Metrics Table
| Model | Global Pixel Acc | Mean Class Acc | mIoU |
| :--- | :--- | :--- | :--- |
| U-Net (Baseline) | 0.8998 | 0.5491 | 0.4612 |
| DeepLabV3+ | **0.9263** | **0.7224** | **0.6324** |
| SegFormer | 0.9075 | 0.6457 | 0.5389 |

### 6.2 Analysis
*   **DeepLabV3+**: The superior performer (mIoU 0.63). The ASPP module and deep ResNet backbone effectively capture both local texture and global context. It is robust to the unweighted fine-tuning stage.
*   **SegFormer**: Strong performance (mIoU 0.54) given its lightweight backbone (MiT-B0). It excels at global consistency but struggled with rare classes when class weights were removed.
*   **U-Net**: The baseline (mIoU 0.46). While improved by weighted loss, the lack of pretrained features and limited receptive field prevents it from competing with modern architectures on complex urban scenes.

### 6.3 Evaluation Results Breakdown

We analyzed the training progression across different phases to understand the impact of Loss Functions and Transfer Learning strategies. The table below summarizes the Mean IoU (mIoU) evolution.

| Model | Phase 1: Frozen/Weighted (Ep 0-10) | Phase 2: Unfrozen/Weighted (Ep 10-20) | Phase 3: Unfrozen/Unweighted | Final Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net** | 0.4009 (Weighted) | - | **0.4611** (Ep 18) | Mixed Strategy (Weighted $\to$ Unweighted) is best. Pure Unweighted stagnated at ~0.33. |
| **DeepLabV3+** | 0.5393 (Frozen) | - | **0.6324** (Ep 20) | Freezing backbone is critical. Unfreezing + Unweighted Fine-tuning yields best results. |
| **SegFormer** | 0.4279 (Frozen) | **0.5741** (Ep 20) | 0.5287 (Ep 30) | **Weighted Loss is dominant**. Switching to unweighted loss caused degradation (0.57 $\to$ 0.53). |

#### Key Findings:
1.  **Loss Function**:
    *   **CNNs (U-Net, DeepLab)** benefit from a "Mixed" strategy: Use Weighted Loss to learn rare classes, then Unweighted Loss to refine global pixel accuracy.
    *   **Transformers (SegFormer)** require **persistent Class Weighting**. Unlike CNNs, the Self-Attention mechanism tends to overfit to the most frequent textures (Road, Building) if the class penalty is removed, leading to a drop in mIoU.
2.  **Transfer Learning**:
    *   Freezing the backbone for the first 10 epochs was essential for both DeepLab and SegFormer to stabilize the randomly initialized decoder heads.
    *   Unfreezing the backbone (Phase 2) provided the largest performance jump for SegFormer (+15% mIoU).
3.  **Resolution**: Full-scale training was attempted but deemed computationally infeasible (~6h/epoch vs 15m/epoch for crops) on the target hardware. All reported results use 512x1024 crops.


---

## 7. Conclusion

This project successfully implemented a robust training pipeline for semantic segmentation on consumer hardware.
*   **Recommendation**: **DeepLabV3+** is the optimal choice for this setup, offering the best balance of accuracy and stability.
*   **Key Learnings**:
    *   **Gradient Accumulation** allows training complex models on small GPUs.
    *   **Class Weighting** is the single most important factor for handling the "long-tail" of automotive data.
    *   **Transformers** (SegFormer) show great promise but require careful hyperparameter tuning regarding loss weighting compared to robust CNNs.
