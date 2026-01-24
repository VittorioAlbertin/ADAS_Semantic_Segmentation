# Model Architecture Specifications

This document details the specific architectural choices and implementation details for the three models used in this project.

## 1. U-Net (Custom Implementation)

**Overview**: A classic U-shaped encoder-decoder network, implemented from scratch to provide a baseline "clean slate" approach without pretraining.

### Implementation Details (`src/models/unet.py`)
*   **Depth**: 5 Levels.
*   **Base Width**: 64 channels.
*   **VRAM Handling**: Uses **Instance Normalization** (`nn.InstanceNorm2d`) instead of Batch Normalization. This is critical because we train with a batch size of 1 per GPU (effective batch size 4 via accumulation), which renders BatchNorm unstable.

### Structure
1.  **Encoder (Contracting Path)**:
    *   **Level 1**: Input (3) -> 64 ch.
    *   **Level 2**: Downsample (Maxpool 2x2) -> 128 ch.
    *   **Level 3**: Downsample -> 256 ch.
    *   **Level 4**: Downsample -> 512 ch.
    *   **Level 5 (Bottleneck)**: Downsample -> 1024 ch.
    *   *Block Structure*: `DoubleConv` = [Conv3x3 -> InstNorm -> ReLU] x 2.

2.  **Decoder (Expansive Path)**:
    *   **Upsampling Application**: Uses **Bilinear Interpolation** followed by a Convolution, rather than Transposed Convolutions, to reduce checkerboard artifacts.
    *   **Skip Connections**: Features from the Encoder are concatenated with upsampled features to preserve spatial details.
    *   **Head**: Final 1x1 Convolution maps 64 channels to 19 classes.

---

## 2. DeepLabV3+ (ResNet-50 Backbone)

**Overview**: A state-of-the-art CNN-based model that combines a powerful feature extractor (ResNet) with a multi-scale context module (ASPP) and a decoder module for boundary refinement.

### Implementation Details (`src/models/deeplab.py`)
*   **Backbone**: **ResNet-50** (`torchvision.models.resnet50`), pretrained on **ImageNet**.
*   **Modifications for Segmentation**:
    *   **Dilated (Atrous) Convolutions**: The strides in Layer 3 and Layer 4 are replaced with dilations (2 and 4 respectively) to maintain a larger output feature map (Output Stride = 16) without losing coverage.
*   **ASPP (Atrous Spatial Pyramid Pooling)**:
    *   Captures context at multiple scales (Rates: 6, 12, 18).
    *   Includes a global average pooling branch.

### The "Plus" Decoder
Standard DeepLabV3 outputs directly from ASPP (1/16 resolution) and upsamples 16x, often losing fine details. We implement the **V3+ Decoder**:
1.  **Low-Level Features**: Extracted from ResNet **Layer 1** (1/4 resolution, 256 channels).
2.  **Projection**: Low-level features are projected to 48 channels via 1x1 Conv.
3.  **Fusion**: ASPP output (256 ch) is upsampled 4x and concatenated with projected low-level features (48 ch).
4.  **Refinement**: Two 3x3 Convolutions refine the fused features.
5.  **Final Output**: Upsampled 4x to restore full resolution.

---

## 3. SegFormer (Transformer-Based)

**Overview**: A modern hierarchical Transformer framework that unifies Encoders and Decoders. It is efficient and achieves high performance without complex dilated convolutions.

### Implementation Details (`src/models/segformer.py`)
*   **Framework**: Implemented using the Hugging Face `transformers` library.
*   **Backbone**: **MiT-B0** (`nvidia/mit-b0`).
    *   The "B0" variant is the lightest mix-transformer backbone, chosen strictly to fit within the **8GB VRAM** constraint.
    *   Pretrained on **ImageNet**.
*   **Decoder**: MLP Decoder.
    *   The SegFormer decoder aggregates features from all 4 stages of the encoder (1/4, 1/8, 1/16, 1/32).
    *   It projects them to a common dimension and fuses them to predict masks.

### Fine-Tuning Process
*   We load `nvidia/mit-b0` (which provides the encoder weights).
*   The Decoder head is initialized randomly for the 19 Cityscapes classes.
*   **Output Handling**: The model natively outputs logits at 1/4 resolution. We perform a final Bilinear Upsampling (4x) in the `forward` pass to match the input resolution.
