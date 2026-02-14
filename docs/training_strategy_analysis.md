# Training Strategy Analysis

This document details the training configurations, architectural decisions, and the evolution of our approach for each semantic segmentation model. The choices were driven by the need to balance model convergence, class imbalance handling, and the constraints of a single 8GB GPU.

---

## 1. U-Net (The Baseline)

**Goal**: Establish a baseline performance with a standard encoder-decoder architecture.

### Strategy Evolution
1.  **Attempt 1: Unweighted Training (10 Epochs)**
    *   **Configuration**: Standard Cross-Entropy Loss, Learning Rate 1e-4.
    *   **Outcome**: Failed. The model converged to predicting only the most frequent classes (*Road*, *Building*, *Vegetation*).
    *   **Evidence**: [Unweighted Training Plot](../results/unet/unet_unweighted/training_plot.png). The mIoU stagnated around 0.32, with rare classes having nearly 0.0 IoU.

2.  **Attempt 2: Weighted Initialization + Unweighted Fine-tuning (Selected Strategy)**
    *   **Phase 1 (Epochs 0-10)**: **Weighted Cross-Entropy Loss** (LR 1e-4).
        *   *Motivation*: Aggressively penalize the model for missing rare classes (*Wall*, *Pole*, *Traffic Light*).
    *   **Phase 2 (Epochs 10-20)**: **Unweighted Cross-Entropy Loss** (LR 1e-5).
        *   *Motivation*: Once the model has learned the existence of rare objects, we switch to unweighted loss with a lower learning rate. This refines the pixel-wise accuracy and prevents the "noisy" gradients of high-weight classes from distorting the well-learned features of frequent classes.

### Result Analysis
*   [Combined Training Plot](../results/unet/training_plot.png)
*   **Observation**: A sharp drop in loss is visible at **Epoch 10** when switching to the unweighted loss. The validation mIoU continues to improve, reaching ~0.46.

---

## 2. DeepLabV3+ (The Specialist)

**Goal**: Leverage a powerful pretrained backbone (ResNet-50) and Atrous Spatial Pyramid Pooling (ASPP) for context.

### Strategy: Two-Stage Transfer Learning
1.  **Phase 1: Frozen Backbone (Epochs 0-10)**
    *   **Configuration**: Weighted Loss, LR 1e-4. **ResNet-50 Frozen**.
    *   **Motivation**: We only train the randomly initialized Decoder and ASPP heads. Freezing the backbone preventing "catastrophic forgetting" of the ImageNet features due to the initial unstable gradients from the new layers.
    *   **Evidence**: [Phase 1 Training Plot](../results/deeplab/tenth_epoch_weighted_frozen/training_plot.png).

2.  **Phase 2: Unfrozen Fine-tuning (Epochs 10-20)**
    *   **Configuration**: Unweighted Loss, LR 1e-4. **Backbone Unfrozen**.
    *   **Motivation**: We unfreeze the backbone to allow the ResNet layers to adapt from general object classification (ImageNet) to specific urban scene understanding (Cityscapes). We switch to unweighted loss to maximize global consistency now that the rare classes are registered.

### Result Analysis
*   [Full Training Plot](../results/deeplab/training_plot.png)
*   **Observation**: The transition at Epoch 10 is dramatic. The loss drops significantly (from ~0.51 to ~0.26) immediately after unfreezing the backbone. This confirms that the pretrained features, while useful, need significant adaptation for the dense prediction task of segmentation. This strategy yielded the best overall performance (mIoU 0.63).

---

## 3. SegFormer (The Transformer)

**Goal**: Evaluate the efficacy of Vision Transformers (MiT-B0) on limited hardware.

### Strategy: Two-Stage Weighted Training
The training process can be observed in the combined [Phase 1+2 Training Plot](../results/segformer/10plus10/training_plot.png), which covers the first 20 epochs.

1.  **Phase 1: Frozen Backbone (Epochs 0-10)**
    *   **Configuration**: Weighted Loss. Backbone Frozen.
    *   **Outcome**: The model learns to use the pretrained features but performance plateaus around mIoU ~0.52.

2.  **Phase 2: Unfrozen Weighted (Epochs 10-20)**
    *   **Configuration**: Weighted Loss. Backbone Unfrozen.
    *   **Outcome**: **Significant Improvement**. As seen in the plot, immediately after Epoch 10 (when the backbone is unfrozen), there is a sharp increase in mIoU and a reduction in loss. The model effectively refined its internal representations for the dataset, boosting performance on less frequent classes like *Pole* and *Sign*. The mIoU peaked at **0.57**.

3.  **Phase 3: Unfrozen Unweighted (Epochs 20-30)**
    *   **Configuration**: Unweighted Loss.
    *   **Outcome**: **Degradation**.
    *   **Evidence**: [Full Training Plot](../results/segformer/training_plot.png).
    *   **Analysis**: While the loss dropped further (to ~0.23), the mIoU **decreased** to 0.53. This strongly suggests that without the class weights, the Transformer begun to overfit the texture and frequency of the dominant classes (*Road*, *Building*), ignoring the structural cues for smaller objects. Unlike the CNN (DeepLab), the Transformer's self-attention mechanism appears more sensitive to the class distribution shift.

### Summary
For SegFormer, **unfreezing the backbone** (Phase 2) provided the critical performance boost, but maintaining **Weighted Loss** was essential to prevent regression on rare classes. The final unweighted stage, while beneficial for CNNs (U-Net, DeepLab), was detrimental to the Transformer.

---

## 4. Full Scale Training (The Next Step)

**Goal**: Improve fine-grained segmentation details by training on full-resolution images (1024x2048) instead of random crops (512x1024).

### Hypothesis
Models with efficient attention mechanisms (SegFormer) or dilated convolutions (DeepLabV3+) might benefit significantly from seeing the global context and full resolution during training, provided the hardware can support it.

### Implementation
*   **Flag**: `--full_scale`
*   **modification**: The `CityscapesDataset` now supports a `crop=False` mode during training.
*   **VRAM Consideration**: This mode is significantly more memory-intensive. It is recommended primarily for **SegFormer** (efficient attention) or **DeepLabV3+** with a frozen backbone. U-Net may OOM on 8GB VRAM with full-scale images.

