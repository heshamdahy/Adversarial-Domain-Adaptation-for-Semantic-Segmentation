# Baseline 1 — U-Net Adversarial Domain Adaptation

## Overview

This baseline investigates **synthetic-to-real domain adaptation for semantic segmentation** using a U-Net as the segmentation generator.

The model is trained on two domains:

* **GTA5** — Synthetic source domain with pixel-level segmentation annotations.
* **Cityscapes** — Real target domain.

The main idea is to train the U-Net with two objectives:

1. Learn accurate semantic segmentation using the labeled GTA5 dataset.
2. Learn domain-invariant segmentation representations through adversarial training.

A CNN-based discriminator is introduced to distinguish between segmentation maps generated from the two domains.

---

## Problem Definition

Semantic segmentation models trained on synthetic datasets often suffer from a significant **domain gap** when applied to real-world images.

Although GTA5 provides large amounts of automatically generated pixel-level annotations, its visual and semantic distribution differs from real-world datasets such as Cityscapes.

The goal of this baseline is therefore:

> Train a segmentation model using labeled synthetic data while adapting its predictions toward the real target domain.

The adaptation can be summarized as:

```text
Synthetic Domain                         Real Domain
     GTA5                                  Cityscapes
       │                                       │
       ▼                                       ▼
   Input Image                             Input Image
       │                                       │
       └──────────────┐       ┌────────────────┘
                      ▼       ▼
                    ┌──────────┐
                    │  U-Net   │
                    │Generator │
                    └────┬─────┘
                         │
                         ▼
                 Segmentation Maps
                         │
                         ▼
                ┌─────────────────┐
                │ Discriminator D │
                │      CNN        │
                └────────┬────────┘
                         │
                         ▼
                    Domain Label
```

---

## Architecture

### Generator — U-Net

The U-Net is used as the segmentation generator.

```text
Input Image
     │
     ▼
┌─────────────┐
│   Encoder   │
└──────┬──────┘
       │
       ▼
   Bottleneck
       │
       ▼
┌─────────────┐
│   Decoder   │
└──────┬──────┘
       │
       ▼
Segmentation Map
```

The encoder extracts hierarchical visual features, while the decoder progressively reconstructs spatial information.

Skip connections transfer high-resolution features from the encoder to the decoder, allowing the network to preserve object boundaries and fine-grained spatial information.

For an image of spatial size `H × W` and `C` semantic classes, the generator produces:

```text
G(x) → [C, H, W]
```

where each pixel contains a prediction over the semantic classes.

---

## Discriminator

A CNN-based discriminator is used as the adversarial component.

Its input is the **segmentation map produced by the generator**, rather than the original RGB image.

```text
Segmentation Map
       │
       ▼
┌──────────────┐
│ CNN          │
│ Discriminator│
└──────┬───────┘
       │
       ▼
   Domain Prediction
```

The discriminator learns to distinguish between segmentation predictions originating from the source and target domains.

Conceptually:

```text
GTA5 → U-Net → Source Segmentation → Discriminator → Source / Fake

Cityscapes → U-Net → Target Segmentation → Discriminator → Target / Real
```

The discriminator therefore learns the difference between the segmentation-map distributions of the two domains.

---

## Training Objective

The training process combines **supervised segmentation learning** with **adversarial domain adaptation**.

### 1. Segmentation Loss

Since GTA5 provides ground-truth segmentation masks, the generator is trained using a supervised segmentation loss.

For a source image `x_s` and its ground-truth mask `y_s`:

```text
x_s → U-Net → ŷ_s
```

The segmentation loss can be expressed as:

```text
L_seg = CE(G(x_s), y_s)
```

where `CE` represents the pixel-wise cross-entropy loss.

This loss ensures that the generator learns the actual semantic segmentation task.

---

### 2. Adversarial Loss

The discriminator receives segmentation predictions and learns to distinguish their domain.

The generator, on the other hand, tries to make the source-domain predictions appear similar to the target-domain predictions from the discriminator's perspective.

Therefore, adversarial learning encourages the generator to produce **domain-invariant segmentation predictions**.

The generator objective can be represented conceptually as:

```text
L_G = L_seg + λ_adv L_adv
```

where:

* `L_seg` — supervised segmentation loss.
* `L_adv` — adversarial domain adaptation loss.
* `λ_adv` — weight controlling the contribution of adversarial learning.

---

## Training Pipeline

### Source Domain

```text
GTA5 Image
    │
    ▼
   U-Net
    │
    ▼
Source Segmentation Prediction
    │
    ├──────────────► Segmentation Loss
    │                      ▲
    │                      │
    │                  Ground Truth
    │
    ▼
Discriminator
    │
    ▼
Domain Prediction
```

### Target Domain

```text
Cityscapes Image
      │
      ▼
     U-Net
      │
      ▼
Target Segmentation Prediction
      │
      ▼
Discriminator
      │
      ▼
Domain Prediction
```

During training, GTA5 provides the semantic supervision, while Cityscapes contributes target-domain information for adversarial adaptation.

---

## Why Use Segmentation Maps for the Discriminator?

Instead of discriminating between RGB images, the discriminator operates on the output of the segmentation network.

This focuses the adversarial learning process on the **semantic structure** of the predictions.

The objective is not simply:

```text
Synthetic image ≠ Real image
```

but rather:

```text
Synthetic-domain segmentation
        ↓
make its distribution
        ↓
closer to
        ↓
Real-domain segmentation
```

This encourages the generator to learn segmentation representations that are less sensitive to the source-target domain gap.

---

## Datasets

### GTA5

GTA5 is used as the **synthetic source domain**.

Each training example contains:

```text
RGB Image
     +
Pixel-level Segmentation Mask
```

The ground-truth annotations provide the supervision required for the segmentation loss.

### Cityscapes

Cityscapes is used as the **real target domain**.

During adversarial adaptation, the RGB images are used to expose the generator and discriminator to the target-domain distribution.

The target annotations are not required for the adversarial training objective and can instead be reserved for evaluation.

---

## Baseline Goal

The purpose of this baseline is to establish a strong CNN-based reference model using U-Net.

The experiment evaluates whether adversarial learning can improve the transfer of a segmentation model from:

```text
GTA5
Synthetic Domain
      ↓
      ↓ Domain Adaptation
      ↓
Cityscapes
Real Domain
```

---

## Evaluation

The primary evaluation should focus on semantic segmentation performance on the target domain.

Recommended metrics include:

* **Mean Intersection over Union (mIoU)**
* **Pixel Accuracy**
* **Per-class IoU**
* **Mean Pixel Accuracy**

The most important metric for comparing the three baselines is **mIoU on Cityscapes**.

---

## Expected Comparison

This baseline will serve as the first reference point for the following architectures:

```text
Baseline 1
U-Net
   ↓
Adversarial Adaptation
   ↓
Cityscapes


Baseline 2
DeepLabV3+
   ↓
Adversarial Adaptation
   ↓
Cityscapes


Baseline 3
Transformer-based Segmentation
   ↓
Adversarial Adaptation
   ↓
Cityscapes
```

The discriminator, domain-adaptation strategy, datasets, and evaluation protocol should remain as consistent as possible across all baselines.

This allows the experiment to investigate how the **choice of segmentation architecture** affects synthetic-to-real domain adaptation.
