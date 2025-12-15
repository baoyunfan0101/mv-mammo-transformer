# **mv-mammo-transformer**

## Overview

This project implements a single-view / multi-view multi-task mammography classification framework.

Key features:
- Plugable model training pipelines
- Transformer-based multi-view feature fusion
- GradCAM losses for lesion guidance

## Author

Yunfan Bao  
Ziyuan Zhang

## Installation

```bash
git clone https://github.com/baoyunfan0101/mv-mammo-transformer.git
cd mv-mammo-transformer
pip install -r requirements.txt
```

## Environment Configuration

This project relies on environment variables for resolving paths.

Before running any training or evaluation scripts:
1. Update `.env` (local) or `.env.colab` (Colab) according to your environment
2. Run `update_status.py` to synchronize actual file paths with your setup

```bash
python -m scripts.update_status
```

## Data Preparation

This project assumes mammography data organized at the **breast level** with **four standard views** per study:
- L-CC, L-MLO, R-CC, R-MLO

For experiments involving lesion guidance, optional lesion bounding boxes can be provided.
These bounding boxes are used **only for GradCAM losses** during training and are **not required** for standard classification or inference.

This project was evaluated using [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/).
The dataset itself is not distributed with this repository.

## Project Tree
```
mv-mammo-transformer/
│
├── README.md
├── requirements.txt
├── bbox.txt
├── status.txt
├── status_colab.txt
├── .env
├── .env.aws
├── .env.colab
├── config.py
│
├── notebooks/
│   ├── colab_train_template.ipynb
│   ├── first_1000_overview.ipynb
│   ├── full_overview.ipynb
│   ├── local_evaluate_template.ipynb
│   ├── local_train_template.ipynb
│   └── local_visualize_preprocess.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── download.py             # main downloader
│   ├── evaluate.py             # main evaluator
│   ├── preprocess.py           # main preprocessor
│   ├── split_data.py           # main data splitter
│   ├── train.py                # main trainer
│   ├── update_bbox.py          # update bboxes for preprocessed images
│   ├── update_status.py        # update status for downloaded and preprocessed path
│   ├── visualize_bbos.py       # visualize preprocessed images with bboxes
│   ├── visualize_preprocess.py # visualize preprocessed images or preprocess flow
│   └── visualize_raw.py        # visualize raw images
│
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── bbox.py             # load and update bbox.csv
    │   ├── breast_level.py     # load breast-level_annotations.csv
    │   ├── finding.py          # load finding_annotations.csv
    │   ├── metadata.py         # load metadata.csv
    │   └── status.py           # load and update status.csv
    │
    ├── datasets/
    │   ├── __init__.py
    │   ├── image_dataset.py    # ImageDataset
    │   ├── image_provider.py   # ImageProvider for ImageDataset
    │   ├── index_provider.py   # IndexProvider for ImageDataset
    │   ├── keys.py             # ImageKey and MultiViewKey
    │   └── label_provider.py   # LabelProvider for ImageDataset
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluator.py        # main function of evaluation
    │   ├── gradcam.py          # GradCAM
    │   └── metrics.py          # metrics
    │
    ├── losses/
    │   ├── __init__.py                 # loss registry, loss factory
    │   ├── composite_loss.py           # CompositeLoss for classification loss and GradCAM loss
    │   ├── evidential_gradcam_loss.py  # EvidentialGradCAMLoss
    │   ├── evidential_loss.py          # EvidentialCELoss and EvidentialKLCELoss
    │   ├── gradcam_loss.py             # GradCAMLoss for leision guide
    │   ├── softmax_gradcam_loss.py     # SoftmaxGradCAMLoss
    │   └── softmax_loss.py             # SoftmaxCELoss and SoftmaxLSCELoss
    │
    ├── models/
    │   ├── backbones/
    │   │   ├── __init__.py         # backbone registry, backbone factory
    │   │   ├── base.py             # BackboneBase for backbone freeze schedule
    │   │   ├── effnet_backbone.py  # EffNetV2SBackbone
    │   │   ├── resnet_backbone.py  # ResNet50Backbone
    │   │   └── swin_backbone.py    # SwinTBackbone, SwinSBackbone, SwinBBackbone
    │   │
    │   ├── heads/
    │   │   ├── __init__.py         # head registry, head factory
    │   │   ├── evidential_head.py  # MultiTaskEvidentialHead
    │   │   └── softmax_head.py     # MultiTaskSoftmaxHead
    │   │
    │   ├── multiview/
    │   │   ├── __init__.py
    │   │   ├── mv_concat.py        # MVConcatFusion
    │   │   └── mv_transformer.py   # MVTransformerFusion
    │   │
    │   ├── singleview/
    │   │   ├── __init__.py
    │   │   └── sv_baseline.py      # SVBaseline
    │   │
    │   └── __init__.py             # model registry, model factory
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── crop.py                 # crop images
    │   ├── dicom.py                # load DICOM files
    │   ├── intensity.py            # CLAHE equalize images
    │   ├── orientation.py          # horizental flip right images
    │   └── pipeline.py             # main preprocess pipeline
    │
    ├── training/
    │   ├── __init__.py
    │   ├── collate.py              # CollateFn
    │   ├── dataloader.py           # build dataset and dataloader
    │   ├── evaluate.py             # evaluate one epoch
    │   ├── forward.py              # forward a batch
    │   ├── freeze_scheduler.py     # backbone freeze scheduler
    │   ├── lr_scheduler.py         # learning rate scheduler
    │   ├── train.py                # train one epoch
    │   └── train_epoch.py          # main trainer
    │
    ├── transforms/
    │   ├── __init__.py             # transform registry, transform factory
    │   ├── mv_consistent.py        # consistent transforms for multi-view
    │   ├── mv_test.py              # default transforms for multi-view test
    │   ├── sv_baseline.py          # tranforms for single-view
    │   └── sv_test.py              # default transforms for single-view test
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── config_utils.py         # config utilities
    │   ├── device_utils.py         # device utilities
    │   └── log_utils.py            # log utilities
    │
    └── __init__.py
```

## Model

### Preprocess
```
┌───────────────────────────────────────────────┐
│                                               │
│       L-CC     L-MLO     R-CC     R-MLO       │
│     (raw images, single-channel grayscale)    │
│                                               │
└─────────┬────────┬────────┬────────┬──────────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │                 Clean                │
    │         (Remove Corner Text)         │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
          │        │  ┌─────▼────────▼─────┐
          │        │  │   Horizontal Flip  │
          │        │  └─────┬────────┬─────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │                 Crop                 │
    │             (Fixed Size)             │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │                Enhance               │
    │         (CLAHE equalization)         │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
┌─────────▼────────▼────────▼────────▼──────────┐
│                                               │
│       L-CC     L-MLO     R-CC     R-MLO       │
│(preprocessed images, single-channel grayscale)│
│                                               │
└───────────────────────────────────────────────┘
```

### Transforms
```
┌───────────────────────────────────────────────┐
│                                               │
│       L-CC     L-MLO     R-CC     R-MLO       │
│(preprocessed images, single-channel grayscale)│
│                                               │
└─────────┬────────┬────────┬────────┬──────────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │      Shared Geometric Operations     │
    │     (Ratation, Translation, ...)     │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │      View-specific augmentations     │
    │  (Gaussian Blur, Color jitter, ...)  │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
┌─────────▼────────▼────────▼────────▼──────────┐
│                                               │
│       L-CC     L-MLO     R-CC     R-MLO       │
│   (input images, single-channel grayscale)    │
│                                               │
└───────────────────────────────────────────────┘
```

### Single-View
```
┌───────────────────────────────────────────────┐
│                                               │
│       L-CC  /  L-MLO  /  R-CC  /  R-MLO       │
│   (input images, single-channel grayscale)    │
│                                               │
└──────────────────────┬────────────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │                Backbone              │
    │   (CNN: ResNet50, EffientNet_V2_S;   │
    │ViT: Swin_Tiny, Swin_Small, Swin_Base)│
    └──────────┬────────────────┬──────────┘
               │                │
       ┌───────▼───────┐┌───────▼───────┐
       │    BI-RADS    ││    Density    │
       │ Classification││ Classification│
       │     Heads     ││     Heads     │
       │   (Softmax,   ││   (Softmax,   │
       │  Evidential)  ││  Evidential)  │
       └───────┬───────┘└───────┬───────┘
       ┌───────▼───────┐┌───────▼───────┐
       │    BI-RADS    ││    Density    │
       │    Outputs    ││ Classification│
       │   - logits    ││   - logits    │
       │   - probs     ││   - probs     │
       │   - losses    ││   - losses    │
       └───────────────┘└───────────────┘
```

### Multi-View
```
┌───────────────────────────────────────────────┐
│                                               │
│       L-CC     L-MLO     R-CC     R-MLO       │
│   (input images, single-channel grayscale)    │
│                                               │
└─────────┬────────┬────────┬────────┬──────────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │            Shared Backbone           │
    │   (CNN: ResNet50, EffientNet_V2_S;   │
    │ViT: Swin_Tiny, Swin_Small, Swin_Base)│
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │     View-Level Feature Embeddings    │
    └─────┬────────┬────────┬────────┬─────┘
          │        │        │        │
    ┌─────▼────────▼────────▼────────▼─────┐
    │       Cross-View Fusion Module       │
    │ (Concatenation, Transformer Encoder) │
    └──────────┬────────────────┬──────────┘
               │                │
       ┌───────▼───────┐┌───────▼───────┐
       │    BI-RADS    ││    Density    │
       │ Classification││ Classification│
       │     Heads     ││     Heads     │
       │   (Softmax,   ││   (Softmax,   │
       │  Evidential)  ││  Evidential)  │
       └───────┬───────┘└───────┬───────┘
       ┌───────▼───────┐┌───────▼───────┐
       │    BI-RADS    ││    Density    │
       │    Outputs    ││ Classification│
       │   - logits    ││   - logits    │
       │   - probs     ││   - probs     │
       │   - losses    ││   - losses    │
       └───────────────┘└───────────────┘
```

## Entry Points

### Training
notebooks/local_train_template.ipynb
notebooks/colab_train_template.ipynb

### Evaluation
notebooks/local_evaluate_template.ipynb

## Disclaimer

This project is intended for research and educational purposes only.
It is not designed for clinical use.