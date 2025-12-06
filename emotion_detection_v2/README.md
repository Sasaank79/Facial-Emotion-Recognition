# Emotion Detection v2 - Training Code

This folder contains the training pipeline I used to build the EfficientNetV2 model.

## What's Here

```
src/
├── models/          # EfficientNetV2 wrapper
├── data/            # Data loaders and augmentation
├── inference/       # Predictor class for inference
└── utils/           # Training utilities (EMA, Label Smoothing)

notebooks/
└── train_on_colab.ipynb   # Main training notebook (use this)
```

## Results I Got

Trained on FER-2013 (resized from 48x48 to 224x224):

| Metric | Score |
|--------|-------|
| Test Accuracy | 72.68% |
| F1 Macro | 0.72 |

## Key Training Choices

- **Model**: EfficientNetV2-S (pretrained on ImageNet)
- **Loss**: CrossEntropy with Label Smoothing (0.1)
- **Optimizer**: AdamW with cosine LR schedule
- **Augmentation**: Albumentations (ShiftScaleRotate, CoarseDropout, brightness/contrast)
- **Stability**: EMA weights for better generalization

## How to Train

Easiest way is Google Colab (free T4 GPU):

1. Upload `notebooks/train_on_colab.ipynb` to Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Models auto-save to your Google Drive

See `COLAB_GUIDE.md` for detailed instructions.

## Compared to the Old Version

| What | Old (ResNet50) | New (EfficientNetV2) |
|------|----------------|---------------------|
| Accuracy | ~68% | 72.68% |
| F1 Score | ~65% | 72% |
| Speed | Slower | Faster (smaller model) |

The main improvements came from Label Smoothing (helped with class imbalance) and EMA weights (better generalization than raw checkpoints).
