# Facial Emotion Recognition

A deep learning project that detects facial emotions in real-time using a webcam. Built with PyTorch and EfficientNetV2, trained on the FER-2013 dataset.

## What I Built

I started this project with a simple CNN, then moved to ResNet50, and finally landed on **EfficientNetV2** which gave me the best results. The model hits **72.7% accuracy** on FER-2013, which is pretty solid for this notoriously noisy dataset.

The real challenge wasn't just training the model—it was making it work *reliably* in real-time. The raw webcam feed is messy, and I had to solve a bunch of issues to get stable predictions.

## Quick Start

```bash
git clone https://github.com/Sasaank79/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition

chmod +x run_demo.sh
./run_demo.sh
```

Press `d` to toggle debug mode (shows all emotion probabilities), `q` to quit.

## How It Works

### The Model
- **Architecture**: EfficientNetV2-S (via `timm`)
- **Training tricks**: Label Smoothing to handle class imbalance, EMA weights for stability
- **Augmentation**: Albumentations (ShiftScaleRotate, CoarseDropout, brightness/contrast)

### Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 72.68% |
| F1 Macro | 0.72 |

The hardest class was Disgust (very few training samples), but Label Smoothing helped a lot there.

### Real-Time Inference

Getting the model to work on webcam was tricky. Here's what I had to fix:

1. **Face Detection**: Switched from Haar Cascade to **MediaPipe**. Haar was cropping faces too tight and cutting off chins, which confused the model.

2. **Preprocessing Match**: Made sure the inference pipeline uses the exact same transforms as training (224x224 resize, ImageNet normalization).

3. **Smoothing**: Added a 10-frame rolling average to stop the predictions from flickering every frame.

### Known Limitations
- Runs at ~5 FPS on my MacBook Air (CPU-only)
- Works best in good lighting
- May need fine-tuning for your specific face/camera

## Project Structure

```
├── demo_webcam_fixed.py       # Real-time inference
├── run_demo.sh                # Sets up venv and runs demo
├── emotion_detection_v2/      # Training code
│   ├── src/                   # Models, data loaders, utils
│   └── notebooks/             # Colab training notebook
└── Emotion_Detection_From_Colab/
    ├── models/                # Trained weights
    └── results/               # Metrics and plots
```

## Training Your Own

If you want to retrain or fine-tune:
- Use `emotion_detection_v2/notebooks/train_on_colab.ipynb` (free T4 GPU)
- Or run locally if you have a GPU

## Tech Stack
Python, PyTorch, OpenCV, MediaPipe, Albumentations
