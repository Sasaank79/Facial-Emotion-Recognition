# Project Walkthrough

This is a detailed breakdown of how I built this emotion recognition system, the problems I ran into, and how I solved them.

## The Journey

I went through three major iterations:

1. **Custom CNN** (~55% accuracy) - Good for learning, but hit a ceiling fast
2. **ResNet50** (~68% accuracy) - Better, but slow and still struggled with minority classes
3. **EfficientNetV2** (~72.7% accuracy) - Best balance of speed and accuracy

## Training Setup

### Model
I used `tf_efficientnetv2_s` from the `timm` library. It's lighter than ResNet50 and trains faster.

### Handling Class Imbalance
FER-2013 has very few "Disgust" samples. To fix this, I used:
- **Label Smoothing (0.1)**: Prevents the model from being overconfident
- **Weighted sampling**: Gives minority classes more attention during training

### Augmentation
Used Albumentations for data augmentation:
- ShiftScaleRotate
- CoarseDropout (simulates occlusion)
- RandomBrightnessContrast

### Training Stability
- **EMA (Exponential Moving Average)**: Keeps a smoothed copy of model weights, which generalizes better than the raw final checkpoint
- **AdamW + Cosine LR**: Standard but effective

## The Inference Problem

After training, I had a model that scored 72% on the test set. Great. But when I pointed a webcam at my face? It predicted "Neutral" for everything, or flickered randomly.

### What Went Wrong

1. **Face cropping**: OpenCV's Haar Cascade was cutting off chins and foreheads. The model was trained on full faces, so it got confused.

2. **Preprocessing mismatch**: My inference code was resizing differently than training. Small difference, big impact.

3. **Frame-by-frame noise**: Webcam frames are noisy. One frame looks "Sad" because of a shadow, next frame looks "Happy".

### How I Fixed It

1. **Switched to MediaPipe**: Google's face detection is way more robust. Added 20% padding around detected faces to capture the full head.

2. **Matched preprocessing exactly**: Same resize (224x224), same normalization (ImageNet mean/std), same tensor format.

3. **Temporal smoothing**: Built a simple class that averages predictions over 10 frames. Now the output is stable and reflects actual expressions, not random noise.

## File Overview

| File | Purpose |
|------|---------|
| `demo_webcam_fixed.py` | Main inference script with all the fixes |
| `run_demo.sh` | Creates a venv and runs the demo cleanly |
| `emotion_detection_v2/src/` | Training code (model, data loaders, utils) |
| `Emotion_Detection_From_Colab/` | Trained model and results |

## What I'd Do Differently

- **Add Mixup/CutMix**: Didn't implement these, but they'd probably add another 1-2% accuracy
- **Fine-tune on personal data**: The model works okay on my face but not great—fine-tuning on a few hundred samples of my own expressions would help
- **Try a lighter model**: For real mobile deployment, something like MobileNet might be better

## Lessons Learned

- A good model on paper can fail completely in the real world if your inference pipeline doesn't match training
- Face detection quality matters more than I expected
- Temporal smoothing is a simple trick that makes a huge difference for real-time applications
