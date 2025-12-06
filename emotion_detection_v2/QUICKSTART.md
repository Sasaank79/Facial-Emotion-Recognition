# 🚀 Quick Start Guide - Emotion Detection v2.0

## Prerequisites

- Python 3.8+
- CUDA GPU (recommended) or CPU (slower)
- 8GB+ RAM

## Installation

```bash
# Navigate to project
cd emotion_detection_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Setup

### Option 1: Use FER-2013 (You Already Have)

```bash
# Your existing FER-2013 data structure should work:
# emotion_detection_v2/data/
# ├── train/
# │   ├── angry/
# │   ├── disgust/
# │   ├── fear/
# │   ├── happy/
# │   ├── neutral/
# │   ├── sad/
# │   └── surprise/
# ├── val/ (or use test/)
# └── test/
```

**IMPORTANT**: FER-2013 images are 48x48. The notebook will resize them to 224x224 automatically. This is suboptimal but works.

### Option 2: Use Higher Quality Dataset (Recommended)

**RAF-DB** (Facial Expression Recognition Database):
- 224x224 high-quality images
- Cleaner labels than FER-2013
- ~15,000 images

Download from: http://www.whdeng.cn/RAF/model1.html (requires registration)

## Training

### Option 1: Jupyter Notebook (Recommended for First-Time)

```bash
# Start Jupyter
cd notebooks
jupyter notebook train_emotion_model.ipynb

# Follow the cells step-by-step
# - First cell: imports and setup
# - Second cell: configure hyperparameters (adjust batch_size if needed)
# - Run all cells to train
```

**Key Hyperparameters to Adjust**:
```python
CONFIG = {
    'batch_size': 64,    # Reduce to 32 if GPU memory limited
    'epochs': 50,        # Can reduce to 30 for faster testing
    'lr': 1e-3,
    'augmentation': 'medium',  # 'light', 'medium', 'strong'
}
```

### Option 2: Google Colab (Free GPU)

```bash
# 1. Zip your project
cd ..
zip -r emotion_v2.zip emotion_detection_v2/

# 2. Upload to Google Drive

# 3. In Colab:
from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/emotion_v2.zip -d /content/
%cd /content/emotion_detection_v2

!pip install -r requirements.txt

# 4. Open and run notebooks/train_emotion_model.ipynb
```

**Colab Tips**:
- Use T4 GPU (Runtime > Change runtime type > T4 GPU)
- Training will take ~2-3 hours for 50 epochs
- Save models back to Drive to avoid losing them

## Monitoring Training

The notebook automatically:
- Shows progress bars for each epoch
- Prints loss, accuracy, F1 score
- Saves best model based on F1 score
- Plots training curves (loss, accuracy, F1)
- Generates confusion matrix after training

**What to Watch**:
- Train loss should decrease steadily
- Val accuracy should improve and plateau
- If val loss increases while train loss decreases → overfitting (reduce epochs or increase augmentation)

## Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total (50 epochs) |
|----------|------------|----------------|-------------------|
| CPU (M1 Mac) | 32 | ~15 min | ~12 hours |
| CPU (Intel) | 32 | ~25 min | ~20 hours |
| GPU (T4 Colab) | 64 | ~2 min | ~2 hours |
| GPU (RTX 3060) | 64 | ~1 min | ~1 hour |

**Recommendation**: Use Google Colab's free T4 GPU.

## Evaluation

After training completes, the notebook automatically:

1. **Loads best model** (based on validation F1)
2. **Runs test set evaluation**
3. **Generates**:
   - Confusion matrix → `models/confusion_matrix.png`
   - Classification report → `models/test_results.txt`
   - Training curves → `models/training_curves.png`

4. **Exports ONNX model** → `models/emotion_model.onnx`

## Using the Trained Model

### Python API

```python
from src.inference.predictor import EmotionPredictor
import numpy as np

# Load model
predictor = EmotionPredictor(
    model_path='models/best_model.pth',
    device='cpu'  # or 'cuda'
)

# Predict single image
face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # Replace with real image
emotion, confidence, all_probs = predictor.predict(face_image)

print(f"Emotion: {emotion}")
print(f"Confidence: {confidence:.2%}")
print(f"All probabilities: {all_probs}")
```

### Webcam Demo

```python
from src.inference.predictor import WebcamEmotionDetector
import cv2

# Initialize detector
detector = WebcamEmotionDetector(
    model_path='models/best_model.pth',
    device='cpu'
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame (detects faces and predicts emotions)
    annotated_frame, results = detector.process_frame(frame)
    
    # Display
    cv2.imshow('Emotion Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ONNX Inference (Production)

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/emotion_model.onnx')

# Prepare input (224x224x3, normalized like training)
input_image = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_image})
logits = outputs[0]
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

print(f"Predicted class: {np.argmax(probs)}")
print(f"Probabilities: {probs}")
```

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size in CONFIG
'batch_size': 32  # or 16
```

### "Dataset not found"
```bash
# Check data directory structure
ls -R data/

# Should show:
# data/train/angry/*.jpg
# data/train/disgust/*.jpg
# etc.
```

### "Model accuracy is low (<60%)"
Possible causes:
1. Data quality (FER-2013 is noisy)
2. Not enough epochs (try 50)
3. Wrong normalization (check transforms use ImageNet stats)
4. Corrupted dataset

### "Training is too slow"
- Use Google Colab with T4 GPU (free)
- Or reduce epochs to 30 for faster iteration
- Or reduce image size to 128x128 (edit CONFIG['img_size'])

## Next Steps

1. **Improve Data**:
   - Get RAF-DB or AffectNet (higher quality than FER-2013)
   - Or use data augmentation more aggressively

2. **Hyperparameter Tuning**:
   - Try `augmentation='strong'`
   - Try different learning rates (1e-4, 5e-4)
   - Try different models (`convnext_tiny`, `tf_efficientnetv2_m`)

3. **Ensemble**:
   - Train 3-5 models with different seeds
   - Average their predictions (usually +1-2% accuracy)

4. **Deploy**:
   - Use ONNX model for production
   - Optimize with ONNX Runtime `SessionOptions.set_optimization_level(RT_OPT_LEVEL_ALL)`
   - Benchmark on your target device

## File Outputs

After training, check these files:

```
models/
├── best_model.pth              # Best checkpoint (PyTorch)
├── emotion_model.onnx          # ONNX export (deployment)
├── confusion_matrix.png        # Confusion matrix visualization
├── test_results.txt            # Detailed metrics
├── training_curves.png         # Loss/accuracy/F1 curves
```

## Performance Expectations

**FER-2013 Dataset** (48x48, noisy):
- Test Accuracy: 68-74%
- Macro F1: 65-72%
- Training Time: ~2-3 hours (GPU)

**RAF-DB Dataset** (224x224, clean):
- Test Accuracy: 75-80%
- Macro F1: 73-78%
- Training Time: ~3-4 hours (GPU)

**State-of-the-art** (with ensembles, TTA):
- Test Accuracy: 85-88%

---

**Questions?** Check the detailed code comments in the notebook cells.
