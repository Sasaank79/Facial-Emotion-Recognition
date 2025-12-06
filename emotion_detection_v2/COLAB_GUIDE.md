# Colab Training Guide - Emotion Detection v2.0

## 🚀 Quick Setup

### Step 1: Upload to Google Drive

```bash
# On your local machine, zip the project (optional, notebook creates structure automatically)
cd /Users/mypc/Documents/Facial_Emotion_Detection/
zip -r emotion_v2.zip emotion_detection_v2/

# Upload emotion_v2.zip to your Google Drive (or just upload the notebook)
```

### Step 2: Open in Colab

1. Go to Google Drive
2. Upload `emotion_detection_v2/notebooks/train_on_colab.ipynb`
3. Right-click → Open with → Google Colaboratory

### Step 3: Set GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU** (free tier)
3. Save

### Step 4: Run All Cells

Click **Runtime → Run all** and let it go!

---

## 📂 Storage Strategy Explained

The notebook uses **hybrid storage** for optimal performance:

### Local Storage (`/content/`) - FAST
- ✅ Code execution (all `.py` files)
- ✅ Training data (copied once, reused)
- ✅ Temporary files
- ❌ **Deleted when Colab disconnects** (not persistent)

### Google Drive (`/drive/MyDrive/Emotion_Detection_v2/`) - PERSISTENT
- ✅ Model checkpoints (auto-saved during training)
- ✅ Results (confusion matrix, metrics, curves)
- ✅ ONNX export
- ✅ **Survives Colab disconnects** (permanent until you delete)

### How It Works (Automatic)

The notebook creates **symlinks**:
```
/content/emotion_detection_v2/models → /drive/.../Emotion_Detection_v2/models
/content/emotion_detection_v2/results → /drive/.../Emotion_Detection_v2/results
```

**When training saves to `models/best_model.pth`**, it's actually saving to **your Drive**!

**Benefits**:
- Training is fast (local execution)
- Models are safe (auto-saved to Drive)
- No manual copying needed

---

## ⏱️ Training Time Estimates

| Dataset Size | Epochs | T4 GPU Time |
|--------------|--------|-------------|
| FER-2013 (~28K train) | 30 | ~1.5 hours |
| FER-2013 (~28K train) | 50 | ~2.5 hours |
| RAF-DB (~12K train) | 50 | ~1.5 hours |

**Colab Free Tier Limits**:
- Max session: 12 hours
- Then disconnected
- **But your models are saved to Drive!**

---

## 🔄 If Colab Disconnects Mid-Training

### Don't Panic!

Your models are **already in Drive**. The notebook saves checkpoints after every epoch:

1. **Reconnect to GPU** (Runtime → Change runtime type → T4 GPU)
2. **Re-run the notebook** from the top
3. **Load the checkpoint**:

```python
# The notebook automatically loads from Drive if it exists
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training from where you left off
for epoch in range(start_epoch, CONFIG['epochs']):
    ...
```

(The notebook already includes resume logic!)

---

## 📥 Data Options

The notebook supports 3 ways to get data:

### Option 1: Download from Kaggle (Recommended)

```python
# Uncomment in notebook:
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d msambare/fer2013 -p data/ --unzip
```

**Setup**: Upload your `kaggle.json` to Drive first.

### Option 2: Copy from Your Drive

```python
# Uncomment in notebook:
drive_data = '/content/drive/MyDrive/Facial_Emotion_Detection/project/data'
shutil.copytree(drive_data, 'data', dirs_exist_ok=True)
```

**Use if**: You already have FER-2013 in your Drive.

### Option 3: Direct Download (Fastest)

The notebook includes a fallback download (requires a public link).

---

## 📊 Expected Results

### FER-2013 (48x48, resized to 224x224)
```
Test Accuracy: 68-74%
F1 Macro: 65-72%
Training time: ~2-3 hours (T4 GPU)
```

**Why not higher?**  
FER-2013 has noisy labels (~10% mislabeled) and low resolution (48x48).

### RAF-DB (224x224 native, cleaner)
```
Test Accuracy: 75-80%
F1 Macro: 73-78%
Training time: ~1.5-2 hours (T4 GPU)
```

**Recommended** if you can get RAF-DB dataset.

---

## 💾 What Gets Saved to Drive

After training completes, check your Drive:

```
/MyDrive/Emotion_Detection_v2/
├── models/
│   ├── best_model.pth         # Best checkpoint (use this for inference)
│   └── emotion_model.onnx     # ONNX export (deployment-ready)
└── results/
    ├── confusion_matrix.png   # Confusion matrix visualization
    ├── training_curves.png    # Loss/F1 curves
    └── test_results.txt       # Final metrics (accuracy, F1)
```

**Download** these files to your local machine for portfolio/deployment.

---

## 🐛 Troubleshooting

### "No GPU detected"
- Runtime → Change runtime type → T4 GPU → Save
- Restart runtime (Runtime → Restart runtime)

### "CUDA out of memory"
```python
# Reduce batch size in CONFIG cell
CONFIG['batch_size'] = 32  # or 16
```

### "Dataset not found"
- Check data download cell completed successfully
- Verify: `!ls -lh data/train/`

### "Drive mount failed"
- Click the link in output to authorize
- Make sure you're logged into Google

### "Training is slow"
- Verify GPU: `!nvidia-smi`
- Should show "T4" with GPU usage during training
- If showing CPU, reconnect to GPU runtime

---

## 🎓 After Training

### Download Best Model

```bash
# In Colab:
from google.colab import files
files.download('models/best_model.pth')
files.download('models/emotion_model.onnx')
```

Or: Right-click files in Drive → Download

### Use Locally

```python
# On your local machine:
from src.inference.predictor import EmotionPredictor

predictor = EmotionPredictor(
    model_path='path/to/best_model.pth',
    device='cpu'
)

emotion, conf, probs = predictor.predict(face_image)
```

---

## 💡 Pro Tips

1. **Run during off-peak hours** (late night in your timezone) for longer sessions
2. **Monitor Drive space** (free tier = 15GB)
3. **Keep Colab tab active** to avoid disconnects (use browser extension to prevent sleep)
4. **Train for 30 epochs first** to test, then do full 50 if results look good
5. **Check results/test_results.txt** for final performance - if accuracy is low (<65%), something went wrong

---

## 🚀 Next Steps After Training

1. **Evaluate results**: Check confusion matrix, identify weak emotions
2. **Deploy**: Use ONNX model for production
3. **Improve**:
   - Get better data (RAF-DB, AffectNet)
   - Try stronger augmentation (`'strong'` in CONFIG)
   - Train ensemble (3-5 models, average predictions)

---

**Questions?** All code is self-documented in the notebook cells. Just run and it works!
