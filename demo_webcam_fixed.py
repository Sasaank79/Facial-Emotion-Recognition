"""
FIXED Real-Time Emotion Detection - Production Pipeline
Systematic fixes to match training preprocessing exactly.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
import sys
import time

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from emotion_detection_v2.src.models.efficientnet import create_model

# ============================================================================
# FIX 1: EXACT CLASS MAPPING FROM TRAINING
# ============================================================================
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

EMOTION_COLORS = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 128),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Neutral': (128, 128, 128),
    'Sad': (255, 0, 0),
    'Surprise': (0, 255, 255)
}

# ============================================================================
# FIX 2: EXACT PREPROCESSING PIPELINE
# ============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_face_for_model(face_bgr: np.ndarray) -> torch.Tensor:
    """Preprocess face EXACTLY as in training"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    face_float = face_resized.astype(np.float32) / 255.0
    face_normalized = (face_float - IMAGENET_MEAN) / IMAGENET_STD
    face_chw = np.transpose(face_normalized, (2, 0, 1))
    tensor = torch.from_numpy(face_chw).unsqueeze(0)
    return tensor

# ============================================================================
# FIX 3: BETTER FACE DETECTION
# ============================================================================
class FaceDetector:
    def __init__(self, confidence_threshold=0.5):
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=confidence_threshold
            )
            print("✅ Using MediaPipe face detection")
            self.use_mediapipe = True
        except ImportError:
            print("⚠️  MediaPipe not found, using built-in Haar Cascade detector")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.use_mediapipe = False
            print("✅ Using Haar Cascade face detection")
    
    def detect(self, frame, padding=0.2):
        if self.use_mediapipe:
            return self._detect_mediapipe(frame, padding)
        else:
            return self._detect_haar(frame, padding)
    
    def _detect_mediapipe(self, frame, padding):
        import mediapipe as mp
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                pad_w = int(width * padding)
                pad_h = int(height * padding)
                x = max(0, x - pad_w)
                y = max(0, y - pad_h)
                width = min(w - x, width + 2 * pad_w)
                height = min(h - y, height + 2 * pad_h)
                faces.append((x, y, width, height))
        return faces
    
    def _detect_haar(self, frame, padding):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, width, height) in detected:
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            width = min(w - x, width + 2 * pad_w)
            height = min(h - y, height + 2 * pad_h)
            faces.append((x, y, width, height))
        return faces

# ============================================================================
# FIX 4: TEMPORAL SMOOTHING
# ============================================================================
class TemporalSmoother:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.prob_buffer = deque(maxlen=buffer_size)
    
    def add_prediction(self, probs):
        self.prob_buffer.append(probs)
    
    def get_smoothed_prediction(self):
        if len(self.prob_buffer) == 0:
            return None, 0.0, None
        avg_probs = np.mean(self.prob_buffer, axis=0)
        pred_idx = np.argmax(avg_probs)
        emotion = EMOTION_LABELS[pred_idx]
        confidence = float(avg_probs[pred_idx])
        return emotion, confidence, avg_probs
    
    def reset(self):
        self.prob_buffer.clear()

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================
class RealTimeEmotionDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        print("="*60)
        print("🎭 Initializing Emotion Detector (FIXED VERSION)")
        print("="*60)
        
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded with EMA weights")
        
        self.face_detector = FaceDetector(confidence_threshold=0.5)
        self.smoother = TemporalSmoother(buffer_size=10)
        print("="*60)
    
    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model = create_model()
        
        if 'ema_shadow' in checkpoint:
            state_dict = {}
            for name, param in model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    state_dict[name] = checkpoint['ema_shadow'][name]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @torch.no_grad()
    def predict_emotion(self, face_bgr):
        input_tensor = preprocess_face_for_model(face_bgr).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        self.smoother.add_prediction(probs)
        emotion, confidence, smoothed_probs = self.smoother.get_smoothed_prediction()
        return emotion, confidence, smoothed_probs
    
    def process_frame(self, frame, show_debug=False):
        faces = self.face_detector.detect(frame, padding=0.2)
        
        if len(faces) == 0:
            self.smoother.reset()
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            try:
                emotion, confidence, probs = self.predict_emotion(face_roi)
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw label
                label = f"{emotion}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y-label_h-15), (x+label_w+10, y), color, -1)
                cv2.putText(frame, label, (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if show_debug:
                    face_display = cv2.resize(face_roi, (150, 150))
                    frame[10:160, 10:160] = face_display
                    cv2.rectangle(frame, (10, 10), (160, 160), (0, 255, 0), 2)
                    cv2.putText(frame, "Model Input", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    self._draw_probabilities(frame, probs)
            
            except Exception as e:
                print(f"Error predicting: {e}")
        
        return frame
    
    def _draw_probabilities(self, frame, probs):
        h, w = frame.shape[:2]
        bar_width = 250
        bar_height = 30
        start_x = w - bar_width - 20
        start_y = 20
        
        sorted_emotions = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        
        for i, (idx, prob) in enumerate(sorted_emotions):
            emotion = EMOTION_LABELS[idx]
            color = EMOTION_COLORS[emotion]
            y_pos = start_y + i * (bar_height + 5)
            
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + bar_width, y_pos + bar_height), (40, 40, 40), -1)
            bar_len = int(bar_width * prob)
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + bar_len, y_pos + bar_height), color, -1)
            text = f"{emotion}: {prob:.1%}"
            cv2.putText(frame, text, (start_x + 5, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Real-Time Emotion Detection')
    parser.add_argument('--model', type=str,
                       default='Emotion_Detection_From_Colab/models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (0=default, 1=external)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug visualizations')
    
    args = parser.parse_args()
    
    detector = RealTimeEmotionDetector(args.model, args.device)
    
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("\n✅ Webcam opened!\n")
    print("Controls:")
    print("  'd' - Toggle debug mode")
    print("  'q' or ESC - Quit")
    print("\nPress any key to start...")
    print("="*60 + "\n")
    
    show_debug = args.debug
    fps_buffer = deque(maxlen=30)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame = detector.process_frame(frame, show_debug=show_debug)
        
        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_buffer.append(fps)
        avg_fps = np.mean(fps_buffer)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        hint = f"Debug: {'ON' if show_debug else 'OFF'} | Press 'd' to toggle, 'q' to quit"
        cv2.putText(frame, hint, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Emotion Detection (FIXED)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Demo closed!")

if __name__ == '__main__':
    main()
