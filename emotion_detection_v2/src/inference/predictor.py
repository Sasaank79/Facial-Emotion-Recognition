"""
Clean inference module for production deployment.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2

from src.models.efficientnet import create_model
from src.data.transforms import get_val_transforms, AlbumentationsWrapper


EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}


class EmotionPredictor:
    """
    Production-ready emotion predictor.
    
    Usage:
        predictor = EmotionPredictor('models/best_model.pth')
        emotion, confidence, probabilities = predictor.predict(face_image)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        use_ema: bool = True
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda'
            use_ema: Whether to use EMA weights (if available)
        """
        self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path, use_ema)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        self.transform = AlbumentationsWrapper(get_val_transforms(224))
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   EMA: {use_ema}")
    
    def _load_model(self, path: str, use_ema: bool) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model
        model = create_model(
            model_name='tf_efficientnetv2_s',
            num_classes=7,
            pretrained=False
        )
        
        # Load weights (EMA if available)
        if use_ema and 'ema_shadow' in checkpoint:
            print("Loading EMA weights...")
            state_dict = {}
            for name, param in model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    state_dict[name] = checkpoint['ema_shadow'][name]
            model.load_state_dict(state_dict, strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume direct state dict
            model.load_state_dict(checkpoint)
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_all_probs: bool = True
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Predict emotion from face image.
        
        Args:
            image: Face image as numpy array (RGB) or PIL Image
            return_all_probs: Whether to return probabilities for all classes
            
        Returns:
            (emotion_name, confidence, probabilities_dict)
            
        Example:
            >>> emotion, conf, probs = predictor.predict(face_img)
            >>> print(f"{emotion}: {conf:.2%}")
            >>> for emotion, prob in probs.items():
            >>>     print(f"  {emotion}: {prob:.2%}")
        """
        # Input validation
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            raise ValueError("Invalid input image")
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # Ensure RGB
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(image.astype('uint8'))
        
        # Transform
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error transforming image: {e}")
        
        # Predict
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        
        # All probabilities
        all_probs = None
        if return_all_probs:
            all_probs = {
                EMOTION_LABELS[i]: float(probs[i])
                for i in range(len(EMOTION_LABELS))
            }
        
        return emotion, confidence, all_probs
    
    def predict_batch(
        self,
        images: list,
        batch_size: int = 32
    ) -> list:
        """
        Predict emotions for a batch of images.
        
        Args:
            images: List of face images
            batch_size: Batch size for processing
            
        Returns:
            List of (emotion, confidence, probabilities) tuples
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Transform batch
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype('uint8'))
                tensors.append(self.transform(img))
            
            input_batch = torch.stack(tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Parse results
            for prob in probs:
                pred_idx = np.argmax(prob)
                emotion = EMOTION_LABELS[pred_idx]
                confidence = float(prob[pred_idx])
                all_probs = {
                    EMOTION_LABELS[j]: float(prob[j])
                    for j in range(len(EMOTION_LABELS))
                }
                results.append((emotion, confidence, all_probs))
        
        return results


class WebcamEmotionDetector:
    """
    Real-time emotion detection from webcam with face detection.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        face_cascade_path: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to trained model
            device: 'cpu' or 'cuda'
            face_cascade_path: Path to Haar Cascade XML (optional)
        """
        self.predictor = EmotionPredictor(model_path, device)
        
        # Load face detector
        if face_cascade_path is None:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load face cascade from {face_cascade_path}")
    
    def detect_faces(self, frame: np.ndarray):
        """Detect faces in frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Process video frame: detect faces and predict emotions.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            (annotated_frame, list of (x, y, w, h, emotion, confidence))
        """
        faces = self.detect_faces(frame)
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            #Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            try:
                # Predict emotion
                emotion, confidence, _ = self.predictor.predict(face_rgb, return_all_probs=False)
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{emotion}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                results.append((x, y, w, h, emotion, confidence))
            except Exception as e:
                print(f"Error predicting emotion: {e}")
        
        return frame, results


if __name__ == '__main__':
    print("Testing EmotionPredictor...")
    
    # Create dummy model for testing
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # This will fail without a trained model, but shows usage
    try:
        predictor = EmotionPredictor('models/best_model.pth')
        
        # Test with dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emotion, conf, probs = predictor.predict(dummy_img)
        print(f"Prediction: {emotion} ({conf:.2%})")
        print(f"All probabilities: {probs}")
    except FileNotFoundError:
        print("Model not found (expected for testing)")
    
    print("✅ Inference module loaded successfully")
