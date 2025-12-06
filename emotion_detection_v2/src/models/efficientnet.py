"""
EfficientNetV2-based emotion recognition model.
Uses timm library for pretrained weights and efficient training.
"""

import torch
import torch.nn as nn
import timm


class EmotionEfficientNet(nn.Module):
    """
    EfficientNetV2 model for facial emotion recognition.
    
    Features:
    - Pretrained on ImageNet for better transfer learning
    - Dropout for regularization
    - Clean, simple architecture
    """
    
    def __init__(
        self,
        model_name: str = 'tf_efficientnetv2_s',
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained EfficientNetV2 from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Use global average pooling
        )
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes)
        )
        
        # Initialize classifier
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_params_groups(self, lr: float, backbone_lr_mult: float = 0.1):
        """
        Get parameter groups for differential learning rates.
        Backbone gets lower LR, classifier gets higher LR.
        
        Args:
            lr: Base learning rate
            backbone_lr_mult: Multiplier for backbone LR
            
        Returns:
            List of parameter groups
        """
        return [
            {'params': self.backbone.parameters(), 'lr': lr * backbone_lr_mult},
            {'params': self.classifier.parameters(), 'lr': lr}
        ]


def create_model(
    model_name: str = 'tf_efficientnetv2_s',
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.3
) -> EmotionEfficientNet:
    """
    Factory function to create emotion recognition model.
    
    Available models:
    - tf_efficientnetv2_s: Small, fast, 24M params
    - tf_efficientnetv2_m: Medium, 54M params
    - tf_efficientnetv2_l: Large, 120M params
    - convnext_tiny: ConvNeXt Tiny, 28M params
    - convnext_small: ConvNeXt Small, 50M params
    
    Args:
        model_name: Model architecture name
        num_classes: Number of emotion classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        
    Returns:
        Initialized model
    """
    return EmotionEfficientNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test model creation
    model = create_model()
    print(f"Model: {model.__class__.__name__}")
    print(f"Num features: {model.num_features}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 7), f"Expected (2, 7), got {out.shape}"
    print("✅ Model test passed!")
