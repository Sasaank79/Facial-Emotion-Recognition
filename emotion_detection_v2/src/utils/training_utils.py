"""
Utility functions for training: EMA, metrics, checkpointing, etc.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.
    Helps stabilize training and often improves final performance.
    
    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower update)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> str:
    """Generate detailed classification report."""
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize by row (true label)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    Prevents overconfidence and improves generalization.
    
    Reference: "Rethinking the Inception Architecture" (Szegedy et al., 2016)
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (batch_size, num_classes)
            target: Target labels (batch_size,) or one-hot (batch_size, num_classes)
            
        Returns:
            Smoothed cross entropy loss
        """
        n_class = pred.size(1)
        
        # Convert to one-hot if needed
        if target.dim() == 1:
            target = torch.nn.functional.one_hot(target, num_classes=n_class).float()
        
        # Apply label smoothing
        target = target * (1 - self.smoothing) + self.smoothing / n_class
        
        # Compute cross entropy
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(target * log_prob).sum(dim=1).mean()
        
        return loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    ema: ExponentialMovingAverage = None,
    path: str = 'checkpoint.pth'
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }
    
    if ema:
        checkpoint['ema_shadow'] = ema.shadow
    
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    ema: ExponentialMovingAverage = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if ema and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0)
    }


if __name__ == '__main__':
    print("Testing utils...")
    
    # Test EMA
    model = nn.Linear(10, 5)
    ema = ExponentialMovingAverage(model, decay=0.999)
    print("✅ EMA initialized")
    
    # Test label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(4, 7)
    target = torch.tensor([0, 1, 2, 3])
    loss = criterion(pred, target)
    print(f"✅ Label smoothing loss: {loss.item():.4f}")
    
    # Test metrics
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    print(f"✅ Metrics: {metrics}")
    
    print("All tests passed!")
