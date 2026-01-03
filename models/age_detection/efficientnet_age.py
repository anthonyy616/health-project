"""
EfficientNet Age Estimation Model (Transfer Learning)
======================================================
Uses pretrained EfficientNet-B0 backbone with custom regression head.
Better feature extraction than MobileNetV3 with reasonable inference time.

EfficientNet-B0 specs:
- Parameters: ~5.3M
- Top-1 Accuracy: 77.1% on ImageNet
- Inference: ~50ms on CPU (vs ~20ms for MobileNetV3)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EfficientNetAge(nn.Module):
    """
    EfficientNet-B0 with custom regression head for age estimation.
    
    Uses pretrained ImageNet weights and replaces classifier with
    age regression head optimized for the task.
    
    Usage:
        model = EfficientNetAge(pretrained=True)
        age = model(face_tensor)  # (B, 3, 224, 224) -> (B,)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet age model.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers (only train head)
            dropout: Dropout rate in classifier head
        """
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
            logger.info("Loaded pretrained EfficientNet-B0 weights")
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features  # 1280
        
        # Replace classifier with age regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except classifier"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Backbone frozen - only classifier will train")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen - all layers will train")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            
        Returns:
            Age predictions (B,)
        """
        out = self.backbone(x)
        return out.squeeze(-1)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return number of parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_efficientnet_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    weights_path: Optional[str] = None
) -> EfficientNetAge:
    """
    Factory function to create EfficientNet age model.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Only train classifier head
        weights_path: Path to saved model weights
        
    Returns:
        Initialized model
    """
    model = EfficientNetAge(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    if weights_path:
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    total_params = model.get_num_parameters()
    trainable_params = model.get_num_parameters(trainable_only=True)
    
    logger.info(f"EfficientNetAge: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


# Quick test
if __name__ == "__main__":
    model = create_efficientnet_model(pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Sample predictions: {y}")
    print(f"Total parameters: {model.get_num_parameters():,}")
