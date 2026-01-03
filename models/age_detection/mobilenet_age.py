"""
MobileNetV3 Age Estimation Model (Transfer Learning)
=====================================================
Uses pretrained MobileNetV3-Small backbone with custom regression head.
Much better accuracy than training from scratch.

Advantages:
- Pretrained on ImageNet (learns general features)
- Smaller model size (~2.5M params vs 1.6M but better features)
- Faster convergence (fewer epochs needed)
- Better generalization
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MobileNetV3Age(nn.Module):
    """
    MobileNetV3-Small with custom regression head for age estimation.
    
    Uses pretrained ImageNet weights and replaces classifier with
    age regression head.
    
    Usage:
        model = MobileNetV3Age(pretrained=True)
        age = model(face_tensor)  # (B, 3, 224, 224) -> (B,)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2
    ):
        """
        Initialize MobileNetV3 age model.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers (only train head)
            dropout: Dropout rate in classifier head
        """
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
            logger.info("Loaded pretrained MobileNetV3-Small weights")
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[0].in_features  # 576
        
        # Replace classifier with age regression head (no inplace ops for gradient safety)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, 1)
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


def create_mobilenet_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    weights_path: Optional[str] = None
) -> MobileNetV3Age:
    """
    Factory function to create MobileNetV3 age model.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Only train classifier head
        weights_path: Path to saved model weights
        
    Returns:
        Initialized model
    """
    model = MobileNetV3Age(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    if weights_path:
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    total_params = model.get_num_parameters()
    trainable_params = model.get_num_parameters(trainable_only=True)
    
    logger.info(f"MobileNetV3Age: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


# Quick test
if __name__ == "__main__":
    model = create_mobilenet_model(pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Sample predictions: {y}")
