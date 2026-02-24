from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_cnn(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a CNN classifier.

    Supported:
      - resnet18

    Notes:
      - If pretrained=True, your input pipeline should use ImageNet normalization:
          mean = (0.485, 0.456, 0.406)
          std  = (0.229, 0.224, 0.225)
        Otherwise pretrained weights are less effective.
    """
    if backbone == "resnet18":
        model = tvm.resnet18(
            weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


@torch.no_grad()
def extract_cnn_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return a penultimate-layer embedding for ResNet-style backbones.

    For resnet18, this returns the pooled feature vector right before the final fc:
      (N, 512)

    x: (N, 3, H, W)
    """
    model.eval()

    # This works for torchvision ResNet models
    if not hasattr(model, "conv1") or not hasattr(model, "fc"):
        raise ValueError("extract_cnn_embedding currently supports torchvision ResNet-style models only.")

    # Forward through everything except the final fc
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)              # (N, 512, 1, 1) for resnet18
    x = torch.flatten(x, 1)           # (N, 512)
    return x