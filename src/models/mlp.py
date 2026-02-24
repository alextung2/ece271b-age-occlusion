from __future__ import annotations
from typing import List, Literal

import torch
import torch.nn as nn

ActivationName = Literal["relu", "gelu", "tanh", "leaky_relu"]


def _make_activation(name: ActivationName) -> nn.Module:
    """Factory for activation functions."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    Simple MLP classifier.

    IMPORTANT CONTRACT (do not change):
      input:  x of shape (N, D)
      output: logits of shape (N, K)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: List[int],
        dropout: float,
        activation: ActivationName = "relu",
        use_batchnorm: bool = False,
    ):
        super().__init__()

        # ---- light validation (safe guard) ----
        assert isinstance(input_dim, int) and input_dim > 0
        assert isinstance(num_classes, int) and num_classes > 0
        assert isinstance(hidden_sizes, list)
        assert 0.0 <= float(dropout) < 1.0

        layers: List[nn.Module] = []
        prev = input_dim
        # NOTE: do NOT pre-create activation (must instantiate per layer)

        # ============================================================
        # ✨ EDIT HERE (Alexander Wang)
        #
        # Your goal:
        #   Improve the MLP architecture while keeping shapes the same.
        #
        # SAFE things to try:
        #   • turn on BatchNorm
        #   • move dropout before/after activation
        #   • add LayerNorm instead of BatchNorm
        #   • add one extra hidden layer
        #
        # ❗ DO NOT change:
        #   • input/output shapes
        #   • forward()
        #   • final head layer
        #
        # ------------------------------------------------------------
        # ✅ EXAMPLE (baseline block — already implemented)
        #
        # for h in hidden_sizes:
        #     layers.append(nn.Linear(prev, h))
        #
        #     if use_batchnorm:
        #         layers.append(nn.BatchNorm1d(h))
        #
        #     layers.append(act)
        #
        #     if dropout > 0:
        #         layers.append(nn.Dropout(dropout))
        #
        #     prev = h
        #
        # ------------------------------------------------------------
        # 🧪 EXAMPLE MODIFICATION (LayerNorm version)
        #
        # Replace BatchNorm block with:
        #
        #     layers.append(nn.LayerNorm(h))
        #
        # ------------------------------------------------------------
        # 🧪 EXAMPLE MODIFICATION (extra hidden layer)
        #
        # After the loop you could add:
        #
        #     layers.append(nn.Linear(prev, prev))
        #     layers.append(act)
        #
        # ============================================================

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))

            # ----- optional normalization -----
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))

            # ----- activation -----
            # IMPORTANT FIX: create a NEW activation each layer
            layers.append(_make_activation(activation))

            # ----- dropout -----
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = h

        # ============================================================
        # OPTIONAL EXTRA LAYERS (safe place to experiment)
        #
        # Example:
        # layers.append(nn.Linear(prev, prev))
        # layers.append(act)
        # ============================================================

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D)
        returns: logits (N, K)
        """
        z = self.backbone(x)
        logits = self.head(z)
        return logits


def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_sizes: List[int],
    dropout: float,
    activation: ActivationName = "relu",
    use_batchnorm: bool = False,
) -> MLP:
    """
    Factory function used by training scripts.

    Training code should call THIS instead of directly constructing MLP.
    """
    return MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        use_batchnorm=use_batchnorm,
    )