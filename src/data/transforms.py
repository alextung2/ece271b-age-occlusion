from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ToTensorCHW:
    """
    Convert HxWxC float32 [0,1] -> torch float32 CxHxW
    """
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        assert isinstance(img, np.ndarray), "img must be a numpy array"
        assert img.ndim == 3 and img.shape[2] == 3, "expected HxWx3"
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return x.float()


@dataclass(frozen=True)
class ToTensorHW:
    """
    Convert HxW float32 [0,1] -> torch float32 HxW
    """
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        assert isinstance(img, np.ndarray), "img must be a numpy array"
        assert img.ndim == 2, "expected HxW"
        return torch.from_numpy(img).contiguous().float()