from __future__ import annotations
from typing import Literal
import numpy as np

FillMode = Literal["mean", "zero", "noise"]

def occlude_region(img: np.ndarray, region: str, fill: FillMode = "mean") -> np.ndarray:
    """
    img: grayscale float32 in [0,1], shape (H,W)
    region: "none" | "eyes" | "mouth" | "center"
    fill: "mean" | "zero" | "noise"
    """
    assert isinstance(img, np.ndarray), "img must be a numpy array"
    assert img.ndim == 2, f"expected grayscale HxW, got shape {img.shape}"
    H, W = img.shape
    assert H > 0 and W > 0, "empty image?"

    out = img.copy()

    if region == "none":
        return out

    # region boxes are fractions for roughly aligned faces
    if region == "eyes":
        x0, x1 = int(0.15 * W), int(0.85 * W)
        y0, y1 = int(0.20 * H), int(0.45 * H)
    elif region == "mouth":
        x0, x1 = int(0.20 * W), int(0.80 * W)
        y0, y1 = int(0.65 * H), int(0.85 * H)
    elif region == "center":
        x0, x1 = int(0.35 * W), int(0.65 * W)
        y0, y1 = int(0.35 * H), int(0.65 * H)
    else:
        raise ValueError(f"unknown region: {region}")

    # clamp (just in case)
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        return out  # degenerate box -> no-op

    patch = out[y0:y1, x0:x1]

    if fill == "mean":
        patch[:] = float(out.mean())
    elif fill == "zero":
        patch[:] = 0.0
    elif fill == "noise":
        mu = float(out.mean())
        sigma = float(out.std() + 1e-6)
        noise = np.random.normal(mu, sigma, size=patch.shape).astype(out.dtype, copy=False)
        patch[:] = np.clip(noise, 0.0, 1.0)
    else:
        raise ValueError(f"unknown fill mode: {fill}")

    return out