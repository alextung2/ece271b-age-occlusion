from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .metrics import overall_accuracy, macro_f1


@dataclass
class EvalRecord:
    method: str
    occlusion: str
    acc: float
    macro_f1: float


def compute_robustness_drop(clean_value: float, occ_value: float) -> float:
    """
    Absolute drop: clean - occluded. Positive means performance got worse under occlusion.
    """
    clean_value = float(clean_value)
    occ_value = float(occ_value)
    return clean_value - occ_value


def compute_robustness_ratio(clean_value: float, occ_value: float, eps: float = 1e-12) -> float:
    """
    Relative drop ratio: (clean - occluded) / clean.
    Useful when comparing drops across metrics with different scales.
    """
    clean_value = float(clean_value)
    occ_value = float(occ_value)
    return (clean_value - occ_value) / max(clean_value, eps)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str,
    occlusion: str,
    num_classes: Optional[int] = None,
) -> EvalRecord:
    """
    Convenience wrapper to generate one EvalRecord from predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = overall_accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred, num_classes=num_classes)
    return EvalRecord(method=method, occlusion=occlusion, acc=acc, macro_f1=mf1)