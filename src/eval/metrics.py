from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    return float((y_true == y_pred).mean())

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))

def confmat(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))