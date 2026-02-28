from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


SelectMetric = Literal["acc", "macro_f1"]
SchedulerName = Literal["none", "cosine"]


@dataclass
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_metric: float


def _macro_f1_from_confusion(cm: np.ndarray) -> float:
    """
    Macro-F1 computed from confusion matrix.
    cm[i,j] = count true=i predicted=j
    """
    num_classes = cm.shape[0]
    f1s = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = (2 * tp + fp + fn)
        if denom <= 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denom
        f1s.append(float(f1))
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate_acc_and_macro_f1(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Tuple[float, float]:
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        pred = logits.argmax(dim=1)

        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

        y_true = yb.detach().cpu().numpy().astype(np.int64)
        y_pred = pred.detach().cpu().numpy().astype(np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

    acc = float(correct / max(total, 1))
    mf1 = _macro_f1_from_confusion(cm)
    return acc, mf1


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    *,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    use_amp: bool = True,
    select_metric: SelectMetric = "macro_f1",
    scheduler_name: SchedulerName | str = "cosine",
) -> TrainResult:
    model = model.to(device)

    select_metric = str(select_metric).lower().strip()
    if select_metric not in ("acc", "macro_f1"):
        raise ValueError(f"select_metric must be 'acc' or 'macro_f1', got {select_metric!r}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}. Use 'cosine' or 'none'.")

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    # Infer num_classes safely
    try:
        xb0, _ = next(iter(val_loader))
    except StopIteration:
        xb0, _ = next(iter(train_loader))
    with torch.no_grad():
        logits0 = model(xb0.to(device))
    num_classes = int(logits0.shape[1])

    best_val = -1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train ep {ep}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        val_acc, val_mf1 = evaluate_acc_and_macro_f1(model, val_loader, device=device, num_classes=num_classes)
        if scheduler is not None:
            scheduler.step()

        current = val_mf1 if select_metric == "macro_f1" else val_acc
        if current > best_val:
            best_val = float(current)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[val] ep={ep:03d} acc={val_acc:.4f} macro_f1={val_mf1:.4f} best({select_metric})={best_val:.4f}")

    assert best_state is not None
    return TrainResult(best_state_dict=best_state, best_val_metric=float(best_val))