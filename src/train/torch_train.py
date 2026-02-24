from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

@dataclass
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_metric: float

def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> TrainResult:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"train ep {ep}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        val_acc = evaluate_accuracy(model, val_loader, device=device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    return TrainResult(best_state_dict=best_state, best_val_metric=best_val_acc)

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)