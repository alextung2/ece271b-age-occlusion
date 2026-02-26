from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

import numpy as np


@dataclass(frozen=True)
class Split:
    train: List[int]
    val: List[int]
    test: List[int]


def make_split(n: int, train_frac: float, val_frac: float, seed: int) -> Split:
    """
    Create a random (train/val/test) split of indices [0, n).
    Uses a single shuffled permutation and slices it.
    """
    assert isinstance(n, int) and n > 0, "n must be a positive int"
    assert 0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0, "fractions must be in (0,1)"
    assert train_frac + val_frac < 1.0, "train+val must be < 1"

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # Use floor so we never over-allocate due to rounding
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    # Ensure we don't accidentally consume everything
    assert 0 <= n_train <= n, "bad n_train"
    assert 0 <= n_val <= n, "bad n_val"
    assert n_train + n_val < n, "train+val consumed all samples; need some test samples"

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()
    test_idx = idx[n_train + n_val:].tolist()

    return Split(train=train_idx, val=val_idx, test=test_idx)


def save_split(split: Split, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"train": split.train, "val": split.val, "test": split.test}, f)


def load_split(path: str | Path) -> Split:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return Split(train=list(obj["train"]), val=list(obj["val"]), test=list(obj["test"]))


def validate_split(split: Split, n: int) -> None:
    """
    Raises AssertionError if split is invalid.
    Checks:
      - all indices are ints
      - all indices are within [0, n)
      - no duplicates across splits
      - covers all indices exactly once
    """
    all_lists = [split.train, split.val, split.test]
    all_idx = [i for lst in all_lists for i in lst]

    assert all(isinstance(i, int) for i in all_idx), "split contains non-int indices"
    assert all(0 <= i < n for i in all_idx), "split contains out-of-range indices"

    uniq = set(all_idx)
    assert len(uniq) == len(all_idx), "duplicate index found across train/val/test"

    # If you expect a full partition of 0..n-1, enforce it:
    assert len(uniq) == n, f"split does not cover all indices exactly once (got {len(uniq)} of {n})"