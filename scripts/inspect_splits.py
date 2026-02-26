from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.config import Config
from src.data.utkface import discover_utkface
from src.data.splits import load_split


def main():
    cfg = Config.load("configs/default.yaml")

    root = cfg.get("data.utkface_root")
    bins = cfg.get("labels.bins")
    min_age = cfg.get("labels.min_age", None)
    max_age = cfg.get("labels.max_age", None)

    samples = discover_utkface(
        root=root,
        bins=bins,
        min_age=min_age,
        max_age=max_age,
        debug=False,
    )

    split_path = Path("outputs/splits/utkface_split.json")
    split = load_split(split_path)

    print("=== SPLIT SIZES ===")
    print(f"Train: {len(split.train)}")
    print(f"Val:   {len(split.val)}")
    print(f"Test:  {len(split.test)}")
    print(f"Total: {len(samples)}")
    print()

    def bin_label(k: int) -> str:
        lo = bins[k]
        hi = bins[k + 1]
        if k == len(bins) - 2:
            return f"{lo}+"
        return f"{lo}-{hi-1}"

    def summarize(name: str, indices):
        ys = [samples[i].y for i in indices]
        ages = [samples[i].age for i in indices]
        c = Counter(ys)
        total = len(indices)

        print(f"=== {name} ===")
        print(f"N = {total}")
        for k in range(len(bins) - 1):
            cnt = c.get(k, 0)
            pct = 100.0 * cnt / total if total > 0 else 0.0
            print(f"  class {k} ({bin_label(k)}): {cnt} ({pct:.1f}%)")
        print(f"Age range: {min(ages)}–{max(ages)}")
        print()

    summarize("TRAIN", split.train)
    summarize("VAL", split.val)
    summarize("TEST", split.test)


if __name__ == "__main__":
    main()