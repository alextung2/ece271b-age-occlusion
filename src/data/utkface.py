from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class UtkSample:
    path: Path
    age: int
    y: int  # class index


def age_to_bin(age: int, bins: List[int]) -> int:
    """
    bins example: [0,10,20,30,40,50,60,200] -> 7 classes
    returns class index in [0, K-1]
    """
    assert isinstance(age, int) and age >= 0, "age must be a nonnegative int"
    assert len(bins) >= 2 and all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)), "bins must be increasing"

    for k in range(len(bins) - 1):
        if bins[k] <= age < bins[k + 1]:
            return k
    return len(bins) - 2  # fallback


UTK_RE = re.compile(
    r"^(\d+)_\d+_\d+_\d+\.(?:jpg|jpeg|png)"
    r"(?:\.chip(?:\.(?:jpg|jpeg|png))?)?$",
    re.IGNORECASE,
)


def discover_utkface(root: Union[str, Path], bins: List[int], debug: bool = True) -> List[UtkSample]:
    root = Path(root)
    assert root.exists(), f"UTKFace root not found: {root}"
    assert root.is_dir(), f"UTKFace root is not a directory: {root}"

    all_files = [p for p in root.rglob("*") if p.is_file()]

    samples: List[UtkSample] = []
    scanned = 0
    matched = 0
    first_few_nonmatches: List[str] = []

    for p in all_files:
        scanned += 1
        name = p.name

        m = UTK_RE.match(name)
        if not m:
            if debug and len(first_few_nonmatches) < 10:
                first_few_nonmatches.append(name)
            continue

        matched += 1
        age = int(m.group(1))
        y = age_to_bin(age, bins)
        samples.append(UtkSample(path=p, age=age, y=y))

    if len(samples) == 0:
        msg = (
            "No UTKFace images found that match expected filename patterns.\n"
            f"Root: {root}\n"
            f"Total files scanned: {scanned}\n"
            f"Regex-matched filenames: {matched}\n"
            "Expected examples like:\n"
            "  25_0_2_20170116174525125.jpg\n"
            "  25_0_2_20170116174525125.jpg.chip\n"
            "  25_0_2_20170116174525125.jpg.chip.jpg\n"
        )
        if debug and first_few_nonmatches:
            msg += "First few non-matching filenames seen:\n  - " + "\n  - ".join(first_few_nonmatches) + "\n"
        raise AssertionError(msg)

    if debug:
        print(f"[discover_utkface] root={root}")
        print(f"[discover_utkface] total_files={scanned} matched={matched} samples={len(samples)}")
        print(f"[discover_utkface] example: {samples[0].path.name} (age={samples[0].age}, y={samples[0].y})")

    return samples


def _imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    """
    Robust image read on Windows for unicode paths + sometimes OneDrive quirks.
    Uses fromfile + imdecode instead of cv2.imread.
    Returns None if cannot read/decode.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def load_image_gray(path: Path, image_size: int) -> np.ndarray:
    """
    Returns float32 grayscale image in [0,1], shape (H,W)
    """
    img = _imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image (maybe OneDrive placeholder or path issue): {path}\n"
            f"Tip: Right-click the UTKFace folder in File Explorer -> OneDrive -> 'Always keep on this device'."
        )
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def load_image_rgb(path: Path, image_size: int) -> np.ndarray:
    """
    Returns float32 RGB image in [0,1], shape (H,W,3)
    """
    img = _imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image (maybe OneDrive placeholder or path issue): {path}\n"
            f"Tip: Right-click the UTKFace folder in File Explorer -> OneDrive -> 'Always keep on this device'."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img