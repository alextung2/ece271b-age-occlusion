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
    interval is [bins[k], bins[k+1])
    """
    assert isinstance(age, int) and age >= 0, "age must be a nonnegative int"
    assert len(bins) >= 2, "bins must have at least 2 edges"
    assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)), "bins must be strictly increasing"

    for k in range(len(bins) - 1):
        if bins[k] <= age < bins[k + 1]:
            return k

    # IMPORTANT: do NOT silently map to last class; it hides bugs.
    raise ValueError(f"Age {age} is outside bin edges {bins}")


UTK_RE = re.compile(
    r"^(\d+)_\d+_\d+_\d+\.(?:jpg|jpeg|png)"
    r"(?:\.chip(?:\.(?:jpg|jpeg|png))?)?$",
    re.IGNORECASE,
)


def discover_utkface(
    root: Union[str, Path],
    bins: List[int],
    min_age: int | None = None,
    max_age: int | None = None,
    debug: bool = True,
) -> List[UtkSample]:
    """
    IMPORTANT: returns samples in a deterministic order (sorted by full path).
    This is critical because your split file stores indices into this list.
    """
    root = Path(root)
    assert root.exists(), f"UTKFace root not found: {root}"
    assert root.is_dir(), f"UTKFace root is not a directory: {root}"

    if min_age is not None:
        assert isinstance(min_age, int) and min_age >= 0, "min_age must be a nonnegative int"
    if max_age is not None:
        assert isinstance(max_age, int) and max_age >= 0, "max_age must be a nonnegative int"
    if min_age is not None and max_age is not None:
        assert min_age <= max_age, "min_age must be <= max_age"

    # Deterministic traversal: sort paths
    all_files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower())

    samples: List[UtkSample] = []
    scanned = 0
    matched = 0
    filtered_by_age = 0
    skipped_outside_bins = 0
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

        if min_age is not None and age < min_age:
            filtered_by_age += 1
            continue
        if max_age is not None and age > max_age:
            filtered_by_age += 1
            continue

        try:
            y = age_to_bin(age, bins)
        except ValueError:
            skipped_outside_bins += 1
            continue

        samples.append(UtkSample(path=p, age=age, y=y))

    if len(samples) == 0:
        msg = (
            "No UTKFace images found that match expected filename patterns"
            " (or all were filtered out).\n"
            f"Root: {root}\n"
            f"Total files scanned: {scanned}\n"
            f"Regex-matched filenames: {matched}\n"
            f"Filtered by age: {filtered_by_age}\n"
            f"Skipped outside bins: {skipped_outside_bins}\n"
            f"min_age={min_age}, max_age={max_age}\n"
            f"bins={bins}\n"
            "Expected examples like:\n"
            "  25_0_2_20170116174525125.jpg\n"
            "  25_0_2_20170116174525125.jpg.chip\n"
            "  25_0_2_20170116174525125.jpg.chip.jpg\n"
        )
        if debug and first_few_nonmatches:
            msg += "First few non-matching filenames seen:\n  - " + "\n  - ".join(first_few_nonmatches) + "\n"
        raise AssertionError(msg)

    if debug:
        ages = [s.age for s in samples]
        n_40_49 = sum(40 <= a <= 49 for a in ages)
        n_50_59 = sum(50 <= a <= 59 for a in ages)
        n_60p = sum(a >= 60 for a in ages)

        print(f"[discover_utkface] root={root}")
        print(
            f"[discover_utkface] total_files={scanned} matched={matched} "
            f"filtered_by_age={filtered_by_age} skipped_outside_bins={skipped_outside_bins} "
            f"samples={len(samples)}"
        )
        print(f"[discover_utkface] raw age counts: 40-49={n_40_49}, 50-59={n_50_59}, 60+={n_60p}")
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


def _resize_square(img: np.ndarray, image_size: int) -> np.ndarray:
    """
    Resize to (image_size, image_size) with interpolation chosen by scale.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    # If shrinking, use area; if enlarging, use cubic for better detail
    if image_size < min(h, w):
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (image_size, image_size), interpolation=interp)


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
    img = _resize_square(img, int(image_size))
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
    img = _resize_square(img, int(image_size))
    img = img.astype(np.float32) / 255.0
    return img