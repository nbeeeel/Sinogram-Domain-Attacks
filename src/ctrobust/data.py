from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def preprocess_hu(array: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Paper preprocessing: clip [-1000, 400] HU, bilinear resize, normalize [0,1]."""
    x = np.asarray(array, dtype=np.float32)
    x = np.clip(x, -1000.0, 400.0)
    x = transform.resize(
        x,
        (image_size, image_size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return (x + 1000.0) / 1400.0


def preprocess_normalized(array: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Resize an already-normalized [0,1] slice without per-image min/max remapping."""
    x = np.asarray(array)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / float(info.max)
    else:
        x = x.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return transform.resize(
        x,
        (image_size, image_size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return io.imread(path, as_gray=True)


class FolderSliceDataset(Dataset):
    """Compatibility loader for split/{benign,malignant}/image files.

    Use `input_mode='hu'` for .npy arrays containing HU values. Use
    `input_mode='normalized'` for PNG/JPEG/TIFF data already mapped to [0,1].
    """

    def __init__(self, root_dir: str | Path, image_size: int = 256, input_mode: str = "normalized"):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.input_mode = input_mode
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(("benign", "malignant")):
            folder = self.root_dir / class_name
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))
        if input_mode not in {"normalized", "hu"}:
            raise ValueError("input_mode must be 'normalized' or 'hu'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = _load_array(path)
        x = preprocess_hu(x, self.image_size) if self.input_mode == "hu" else preprocess_normalized(x, self.image_size)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(label, dtype=torch.long)


class ManifestSliceDataset(Dataset):
    """Dataset backed by a CSV manifest with path,label,patient_id,split columns."""

    def __init__(self, manifest: str | Path, split: str, image_size: int = 256, input_mode: str = "hu"):
        self.manifest = Path(manifest)
        self.image_size = image_size
        self.input_mode = input_mode
        self.rows = []
        with self.manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") == split:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        x = _load_array(path)
        x = preprocess_hu(x, self.image_size) if self.input_mode == "hu" else preprocess_normalized(x, self.image_size)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(int(row["label"]), dtype=torch.long)
