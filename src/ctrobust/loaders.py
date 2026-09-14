from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .data import FolderSliceDataset, ManifestSliceDataset


def make_dataset(*, data_root=None, manifest=None, split="train", image_size=256, input_mode="normalized"):
    if manifest:
        split_name = "valid" if split == "validation" else split
        try:
            return ManifestSliceDataset(manifest, split_name, image_size=image_size, input_mode=input_mode)
        except ValueError:
            if split_name == "valid":
                return ManifestSliceDataset(manifest, "validation", image_size=image_size, input_mode=input_mode)
            raise
    if not data_root:
        raise ValueError("Provide either data_root or manifest")
    root = Path(data_root)
    candidates = [split]
    if split in {"valid", "validation"}:
        candidates = ["valid", "validation", "val"]
    for name in candidates:
        ds = FolderSliceDataset(root / name, image_size=image_size, input_mode=input_mode)
        if len(ds):
            return ds
    raise ValueError(f"No samples found for split={split!r} under {root}")


def make_loader(dataset, *, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
