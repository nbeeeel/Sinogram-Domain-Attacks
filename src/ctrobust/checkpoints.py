from __future__ import annotations

from pathlib import Path

import torch

from .models import CTClassifier3, CTClassifier5, DiagnosticPipeline


def build_pipeline(arch: str = "cnn3", use_rsdf: bool = False) -> DiagnosticPipeline:
    if arch == "cnn3":
        backbone = CTClassifier3()
    elif arch == "cnn5":
        backbone = CTClassifier5()
    else:
        raise ValueError("arch must be 'cnn3' or 'cnn5'")
    return DiagnosticPipeline(backbone, use_rsdf=use_rsdf)


def save_checkpoint(path, pipeline, *, arch: str, use_rsdf: bool, n_views: int, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": pipeline.state_dict(),
        "arch": arch,
        "use_rsdf": use_rsdf,
        "n_views": n_views,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pipeline = build_pipeline(ckpt.get("arch", "cnn3"), ckpt.get("use_rsdf", False)).to(device)
    pipeline.load_state_dict(ckpt["state_dict"])
    pipeline.eval()
    return pipeline, ckpt
