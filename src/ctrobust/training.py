from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .geometry import fbp_reconstruct


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def train_epoch(pipeline, loader, radon, angles, optimizer, device) -> EpochStats:
    pipeline.train()
    total_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            recon = fbp_reconstruct(radon(images), angles)
        logits = pipeline(recon)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return EpochStats(total_loss / max(total, 1), correct / max(total, 1))


@torch.no_grad()
def evaluate_clean(pipeline, loader, radon, angles, device) -> dict[str, float]:
    pipeline.eval()
    correct = total = 0
    confs, ents = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = pipeline(fbp_reconstruct(radon(images), angles))
        probs = torch.softmax(logits, dim=1)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
        confs.append(probs.max(1).values.cpu())
        ents.append((-(probs * probs.clamp_min(1e-12).log()).sum(1)).cpu())
    conf = torch.cat(confs) if confs else torch.tensor([])
    ent = torch.cat(ents) if ents else torch.tensor([])
    return {
        "accuracy": correct / max(total, 1),
        "mean_confidence": conf.mean().item() if conf.numel() else float("nan"),
        "mean_entropy": ent.mean().item() if ent.numel() else float("nan"),
    }


def fit(pipeline, train_loader, val_loader, radon, angles, device, *, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(pipeline, train_loader, radon, angles, optimizer, device)
        val_stats = evaluate_clean(pipeline, val_loader, radon, angles, device)
        history.append({"epoch": epoch, "train_loss": train_stats.loss, "train_accuracy": train_stats.accuracy, **{f"val_{k}": v for k, v in val_stats.items()}})
    return history
