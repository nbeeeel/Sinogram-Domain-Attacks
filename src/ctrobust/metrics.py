from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def confidence_and_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)
    return confidence, entropy


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def prediction_flip_rate(clean_logits: torch.Tensor, adv_logits: torch.Tensor) -> float:
    return (clean_logits.argmax(1) != adv_logits.argmax(1)).float().mean().item()


def attack_success_rate(clean_logits: torch.Tensor, adv_logits: torch.Tensor, labels: torch.Tensor) -> float:
    clean_correct = clean_logits.argmax(1) == labels
    denom = clean_correct.sum().item()
    if denom == 0:
        return float("nan")
    success = clean_correct & (adv_logits.argmax(1) != labels)
    return success.sum().item() / denom


def input_saliency(pipeline, recon: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Absolute input-gradient saliency S(reconstruction)."""
    x = recon.detach().clone().requires_grad_(True)
    logits = pipeline(x)
    if target is None:
        target = logits.argmax(1)
    score = logits.gather(1, target.view(-1, 1)).sum()
    grad = torch.autograd.grad(score, x)[0]
    return grad.abs().detach()


def saliency_instability(clean_saliency: torch.Tensor, adv_saliency: torch.Tensor) -> np.ndarray:
    """Eq. (28): 1 - SSIM, computed sample-wise on normalized saliency maps."""
    clean = clean_saliency.detach().cpu().numpy()
    adv = adv_saliency.detach().cpu().numpy()
    values = []
    for a, b in zip(clean, adv):
        a = a.squeeze()
        b = b.squeeze()
        a = a / (a.max() + 1e-8)
        b = b / (b.max() + 1e-8)
        values.append(1.0 - structural_similarity(a, b, data_range=1.0))
    return np.asarray(values, dtype=np.float64)
