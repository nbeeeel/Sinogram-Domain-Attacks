from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn

from .geometry import fbp_reconstruct, resample_views


def _loss_gradient(sinogram: torch.Tensor, labels: torch.Tensor, angles: torch.Tensor, pipeline: nn.Module) -> torch.Tensor:
    x = sinogram.detach().requires_grad_(True)
    logits = pipeline(fbp_reconstruct(x, angles))
    loss = nn.functional.cross_entropy(logits, labels)
    return torch.autograd.grad(loss, x)[0]


def fgsm(sinogram: torch.Tensor, labels: torch.Tensor, angles: torch.Tensor, pipeline: nn.Module, eps: float) -> torch.Tensor:
    """Projection-domain FGSM, Eq. (9)."""
    grad = _loss_gradient(sinogram, labels, angles, pipeline)
    return (sinogram.detach() + eps * grad.sign()).detach()


def physics_proxy(
    sinogram: torch.Tensor,
    labels: torch.Tensor,
    angles: torch.Tensor,
    pipeline: nn.Module,
    eps: float,
    *,
    n0: float = 1e5,
    z_alpha: float = 1.96,
    percentile: float = 0.99,
) -> torch.Tensor:
    """Poisson-inspired simulation proxy, Eqs. (12)-(14).

    Processing order follows the manuscript prose: heteroscedastic bound, numerical
    projection-range clipping, then per-view detector-axis centering. The global
    l-infinity bound is re-applied after centering.
    """
    grad = _loss_gradient(sinogram, labels, angles, pipeline)
    g = sinogram.detach()
    delta = eps * grad.sign()

    sigma = torch.exp(g / 2.0) / math.sqrt(n0)
    hetero_bound = z_alpha * sigma
    delta = torch.maximum(torch.minimum(delta, hetero_bound), -hetero_bound)

    flat = g.flatten(1)
    gmax = torch.quantile(flat, percentile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    adv = torch.minimum(torch.maximum(g + delta, torch.zeros_like(g)), gmax)
    delta = adv - g

    delta = delta - delta.mean(dim=2, keepdim=True)  # detector-axis DC removal per view
    delta = delta.clamp(-eps, eps)
    return (g + delta).detach()


def geometry_aware(
    sinogram: torch.Tensor,
    labels: torch.Tensor,
    angles: torch.Tensor,
    pipeline: nn.Module,
    eps: float,
    *,
    streak_indices: Iterable[int] | None = None,
) -> torch.Tensor:
    """Gradient-sensitivity weighting across views and detector bins, Eq. (15).

    The paper specifies alpha(theta_i) and beta(s_j) but not their normalization or
    combination. Here each is max-normalized, their outer product weights the signed
    gradient, and the result is renormalized to the requested l-infinity budget.
    `streak_indices` can explicitly restrict the perturbation to selected views.
    """
    grad = _loss_gradient(sinogram, labels, angles, pipeline)
    # [B,1,D,A] -> view sensitivity [B,1,1,A], detector sensitivity [B,1,D,1]
    alpha = torch.linalg.vector_norm(grad, dim=2, keepdim=True)
    beta = torch.linalg.vector_norm(grad, dim=3, keepdim=True)
    alpha = alpha / alpha.amax(dim=3, keepdim=True).clamp_min(1e-12)
    beta = beta / beta.amax(dim=2, keepdim=True).clamp_min(1e-12)
    weighted = grad.sign() * alpha * beta

    if streak_indices is not None:
        mask = torch.zeros((1, 1, 1, grad.shape[-1]), device=grad.device, dtype=grad.dtype)
        idx = torch.as_tensor(list(streak_indices), device=grad.device, dtype=torch.long)
        mask[..., idx] = 1.0
        weighted = weighted * mask

    scale = weighted.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    delta = eps * weighted / scale
    return (sinogram.detach() + delta).detach()


def frequency_mask(detector: int, band: str, device=None, dtype=torch.float32) -> torch.Tensor:
    """Masks from Eq. (17): low<=0.1*wmax, mid=(0.1,0.5], high>0.5*wmax."""
    freqs = torch.fft.fftfreq(detector, device=device).abs()
    wmax = freqs.max().clamp_min(1e-12)
    r = freqs / wmax
    if band == "low":
        m = r <= 0.1
    elif band == "mid":
        m = (r > 0.1) & (r <= 0.5)
    elif band == "high":
        m = r > 0.5
    elif band == "all":
        m = torch.ones_like(r, dtype=torch.bool)
    else:
        raise ValueError("band must be one of: all, low, mid, high, adaptive")
    return m.to(dtype=dtype).view(1, 1, detector, 1)


def frequency_attack(
    sinogram: torch.Tensor,
    labels: torch.Tensor,
    angles: torch.Tensor,
    pipeline: nn.Module,
    eps: float,
    *,
    band: str = "high",
) -> torch.Tensor:
    """Frequency-domain projection attack, Eqs. (16)-(18), FFT over detector axis."""
    grad = _loss_gradient(sinogram, labels, angles, pipeline)
    spectrum = torch.fft.fft(grad, dim=2)
    if band == "adaptive":
        weight = spectrum.abs()
        weight = weight / weight.amax(dim=2, keepdim=True).clamp_min(1e-12)
        filtered = spectrum * weight
    else:
        filtered = spectrum * frequency_mask(grad.shape[2], band, grad.device, grad.dtype)
    raw = torch.fft.ifft(filtered, dim=2).real
    scale = raw.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    delta = eps * raw / scale
    return (sinogram.detach() + delta).detach()


def adaptive_pgd(
    sinogram: torch.Tensor,
    labels: torch.Tensor,
    angles: torch.Tensor,
    pipeline: nn.Module,
    eps: float,
    *,
    steps: int = 40,
    restarts: int = 3,
    alpha: float | None = None,
) -> torch.Tensor:
    """Adaptive end-to-end PGD, Eqs. (20)-(21), selecting best restart per sample."""
    if alpha is None:
        alpha = 0.1 * eps
    g = sinogram.detach()
    batch = g.shape[0]
    best_adv = g.clone()
    best_loss = torch.full((batch,), -torch.inf, device=g.device)

    for _ in range(restarts):
        delta = torch.empty_like(g).uniform_(-eps, eps)
        adv = (g + delta).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = pipeline(fbp_reconstruct(adv, angles))
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            grad = torch.autograd.grad(losses.sum(), adv)[0]
            with torch.no_grad():
                delta = (adv + alpha * grad.sign() - g).clamp(-eps, eps)
                adv = (g + delta).detach()
        with torch.no_grad():
            final_loss = nn.functional.cross_entropy(pipeline(fbp_reconstruct(adv, angles)), labels, reduction="none")
            better = final_loss > best_loss
            best_loss = torch.where(better, final_loss, best_loss)
            view = better.view(-1, 1, 1, 1)
            best_adv = torch.where(view, adv, best_adv)
    return best_adv.detach()


def optimize_upp(
    loader,
    radon,
    angles: torch.Tensor,
    pipeline: nn.Module,
    eps: float,
    *,
    epochs: int = 20,
    step_size: float = 0.01,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Projected sign-gradient ascent universal perturbation, Eq. (19)."""
    pipeline.eval()
    delta = None
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                g = radon(images)
            if delta is None:
                delta = torch.zeros((1, 1, g.shape[2], g.shape[3]), device=device, dtype=g.dtype)
            delta = delta.detach().requires_grad_(True)
            adv = g + delta
            loss = nn.functional.cross_entropy(pipeline(fbp_reconstruct(adv, angles)), labels)
            grad = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                delta = (delta + step_size * grad.sign()).clamp(-eps, eps)
    if delta is None:
        raise ValueError("UPP optimization loader is empty")
    return delta.detach()


def apply_upp(sinogram: torch.Tensor, upp: torch.Tensor) -> torch.Tensor:
    """Apply a frozen UPP, angularly resampling it when view counts differ."""
    delta = resample_views(upp, sinogram.shape[-1])
    return (sinogram + delta).detach()
