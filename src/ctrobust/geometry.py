from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_angles(n_views: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Equiangular samples on [0, pi), matching the manuscript definition."""
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    return torch.arange(n_views, device=device, dtype=dtype) * (math.pi / n_views)


class DifferentiableRadon(nn.Module):
    """Numerical parallel-beam Radon transform using differentiable grid sampling.

    Input:  [B, 1, H, W] (square images expected)
    Output: [B, 1, D, A], where D=H and A=len(angles)

    This intentionally follows the rotation-and-sum implementation in the supplied
    research scripts. It is a controlled numerical forward model, not scanner raw-data IO.
    """

    def __init__(self, angles: torch.Tensor):
        super().__init__()
        angles = angles.detach().clone().float()
        c, s = torch.cos(angles), torch.sin(angles)
        mats = torch.zeros((len(angles), 2, 3), dtype=torch.float32)
        mats[:, 0, 0] = c
        mats[:, 0, 1] = -s
        mats[:, 1, 0] = s
        mats[:, 1, 1] = c
        self.register_buffer("angles", angles)
        self.register_buffer("affines", mats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected x with shape [B,1,H,W]")
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("Current numerical Radon implementation expects square images")
        batch = x.shape[0]
        projections = []
        for mat in self.affines:
            theta = mat.unsqueeze(0).expand(batch, -1, -1)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            rotated = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
            projections.append(rotated.sum(dim=2))
        return torch.stack(projections, dim=-1)


def fbp_reconstruct(sinogram: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Differentiable ramp-filtered backprojection from [B,1,D,A] to [B,1,D,D].

    The discrete normalization uses the mean over views (`/ A`), preserving the
    normalization used in the supplied scripts. No apodization window is applied.
    """
    if sinogram.ndim != 4 or sinogram.shape[1] != 1:
        raise ValueError("Expected sinogram with shape [B,1,D,A]")
    batch, _, detector, n_views = sinogram.shape
    if len(angles) != n_views:
        raise ValueError(f"len(angles)={len(angles)} but sinogram has {n_views} views")

    ramp = torch.abs(torch.fft.fftfreq(detector, d=1.0, device=sinogram.device))
    ramp = ramp.view(1, 1, detector, 1).to(sinogram.dtype)
    filtered = torch.fft.ifft(torch.fft.fft(sinogram, dim=2) * ramp, dim=2).real

    recon = torch.zeros((batch, 1, detector, detector), device=sinogram.device, dtype=sinogram.dtype)
    c, s = torch.cos(angles.to(sinogram.device)), torch.sin(angles.to(sinogram.device))
    for i in range(n_views):
        zero = torch.zeros((), device=sinogram.device, dtype=sinogram.dtype)
        row0 = torch.stack((c[i].to(sinogram.dtype), -s[i].to(sinogram.dtype), zero))
        row1 = torch.stack((s[i].to(sinogram.dtype), c[i].to(sinogram.dtype), zero))
        mat = torch.stack((row0, row1)).unsqueeze(0).expand(batch, -1, -1)
        projection = filtered[:, :, :, i].unsqueeze(-1).expand(-1, -1, -1, detector)
        grid = F.affine_grid(mat, projection.size(), align_corners=False)
        recon = recon + F.grid_sample(projection, grid, align_corners=False, padding_mode="zeros")
    return recon / n_views


def resample_views(tensor: torch.Tensor, n_views: int) -> torch.Tensor:
    """Resample the angular axis of [B,C,D,A] data to a new view count.

    Used for UPP cross-view transfer because the manuscript states that a fixed
    180-view perturbation is transferred to other view counts but does not specify
    the discrete resampling rule. Linear angular resampling is therefore an explicit
    implementation choice and is documented as such.
    """
    if tensor.ndim != 4:
        raise ValueError("Expected [B,C,D,A]")
    if tensor.shape[-1] == n_views:
        return tensor
    b, c, d, a = tensor.shape
    x = tensor.reshape(b * c, 1, d, a)
    y = F.interpolate(x, size=(d, n_views), mode="bilinear", align_corners=False)
    return y.reshape(b, c, d, n_views)
