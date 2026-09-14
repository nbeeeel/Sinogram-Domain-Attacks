"""Projection-domain adversarial robustness utilities for synthetic parallel-beam CT."""

from .geometry import DifferentiableRadon, fbp_reconstruct, make_angles
from .models import CTClassifier3, CTClassifier5, DiagnosticPipeline, RSDF

__all__ = [
    "DifferentiableRadon",
    "fbp_reconstruct",
    "make_angles",
    "CTClassifier3",
    "CTClassifier5",
    "DiagnosticPipeline",
    "RSDF",
]

__version__ = "0.1.0"
