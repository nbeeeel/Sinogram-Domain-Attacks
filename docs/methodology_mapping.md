# Methodology-to-code mapping

This document separates three things: **explicit manuscript specification**, **behavior visible in the supplied scripts**, and **implementation choices required to connect gaps**.

| Paper component | Repository implementation | Status / note |
|---|---|---|
| Parallel-beam Radon transform | `ctrobust.geometry.DifferentiableRadon` | Directly consolidated from supplied scripts |
| Eqiangular views on `[0, pi)` | `ctrobust.geometry.make_angles` | Paper-faithful correction to legacy `linspace(0,pi,N)` endpoint duplication |
| Ramp-filtered FBP, no apodization | `ctrobust.geometry.fbp_reconstruct` | Consolidated from supplied scripts; retains their discrete `/N_views` normalization |
| HU `[-1000,400]`, 256x256, `[0,1]` | `ctrobust.data.preprocess_hu` | Paper specification takes precedence over legacy per-image min/max normalization |
| Patient-wise 70/15/15 split | `scripts/build_manifest.py` | Requires already-selected nodule-centred slices; raw DICOM slice selection is not fully specified by paper |
| 3-block CNN | `CTClassifier3` | Exact architecture and exact reported 23,426 parameters |
| RSDF | `RSDF` | Exact 1->16->1 residual architecture; final residual conv initialized with std `1e-3`; 305 parameters |
| 5-block reference CNN | `CTClassifier5` | Layer channels inferred as 1->16->32->64->128->256 because this exactly reproduces the reported 392,834 params; pooling placement is not explicitly tabulated |
| Clean-only Adam training | `ctrobust.training.fit` | Cross-entropy, lr `1e-3`, batch 32, 20 epochs; no adversarial training |
| FGSM Eq. 9 | `attacks.fgsm` | Direct |
| Physics proxy Eqs. 12-14 | `attacks.physics_proxy` | Heteroscedastic Poisson bound, p99 range clip, per-view detector DC centering; global epsilon reapplied after centering |
| Geometry-aware Eq. 15 | `attacks.geometry_aware` | Paper gives view/detector L2 sensitivities but not exact normalization/combination. Repo max-normalizes both, multiplies them, then renormalizes to epsilon |
| ROI streak mode | `geometry_aware(..., streak_indices=...)` | Explicit selected view indices supported; automatic lesion-ROI corridor construction omitted because ROI selection is not specified in the available method/code |
| Frequency Eqs. 16-18 | `attacks.frequency_attack` | FFT on detector axis; low/mid/high thresholds exactly `0.1*wmax`, `0.5*wmax`; adaptive spectrum supported; final perturbation L-inf renormalized |
| Primary frequency-family choice | CLI `--frequency-band` | Manuscript lists fixed or adaptive variants but does not uniquely identify which one produced the primary family-table value; CLI defaults to `adaptive` and exposes the choice |
| UPP Eq. 19 | `attacks.optimize_upp` | Batch 16, 20 epochs, sign-gradient ascent, step 0.01, L-inf projection, frozen for evaluation |
| UPP cross-view transfer | `geometry.resample_views` + `attacks.apply_upp` | Paper does not specify how a 180-view discrete tensor is mapped to 120/90/60/30 views; repository uses documented linear angular resampling |
| Adaptive PGD Eqs. 20-21 | `attacks.adaptive_pgd` | Uniform random start, alpha `0.1*eps`, 10/20/40 steps, 3 restarts, highest-loss restart retained per sample |
| Accuracy drop Eq. 22 | `evaluation.evaluate_attack` | Direct |
| Confidence / entropy Eqs. 23-25 | `metrics.confidence_and_entropy` | Direct |
| Flip rate Eq. 26 | `evaluation.evaluate_attack` | Direct |
| Clean-correct ASR Eq. 27 | `evaluation.evaluate_attack` | Direct |
| Saliency instability Eq. 28 | `metrics.saliency_instability` | Uses `1 - SSIM`, replacing the L2 saliency drift used in one exploratory fragment because the paper is explicit |

## Supplied-code reconciliation

The six fragments are retained verbatim under `legacy/`. They contain overlapping generations of the same experiments, with different epsilon values, view counts, plotting code, and in later versions PGD instead of FGSM. The canonical package removes that duplication and centralizes common CT geometry, model, attack, and metric code.

Notable corrections made during consolidation:

1. **Angular endpoint:** legacy scripts include both 0 and pi; the paper specifies `[0,pi)`.
2. **Preprocessing:** legacy scripts normalize each loaded image by its own min/max; the paper specifies HU clipping followed by a fixed `[0,1]` mapping.
3. **Frequency thresholds:** legacy scripts often use a single cutoff ratio (e.g. 0.3); the paper specifies low/mid/high thresholds at 0.1 and 0.5 of `wmax`.
4. **Adaptive PGD:** one legacy PGD helper hard-codes 60 angles and a non-standard target logit; the canonical version uses labels, the active geometry, random starts, the paper step size, and per-sample best-of-restarts selection.
5. **Saliency drift:** legacy code computes an L2 distance/center-of-mass shift; the manuscript metric is `1 - SSIM` and is used canonically.
6. **RSDF / physics / geometry / UPP:** these are missing from the pasted fragments and were implemented from the manuscript equations and tables.

## What is still needed for exact reproduction

The manuscript and supplied code do not contain the exact patient IDs assigned to each split, the full nodule-centred slice-selection procedure, original model checkpoints, or a complete discrete specification for every structured attack. Consequently, this repository is a faithful and testable reconstruction of the stated methodology, not a claim that the paper's table entries can be regenerated bit-for-bit from arbitrary LIDC preprocessing.
