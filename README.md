# Robustness of AI-Driven CT Diagnostic Systems Against Projection-Domain Adversarial Attacks

A clean, push-ready research-code reconstruction for the paper **“Robustness of AI-Driven CT Diagnostic Systems Against Projection-Domain Adversarial Attacks: A Physics-Aware Evaluation Framework.”**

This repository consolidates the supplied exploratory scripts into one implementation of the paper pipeline:

`CT slice -> differentiable parallel-beam Radon transform -> projection-domain attack -> ramp-filtered FBP -> optional RSDF -> binary CNN classifier`

It includes the paper’s baseline FGSM, Poisson-inspired physics proxy, geometry-aware attack, detector-frequency attacks, universal projection perturbation (UPP), adaptive end-to-end PGD, sparse-view experiments, confidence/entropy metrics, and saliency instability.

## Important scope

This code models **numerically generated 2D parallel-beam sinograms**. It does not read or manipulate raw scanner detector data and does not claim scanner-specific physical realizability.

The repository is a reconstruction from the manuscript plus the supplied code fragments. Where the paper does not uniquely specify a discrete implementation, the choice is documented in [`docs/methodology_mapping.md`](docs/methodology_mapping.md) instead of being hidden.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev,plot]'
pytest
python scripts/smoke_test.py
```

GPU is optional. CUDA is used automatically when available if `--device auto` is selected.

## Data

The manuscript uses nodule-centred LIDC-IDRI axial slices with consensus malignancy labels:

- median malignancy `<= 2`: benign
- median malignancy `>= 4`: malignant
- score `3`: excluded
- 2,400 slices total (1,200/class)
- patient-wise split: 70% train / 15% validation / 15% test
- HU clipping: `[-1000, 400]`
- resize: `256 x 256`, bilinear + anti-aliasing
- normalize to `[0,1]`
- no data augmentation

Two input modes are supported.

### Preferred: manifest + HU arrays

Use `.npy` slices retaining HU values and a CSV:

```text
path,patient_id,label,split
slices/LIDC-IDRI-0001_n1.npy,LIDC-IDRI-0001,0,train
...
```

Then use `--manifest manifest.csv --input-mode hu`.

If you have metadata with `path,patient_id,median_malignancy`, create the labels and patient-wise split with:

```bash
python scripts/build_manifest.py metadata.csv manifest.csv --seed 0
```

The paper does not specify enough detail to reconstruct the exact raw-DICOM nodule extraction/annotation-fusion procedure, so that stage is deliberately not fabricated here.

### Compatibility: preprocessed image folders

The supplied scripts used:

```text
data/
  train/
    benign/
    malignant/
  valid/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

Run these with `--data-root data --input-mode normalized`. Unlike the exploratory scripts, this loader does **not** perform a fresh per-image min/max normalization, because the paper specifies HU clipping followed by a fixed mapping to `[0,1]`.

## Paper configuration

The canonical values are recorded in [`configs/paper.yaml`](configs/paper.yaml): 256 detector samples, view counts `{30,60,90,120,180}`, ideal ramp FBP, `n0=1e5`, Adam at `1e-3`, batch size 32, 20 epochs, and primary `L_inf` budget `epsilon=0.25`.

Angles are generated as `theta_k = k*pi/N`, so they lie on `[0, pi)`. This corrects the duplicated endpoint introduced by `torch.linspace(0, pi, N)` in the exploratory scripts.

## Training

Baseline CNN at 180 views:

```bash
ctrobust train \
  --manifest manifest.csv --input-mode hu \
  --views 180 --arch cnn3 \
  --output outputs/cnn3_180.pt
```

RSDF + CNN:

```bash
ctrobust train \
  --manifest manifest.csv --input-mode hu \
  --views 180 --arch cnn3 --rsdf \
  --output outputs/cnn3_rsdf_180.pt
```

The RSDF and classifier are trained jointly on clean FBP reconstructions only, with no adversarial training.

## Reproducing the experiments

Sparse-view FGSM (trains a configuration at each view count):

```bash
ctrobust sparse-view \
  --manifest manifest.csv --input-mode hu \
  --view-counts 30 60 90 120 180 --eps 0.25 \
  --output outputs/sparse_view.csv
```

Primary attack-family comparison:

```bash
ctrobust attack-family \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt --eps 0.25 \
  --output outputs/attack_family.csv
```

Frequency ablation:

```bash
ctrobust frequency-ablation \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt --eps 0.25 \
  --output outputs/frequency.csv
```

Budget sensitivity:

```bash
ctrobust budget-sweep \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --eps-values 0.05 0.10 0.15 0.20 0.30 \
  --output outputs/budget.csv
```

Adaptive end-to-end PGD through FBP + RSDF + classifier:

```bash
ctrobust adaptive-pgd \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_rsdf_180.pt --eps 0.25 \
  --steps 10 20 40 --restarts 3 \
  --output outputs/adaptive_pgd.csv
```

Confidence sweep:

```bash
ctrobust confidence-sweep \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --eps-values 0 0.2 0.4 0.6 0.65 0.75 \
  --output outputs/confidence.csv
```

UPP cross-view transfer:

```bash
ctrobust upp-transfer \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --optimize-views 180 --view-counts 180 120 90 60 30 \
  --eps 0.25 --output outputs/upp_transfer.csv
```

## Metrics

The implementation reports:

- accuracy degradation: `A_clean - A_adv`
- prediction-flip rate
- attack success rate restricted to clean-correct samples
- maximum-softmax confidence and confidence change
- predictive entropy
- saliency instability: `1 - SSIM(S_clean, S_adv)`

Paper-reported values are transcribed under [`paper_reference/`](paper_reference/) for comparison. They are reference values, not generated outputs.

## Repository layout

```text
src/ctrobust/
  geometry.py       differentiable Radon + ramp FBP
  data.py           paper preprocessing + folder/manifest datasets
  models.py         CNN3, CNN5 reference, RSDF, diagnostic pipeline
  attacks.py        FGSM, physics, geometry, frequency, UPP, adaptive PGD
  metrics.py        confidence, entropy, flips, ASR, saliency SSIM
  training.py       clean-reconstruction training
  evaluation.py     shared attack evaluation
  cli.py            experiment CLI
configs/paper.yaml  manuscript parameters
docs/               methodology mapping and implementation decisions
legacy/             the six supplied code fragments, preserved verbatim
paper_reference/    manuscript result tables for comparison
tests/              unit tests for geometry, bounds, preprocessing and model counts
```

## Reproducibility caveats

Exact numerical reproduction still depends on the original patient split, exact selected nodule-centred slices, original checkpoints, and several manuscript details that are not fully specified. In particular, the exact ROI-based streak-view selection, normalization used to combine geometry weights, the discrete UPP cross-view resampling rule, and the exact choice of the frequency-domain variant used in the primary family table are not uniquely recoverable from the paper. The repository makes each such choice explicit and configurable.

The legacy scripts are retained for provenance, but the package code should be treated as the canonical implementation.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

No software license has been selected in this reconstructed repository. Add the license approved by the paper authors/institutions before public release.

Architecture ablation (after training the three checkpoints):

```bash
ctrobust architecture-ablation \
  --manifest manifest.csv --input-mode hu \
  --cnn3-checkpoint outputs/cnn3_180.pt \
  --rsdf-checkpoint outputs/cnn3_rsdf_180.pt \
  --cnn5-checkpoint outputs/cnn5_180.pt \
  --output outputs/architecture_ablation.csv
```
