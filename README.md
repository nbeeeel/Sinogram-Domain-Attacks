# Robustness of AI-Driven CT Diagnostic Systems Against Projection-Domain Adversarial Attacks

Code for the paper **“Robustness of AI-Driven CT Diagnostic Systems Against Projection-Domain Adversarial Attacks: A Physics-Aware Evaluation Framework.”**

The pipeline is:

`CT slice -> differentiable parallel-beam Radon transform -> projection-domain attack -> ramp-filtered FBP -> optional RSDF -> binary CNN classifier`

The repository includes baseline FGSM, a Poisson-inspired physics proxy, geometry-aware attacks, detector-frequency attacks, universal projection perturbation (UPP), adaptive end-to-end PGD, sparse-view experiments, confidence and entropy metrics, and saliency instability analysis.

## Scope

This code models **numerically generated 2D parallel-beam sinograms**. It does not operate on raw scanner detector data or model scanner-specific acquisition pipelines.

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

The experiments use nodule-centred LIDC-IDRI axial slices with consensus malignancy labels:

* median malignancy `<= 2`: benign
* median malignancy `>= 4`: malignant
* score `3`: excluded
* 2,400 slices total (1,200 per class)
* patient-wise split: 70% train / 15% validation / 15% test
* HU clipping: `[-1000, 400]`
* resize: `256 x 256`, bilinear interpolation with anti-aliasing
* normalization to `[0,1]`
* no data augmentation

Two input modes are supported.

### Manifest + HU arrays

Use `.npy` slices retaining HU values and a CSV manifest:

```text
path,patient_id,label,split
slices/LIDC-IDRI-0001_n1.npy,LIDC-IDRI-0001,0,train
...
```

Run experiments with:

```bash
--manifest manifest.csv --input-mode hu
```

If the metadata contains `path,patient_id,median_malignancy`, generate labels and a patient-wise split with:

```bash
python scripts/build_manifest.py metadata.csv manifest.csv --seed 0
```

### Preprocessed image folders

The following directory structure is also supported:

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

Run with:

```bash
--data-root data --input-mode normalized
```

Images in this mode are expected to already be normalized.

## Configuration

The main experimental parameters are defined in [`configs/paper.yaml`](configs/paper.yaml):

* image size: `256 x 256`
* detector samples: `256`
* angular views: `{30, 60, 90, 120, 180}`
* reconstruction: ideal ramp-filtered FBP
* simulated incident count: `n0 = 1e5`
* optimizer: Adam
* learning rate: `1e-3`
* batch size: `32`
* training epochs: `20`
* primary `L_inf` budget: `epsilon = 0.25`

Angles are generated as:

```text
theta_k = k*pi/N,  k = 0, ..., N-1
```

so the projection angles lie on `[0, pi)`.

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

The RSDF and classifier are trained jointly on clean FBP reconstructions with cross-entropy loss and no adversarial training.

## Experiments

### Sparse-view FGSM

```bash
ctrobust sparse-view \
  --manifest manifest.csv --input-mode hu \
  --view-counts 30 60 90 120 180 --eps 0.25 \
  --output outputs/sparse_view.csv
```

### Attack-family comparison

```bash
ctrobust attack-family \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt --eps 0.25 \
  --output outputs/attack_family.csv
```

### Frequency ablation

```bash
ctrobust frequency-ablation \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt --eps 0.25 \
  --output outputs/frequency.csv
```

### Budget sensitivity

```bash
ctrobust budget-sweep \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --eps-values 0.05 0.10 0.15 0.20 0.30 \
  --output outputs/budget.csv
```

### Adaptive end-to-end PGD

```bash
ctrobust adaptive-pgd \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_rsdf_180.pt --eps 0.25 \
  --steps 10 20 40 --restarts 3 \
  --output outputs/adaptive_pgd.csv
```

### Confidence sweep

```bash
ctrobust confidence-sweep \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --eps-values 0 0.2 0.4 0.6 0.65 0.75 \
  --output outputs/confidence.csv
```

### UPP cross-view transfer

```bash
ctrobust upp-transfer \
  --manifest manifest.csv --input-mode hu \
  --checkpoint outputs/cnn3_180.pt \
  --optimize-views 180 --view-counts 180 120 90 60 30 \
  --eps 0.25 --output outputs/upp_transfer.csv
```

### Architecture ablation

After training the three checkpoints:

```bash
ctrobust architecture-ablation \
  --manifest manifest.csv --input-mode hu \
  --cnn3-checkpoint outputs/cnn3_180.pt \
  --rsdf-checkpoint outputs/cnn3_rsdf_180.pt \
  --cnn5-checkpoint outputs/cnn5_180.pt \
  --output outputs/architecture_ablation.csv
```

## Metrics

The implementation reports:

* accuracy degradation: `A_clean - A_adv`
* prediction-flip rate
* attack success rate on clean-correct samples
* maximum-softmax confidence
* confidence change
* predictive entropy
* saliency instability: `1 - SSIM(S_clean, S_adv)`

Reference values from the paper are available under [`paper_reference/`](paper_reference/).

## Repository layout

```text
src/ctrobust/
  geometry.py       differentiable Radon transform and ramp FBP
  data.py           preprocessing and dataset loaders
  models.py         CNN3, CNN5, RSDF, and diagnostic pipeline
  attacks.py        FGSM, physics, geometry, frequency, UPP, and PGD attacks
  metrics.py        confidence, entropy, flip rate, ASR, and saliency SSIM
  training.py       model training
  evaluation.py     shared evaluation utilities
  cli.py            experiment CLI

configs/paper.yaml  experimental configuration
docs/               methodology and implementation notes
legacy/             original experiment scripts
paper_reference/    paper result tables
tests/              unit tests
```

## Reproducibility

Exact numerical results depend on the dataset split, selected nodule-centred slices, model initialization, trained checkpoints, and hardware/software environment.

For consistent runs:

* use the provided configuration
* keep the random seed fixed
* preserve patient-wise data splits
* use the same preprocessing pipeline
* record package and CUDA versions
* save checkpoints and experiment outputs

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

See the repository license file for usage terms.
