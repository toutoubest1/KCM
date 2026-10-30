# Known-Condition Monitor (KCM)

Source code for **Calibrated Online Monitoring of Known Adverse Visual Conditions in Driving Video Using Frozen Vision-Transformer Representations**.

KCM combines frozen vision-transformer features, a learned condition direction, EWMA monitoring, and drive-level conformal calibration to detect known adverse visual conditions in driving video.

## Contents

- Experiments on BDD100K, ACDC, and RADIATE
- Baselines, ablations, and backbone comparisons
- Calibration, detection, localisation, and statistical analysis

## Installation

Use Python 3.11 and install the dependencies:

    pip install -r requirements.txt
    mkdir -p results/v3 results/v4 results/v5 results/v6

Download the datasets and pretrained checkpoints separately. Update `configs/paths.json`, then run:

    python scripts/setup_data_links.py

## Repository Structure

- `src/rmrm/`: shared methods and feature extraction
- `scripts/data_prep/`: dataset preprocessing
- `scripts/v2/`–`scripts/v6/`: experiment pipelines
- `configs/`: configurations and frozen method specification
- `data/`: metadata, fixed splits, and annotations
- `docs/`: protocols and method documentation

## Usage

Run preprocessing, feature extraction, model fitting, and evaluation in that order. See `docs/` and the corresponding scripts for details.

## Notes

Raw datasets, pretrained checkpoints, extracted features, fitted models, and experimental results are not included.

ACDC was used during development. RADIATE external validation did not meet the preregistered specificity criterion.
