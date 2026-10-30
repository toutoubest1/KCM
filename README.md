# Known-Condition Monitor (KCM): experiment source code

Code-only distribution for *Calibrated Online Monitoring of Known Adverse Visual Conditions in Driving Video Using Frozen Vision-Transformer Representations*.

## Scope

- Synthetic-corruption experiments on real BDD100K video frames (not a separate fully synthetic Monte Carlo simulation).
- BDD100K confirmatory evaluation; ACDC development/supporting analyses; RADIATE external validation.
- Frozen KCM, baselines, ablations, backbone comparisons, calibration, localisation and statistical analysis.

No precomputed experimental results, fitted model files, raw images, extracted features, pretrained checkpoints, logs, manuscript or generated figures are included. The numerical methods are unchanged. Two scripts now read manual BDD annotations from `data/annotations/` rather than `results/`.

## Files

- `src/rmrm/`: shared methods, corruptions, frozen backbones and feature extraction.
- `scripts/data_prep/`: dataset preprocessing.
- `scripts/v2/`–`scripts/v6/`: interdependent experiment pipelines; keep these directory names.
- `configs/`: dataset paths, protocols and frozen specification.
- `data/`: metadata, fixed data splits, stream definitions and manual annotations; no raw images.
- `docs/`: selected protocol and method documentation; no run logs or result reports.
- `weights/README.md`: checkpoint filenames and original-source information.
- `tests/refit_frozen_kcm.py`: optional comparison with a model generated locally by the confirmatory run.

## Setup and limitations

Run commands from the repository root. Use Python 3.11 and install `requirements.txt`, or create the environment from `environment.yml`. Version pins are copied from the supplied archive; installation and the claimed reference environment have not been independently verified.

```bash
pip install -r requirements.txt
mkdir -p results/v3 results/v4 results/v5 results/v6
```

Download the datasets and pretrained checkpoints yourself. Edit `configs/paths.json`, then run `python scripts/setup_data_links.py` after preprocessing. Several scripts write into the output directories above without creating all parents, so create them first.

Execute the feature extraction and fitting stages before evaluation. In particular, `bdd_fit.py` generates `results/v3/bdd_system.pkl`, `run_confirmatory.py` generates `results/v5/dino/kcm_model.pkl`, and `run_backbone.py` generates the thresholds needed by the secondary-backbone RADIATE runs. Those files are local run outputs, not missing files to upload.

This package has passed Python syntax parsing, configuration/metadata JSON parsing and local-module availability checks. Full dataset/model execution has not been run. This is not a claim that every paper result has been reproduced.

## 3. Obtaining the datasets (not included)
| dataset | files used | source |
|---|---|---|
| BDD100K | MOT-2020 images `images20-track-train-{1..4}.zip`, `images20-track-val-1.zip`; `bdd100k_labels.zip` (key-frame attributes) | https://bdd-data.berkeley.edu (registration) |
| ACDC | `rgb_anon_trainvaltest.zip` (adverse + corresponding normal images) | https://acdc.vision.ee.ethz.ch (registration) |
| RADIATE | the 22 pre-registered sequence zips (list in `docs/RADIATE_PREREGISTRATION.md` and `scripts/data_prep/radiate_pack.py`) | https://pro.hw.ac.uk/radiate/ (registration form) |

## 4. Expected raw-data layout and preprocessing
Edit `configs/paths.json` and run `python scripts/setup_data_links.py`, or create these folders directly:
```
data/raw/bdd_track_train/<video>/<video>-0000001.jpg      480x270 JPEG q90
data/raw/bdd_track_val/<video>/<video>-0000001.jpg        480x270 JPEG q90
data/raw/acdc/<fog|night|rain|snow>/<split>[_ref]/<GOPRxxxx>/<name>.jpg
data/raw/radiate/<sequence>/<00000>.jpg + meta.json        zed_left, 5 Hz, 480x270
data/bdd100k_attrs_v2.csv                                  (included; regenerate with bdd_attrs_v2.py)
```
Preprocessing commands. All resizing is 480×270 with INTER_AREA, stored as JPEG at quality 90.

    python scripts/data_prep/bdd_attrs_v2.py /path/bdd100k_labels.zip                     # -> bdd100k_attrs_v2.csv
    python scripts/data_prep/bdd_train_resize_pack_v2.py images20-track-train-K.zip OUT A B   # (cwd with the csv) -> tar; extract into data/raw/bdd_track_train
    python scripts/data_prep/bdd_track_resize_pack.py images20-track-val-1.zip OUT 0 200   # -> tar; extract into data/raw/bdd_track_val
    python scripts/data_prep/acdc_pack.py --zip /path/rgb_anon_trainvaltest.zip          # -> data/raw/acdc
    python scripts/data_prep/radiate_pack.py --zip-dir /path/radiate_zips                # -> data/raw/radiate (22 sequences)

* **BDD100K train:** only clean daytime drives and condition drives are packed; night drives are capped at 40 per
  train part.
* **RADIATE:** frames are taken in `zed_left.txt` timestamp order and every 3rd frame is kept (15 Hz → 5 Hz).
  `city_2_0.zip` contains only 1385 of its 2634 listed frames in the public distribution; the missing frames are
  skipped.

## 5–8. Reproducing the experiments (requires the datasets)
Run every command from the repository root, in this order. The exact split, stream and session lists are already in
`data/v2`, `data/v3` and `data/v4`; regenerating them with steps 0b–0d gives the same files.

**0. Splits (optional regeneration)**

    python scripts/v2/prepare_data_v2.py --config configs/v2/base.json    # BDD100K roles + all streams (clean, synthetic, natural, hard negatives)
    python scripts/v3/prepare_acdc.py                                      # ACDC index (26 recordings)
    python scripts/v4/prepare_v4.py                                        # BDD normT/normS split; ACDC 4-fold recording folds

**1. BDD100K features and reference statistics** (DINOv2; synthetic corruptions are applied on the fly)

    python scripts/v2/extract_v2.py --config configs/v2/base.json --parts ref norm calib dev
    python scripts/v2/extract_v2.py --config configs/v2/base.json --norm_corrupt          # corrupted normT clips (KCM training)
    python scripts/v3/extract_acdc.py                                                       # ACDC features (all 26 recordings)
    python scripts/v3/bdd_fit.py                                                            # k-NN reference scorer + baseline fits -> results/v3/bdd_system.pkl

**2. ACDC and BDD dev (development evidence, stage 4)**
* ACDC 4-fold recording cross-fitting gives fog, rain, snow and night detection. The leave-condition-out
  (withheld-family) analysis covers MEWMA, block-2 k-NN and KCM.
* BDD dev leave-one-family-out.

      python scripts/v4/audit_partitions_v4.py
      python scripts/v4/run_v4_bdd.py results/v4 && python scripts/v4/run_v4_acdc.py
      python scripts/v4/sum_bdd.py && python scripts/v4/sum_acdc.py
  `results/KNOWN_CONDITION_RESULTS_V4.csv` and `results/UNSEEN_SHIFT_FAMILY_RESULTS_V4.csv` consolidate these
  summaries. These consolidated historical CSVs are omitted; use the freshly generated `results/v4/*` summaries.
* Development-history diagnostics cited in §4.6: `scripts/v3/acdc_diag.py` and `scripts/v3/probe_acdc.py`, with
  outputs in `results/v3/diag/`.

**3. Freeze, then the BDD100K confirmatory pass** (synthetic and real streams; one pass)

    python scripts/v5/select_single_layer_dev.py     # ablation A2: b* chosen on dev before the freeze (block 4)
    python scripts/v5/audit_final_split.py           # role disjointness; holdout never touched
    python scripts/v5/extract_dino_holdout.py        # holdout features (clean, synthetic, real-condition, HN clips)
    python scripts/v5/run_confirmatory.py            # frozen KCM + 7 baselines + ablations A1-A5/A7 + LOFO stress test, alpha grid
    python scripts/v5/localization_v5.py             # localisation (ablation A6), overlays in figures/v5_localization
    python scripts/v5/extract_backbone_glob.py deit_s16 && python scripts/v5/run_backbone.py deit_s16
    python scripts/v5/extract_backbone_glob.py clip_b16_openai && python scripts/v5/run_backbone.py clip_b16_openai
    python scripts/v5/compute_v5.py                  # runtime / memory
    python scripts/v5/summarize_v5.py && python scripts/v5/summ_backbones.py && python scripts/v5/aux_results.py
    python scripts/v5/label_audit_sheets_bdd.py      # sheets for the pre-results BDD label audit (manual input annotations in data/annotations)

* **Frozen method:** `scripts/v5/kcm.py` and `configs/v5/frozen.json`; spec in `docs/FROZEN_METHOD.md`.
* **Locally generated frozen DINOv2 model:** `results/v5/dino/kcm_model.pkl`. `python tests/refit_frozen_kcm.py` re-fits it from
  the features and checks equality and h = 8.565.

**4. RADIATE (pre-registered; `docs/RADIATE_PREREGISTRATION.md`, the original run log (omitted from this code-only distribution))**

    python scripts/v6/label_audit_sheets.py          # pre-score visual audit sheets (ratings stored: results/v6/label_audit/ratings.json)
    for b in dinov2_s14 deit_s16 clip_b16_openai; do python scripts/v6/extract_radiate.py $b; python scripts/v6/run_radiate.py $b; done
    python scripts/v6/summarize_radiate.py           # summary / per-sequence / per-session / HN / support rule (classification C)
    python scripts/v6/posthoc_diagnostics.py         # POST HOC session diagnostic (labelled; does not change the result)
    python scripts/v6/make_radiate_figures.py

* **What is re-instantiated:** `run_radiate.py` keeps the direction frozen. It re-estimates only the normalisation
  (on the R sessions) and the threshold (on 49 calibration chunks from 6 sessions).
* **Statistical unit:** the recording session.
* **Detectors:** all four are run (KCM, multilayer MEWMA, block-2 k-NN EWMA, final-layer deep-kNN EWMA).

## 9. Baselines and ablations (where they are implemented)
* **Baselines** (`src/rmrm/baselines.py`, plus the g2 detector in `src/rmrm/ccd.py` and `scripts/v6/run_radiate.py`):
  - multilayer MEWMA;
  - block-2 k-NN EWMA;
  - final-layer MEWMA, deep-kNN EWMA, T² EWMA, T² CUSUM and sliding MMD.
* **Ablations** (all in `run_confirmatory.py`):
  - A1/A2/A7: single-block KCM, blocks 2–12;
  - A3: frozen four-block KCM;
  - A4b: window-quantile calibration;
  - A4c: calibration-set size, in `aux_results.py`;
  - A5: KCM + stillness;
  - leave-one-family-out stress test.
* **A6 localisation:** `localization_v5.py`.
* **Backbones:** `run_backbone.py` (BDD100K) and `run_radiate.py` (RADIATE).
* **Hard-negative analyses:** `summarize_v5.py` and `aux_results.py` (per post-change drive).
* **Statistics:** `summarize_v5.py`, `make_tables.py` and `summarize_radiate.py`. They cover:
  - Clopper–Pearson intervals;
  - drive-cluster bootstrap;
  - Wilcoxon signed-rank, rank-biserial and Holm tests;
  - matched false-alarm curves;
  - the RADIATE sequence bootstrap.


## Optional generated plots

After the experiments and summaries finish, `scripts/v6/make_radiate_figures.py` can generate RADIATE plots. Its cross-dataset panel also reads `results/KNOWN_CONDITION_RESULTS_V4.csv`, a historical consolidated file not emitted by the included v4 summary scripts. That optional plotting stage therefore needs additional consolidation; it is not part of the core detector runs. Manuscript table/figure wrappers and stored-result verification are intentionally omitted.

## Original terminology

`normT` means training drives; `normS` means normalisation drives. Versioned directories reflect development stages, not redundant independent releases. ACDC was used in development. RADIATE failed its preregistered specificity criterion (classification C); retain this interpretation when presenting reruns.
