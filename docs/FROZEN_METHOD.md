# FROZEN METHOD — Known-Condition Monitor (KCM)

**Frozen:** 2026-10-02. This was done **before any confirmatory result was computed**, and is committed
together with `scripts/v5/kcm.py` and `configs/v5/frozen.json`.

KCM is exactly the stage-3/4 "M2" learned-direction formulation, selected on stage-3/4 development
evidence alone:
* best known-condition detection;
* lowest scene-change alarm rate;
* valid calibration;

The evidence for each:
* BDD dev: PFA 0.017, HN 0.004, synthetic 0.925, real night 0.98;
* ACDC (cross-fitted): fog / night / rain / snow 1.00, with HN 0.

V1 (the two-branch variant) failed the stage-4 gate and is not used. No component was changed after the
stage-4 results.

**Claim scope:** KCM monitors adverse visual conditions **represented in its training data**. It is
**not** a detector of arbitrary or unseen shifts. Stage 4 showed that it detects held-out shift families at
only 0.43 (BDD synthetic LOFO) and unseen real weather poorly.

## 1. Representation
* **Encoder:** frozen DINOv2 ViT-S/14, using the official checkpoint
  `dinov2_vits14_pretrain.pth` (sha256 b938bf1b…).
  - Input: the RGB frame resized to 182×322 with INTER_AREA from the 480×270 cached frame; ImageNet
    mean/std; 13×23 patch grid.
* **Layers:** transformer blocks **2, 4, 6, 8** (early/mid). Blocks 10 and 12 are not used.
* **Features:** φ(t) = concat over b ∈ {2, 4, 6, 8} of [ℓ2(CLS_b(t)), ℓ2(mean-patch_b(t))], giving 3072
  dimensions. ℓ2 is unit-norm normalisation of each 384-d vector.

## 2. Condition direction (the only learned component)
* **Standardisation:** a StandardScaler fitted on the training φ.
* **Model:** w = logistic regression (L2, **C = 0.1**, lbfgs, max_iter 5000), with four groups of equal
  total weight:
  1. **(+)** BDD synthetic-corrupted frames of the 26 **normT** drives. One clip of 30 frames per
     (drive, corruption): 10 corruptions (fog, low_light, rain, snow, motion_blur, defocus_blur,
     gaussian_noise, contrast, brightness, occlusion), with severity (i + j) mod 3 + 1. That gives 260 clips
     and 7800 frames.
  2. **(−)** clean frames of the 26 normT drives (5103 frames).
  3. **(+)** all 4006 ACDC adverse frames (fog 1000, night 1006, rain 1000, snow 1000).
  4. **(−)** all 4006 ACDC normal-condition frames.
* **Score:** d(t) = wᵀ StandardScaler(φ(t)).

## 3. Normalisation, monitoring statistic, calibration
* **Normalisation:** d is standardised by median and 1.4826·MAD over all frames of the 26 **normS** BDD
  drives. These drives are disjoint from training.
* **Statistic:** S(t) = EWMA_λ(d̃)(t) / √(λ/(2−λ)), with λ = 0.1. The EWMA is initialised at 0 and started at
  t = 20 (the monitoring start).
* **Alarm:** at the first t ≥ 20 with S(t) > h.
* **Threshold:** h is the drive-level split-conformal threshold at α = 0.05: the ⌈(n+1)(1−α)⌉-th order
  statistic of max over t ∈ [20, 100) of S(t), over the n = 66 BDD **calib** drives (first 100 frames).
* **New camera or dataset:** the frozen procedure re-estimates only the normalisation (median/MAD) and h on
  that domain's normal-condition normalisation and calibration recordings. w is never re-fitted on
  evaluation data.

## 4. Localisation (separate from the alarm; computed after an alarm or on a fixed window)
* **Map:** the block-6 per-position patch k-NN evidence.
  - Each patch token is ℓ2-normalised, centred, and projected onto the reference PCA-64 basis plus its
    residual norm.
  - The map is the log distance to the 10th nearest reference patch at the same grid position (BDD ref
    bank, stride 10).
  - It is robustly standardised per position on normS.
* **Secondary map:** the block-8 temporal-stillness map, for camera-fixed occluders.
* **Evaluation:** maps are averaged over [τ, τ+19] (fixed window, independent of the alarm).
* The best detection representation (blocks 2–8 global) and the best localisation representation (block-6
  patches) differ by design.

## 5. Not part of the alarm
* The stillness scan, the k-NN novelty branch (N1/M1), content residualisation and any combination rule are
  **not** part of KCM. The stillness scan is evaluated only as a pre-specified ablation (KCM + ST, combined
  as in stage-4 rule C1, with normalisers on normS).

## 6. Hyperparameters (complete list)
| parameter | value |
|---|---|
| blocks | 2, 4, 6, 8 (CLS and mean patch, each ℓ2) |
| classifier | logistic regression, C = 0.1, L2, group-balanced weights |
| training groups | BDD normT corrupted / clean; ACDC adverse / normal |
| normalisation | median / MAD on BDD normS frames |
| EWMA λ | 0.1 |
| monitoring start | t = 20 |
| calibration window | [20, 100) |
| α | 0.05, drive-level split conformal, n = 66 BDD calib drives |
| localisation | block-6 patch k-NN (k = 10, PCA-64, per-position normS standardisation) and block-8 stillness |

Code: `scripts/v5/kcm.py`. Configuration: `configs/v5/frozen.json`.
