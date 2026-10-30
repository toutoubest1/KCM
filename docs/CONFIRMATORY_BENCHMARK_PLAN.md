# Confirmatory benchmark plan (stage 5)

**Written and committed with FROZEN_METHOD.md before any confirmatory result was computed.** The method
is not modified after this point. Outcomes that deviate from this plan are reported as deviations.

## Claim under test
> A monitor built on frozen multiscale vision representations can detect adverse visual conditions that
> are represented in its development/training data. It does so online, with reliable clean-drive
> false-alarm calibration, low sensitivity to ordinary scene and content changes, competitive detection
> delay, and useful spatial localisation where a spatial ground truth exists.

**Not claimed:**
* universal out-of-distribution detection;
* detection of arbitrary unseen real conditions.

The stage-4 negative result (unseen-family detection 0.43; unseen real weather poorly detected) is
re-reported as a stress test (E8).

## Data (details and audit: FINAL_DATA_SPLIT_AUDIT.md)
* **Confirmatory evaluation set:** the untouched v2 BDD100K holdout.
  - 62 clean holdout drives give 62 clean streams.
  - 620 synthetic streams: 10 corruptions × 62 drives, severity cycled 1–3.
  - 248 real-condition streams: night 62, dawn/dusk 62, rain 62, snow 62. They use 129 condition-holdout
    drives: night 81, dawn/dusk 24, snow 14, rain 10.
  - 310 hard-negative streams, 62 per category: random scene, road geometry, traffic density, object
    composition, camera motion.
  - No feature of any holdout drive has ever been computed (verified).
* **What the BDD holdout can test:** calibration on unseen drives of the training camera domain; scene-change
  specificity; known synthetic corruptions; and real night, dawn/dusk, rain and snow on a camera and continent
  different from the real-condition training data (ACDC).
* **What it cannot test:**
  - fog (BDD has no fog drives);
  - visually verified rain and snow (BDD labels are per-video tags and often weak; see E1b);
  - a third independent capture platform.
* **ACDC:** all 26 recordings informed stage-3/4 decisions. ACDC results are reported only as
  development-reused evidence (stage-4 cross-fitted protocol, re-run with the frozen code), never as
  confirmatory.
* **Third dataset:** assessed in section E10. If one is necessary and needs a manual download, that part
  pauses and the request goes to the PI.

## Protocol (identical to stage 2/4)
* **BDD streams:** τ = 100 pre-change frames at 5 Hz, monitored from t = 20, post-change horizon H = 60.
* **Clean false alarm:** an alarm in [20, 100).
* **Detection:** an alarm in [τ, τ + 60), counted only for streams without a pre-change false alarm.
* **Delay:** t_alarm − τ + 1. The horizons are 10, 20, 30 and 60 frames.
* **Hard-negative alarm:** an alarm within 60 frames after the scene cut.
* **Statistical units:**
  - the clean or pre-change drive for PFA, synthetic and hard-negative streams;
  - the **condition drive** for real-condition streams (results are aggregated per condition drive, with
    cluster bootstrap over condition drives).
* **Frames are never treated as replicates.**

## Experiments
**E1 — primary, KCM on the BDD holdout.** Clean PFA with Clopper–Pearson CI. Real-condition detection per
condition and pooled, horizons, median delay, and the per-condition-drive delay distribution.
Hard-negative alarm rates, pooled and per category. Synthetic detection per corruption and severity.
* **E1b — label audit (secondary).** Before any holdout score is computed, every condition-holdout drive is
  rated *visibly adverse / ambiguous / not visibly adverse*. The rating uses three frames at positions 5, 30
  and 55 of the 60-frame post-change window, with no detector output available. Rain, snow and dawn/dusk
  detection is also reported on the visibly-adverse subset. The primary analysis uses the original tags.

**E2 — baselines** (same DINOv2 features, the same BDD calib drives, conformal α = 0.05):
1. multilayer MEWMA (CLS of blocks 2–12, PCA-32);
2. best single-layer novelty monitor: block-2 k-NN distance + EWMA, the stage-2 choice;
3. final-layer-only MEWMA (final CLS);
4. deep-kNN + EWMA (final CLS, k = 10);
5. Mahalanobis T² + EWMA (final CLS, PCA-32);
6. CUSUM on T² (κ = 0.5);
7. sliding-window MMD (w = 20, Gaussian kernel, median heuristic).

The baselines are fitted on the BDD ref drives (stride 2) and the 52 norm drives, as in stage 2.

**Matched false-alarm comparison:** thresholds are calibrated at α ∈ {0.02, 0.05, 0.10, 0.20}. Detection and
delay are reported against the realised holdout PFA (delay–PFA curves).

**E3 — pre-specified ablations** (DINOv2, BDD holdout):

| id | variant |
|---|---|
| A1 | KCM on the final layer only (block 12) |
| A2 | KCM on the single best early/mid block b*, chosen **on BDD dev before the freeze** by highest dev synthetic detection (`results/v5/single_layer_selection_dev.json`) |
| A3 | frozen KCM (blocks 2, 4, 6, 8) |
| A4 | calibration: (a) conformal on 66 drives (frozen); (b) the stage-1 overlapping-window quantile; (c) conformal on random 12- and 20-drive calibration pools, 200 draws each, giving the threshold and holdout-PFA distribution |
| A5 | KCM + stillness (rule C1, normalisers on normS) vs KCM |
| A6 | localisation from block-6 vs block-8 vs block-12 (final-layer) patch evidence, and block-8 stillness |
| A7 | layer-role profile: single-block KCM for every block 2…12 (descriptive; contains A1 and A2) |

**E4 — backbone generalisation.** The same recipe, layer mapping, training frames, normalisation and
calibration drives, applied to:
* DINOv2 ViT-S/14 (self-supervised; 182×322);
* OpenAI CLIP ViT-B/16 (language-supervised; 176×320, positional embedding interpolated);
* DeiT-S/16 (supervised ImageNet-1k; 176×320).

All three are 12-block ViTs. "Early/mid" = blocks 2, 4, 6, 8; "final" = block 12. Each backbone is
evaluated with KCM, multilayer MEWMA, final-layer MEWMA and deep-kNN + EWMA on all holdout streams. No
per-backbone redesign.

**E5 — hard negatives as a main experiment.** Holdout categories, pooled and per category, for KCM and all
baselines. KCM is also evaluated on ACDC hard negatives under the development-reused protocol.

**E6 — synthetic controlled experiments.** Severity response, per-corruption detection and delay, layer
roles (A7), and synthetic-vs-real comparison. Synthetic results are never used to claim generalisation to
unseen real weather.

**E7 — localisation.**
* Holdout occlusion streams (62), using the exact occluder mask pooled to the patch grid.
* Metrics: per-stream AUROC and AUPRC of the map averaged over [τ, τ+19], for block-6, block-8 and block-12
  patch evidence and block-8 stillness.
* Qualitative overlays chosen by a fixed rule: the best, median and worst block-6 AUROC streams, plus
  one night stream and one hard-negative stream.

**E8 — stress test (secondary).** The frozen *training procedure* is re-run per held-out family
(F1 illumination incl. ACDC night; F2 blur; F3 noise; F4 weather incl. ACDC fog/rain/snow; F5 occlusion).
Each run is evaluated on the holdout synthetic streams of that family, against MEWMA and g2 on the same
streams.

**E9 — calibration audit.**
* Holdout PFA and CI, the number of calibration and test units, and threshold variability (A4c).
* Development-reused ACDC cross-fitted PFA with per-fold thresholds, and the stage-4 calib/normalisation
  scale-ratio issue revisited.

**E10 — third real dataset.**
* Gap analysis: the confirmatory real-condition evidence has no fog, only weak-label rain and snow, and a
  single platform (BDD).
* If a dataset that closes this gap needs a manual download, the request goes to the PI with exact files and
  sizes, and that part waits.

**E11 — compute.** Encoder ms/frame per backbone (2 CPU threads, batch 16); monitoring update time per frame
for KCM and the baselines; memory (parameters, banks, statistics); fitting and calibration cost.

## Statistics
* **Paired comparisons** (KCM vs MEWMA, g2, final-layer MEWMA, deep-kNN, and the best baseline per endpoint):
  - **synthetic:** per pre-change holdout drive, mean censored delay (no detection = 61) across that drive's
    synthetic streams, giving 62 units;
  - **real conditions:** per condition drive, the mean detection indicator;
  - **reported:** mean difference, drive-cluster bootstrap 95% CI (2000 resamples), Wilcoxon signed-rank p,
    rank-biserial effect size, n units.
* **No multiplicity-free claims:** each family of comparisons is listed with Holm-adjusted p.

## Pre-specified reading of the evidence (guides the final A/B/C classification)
* **Calibration:** holdout PFA CI must contain 0.05, or lie below it, with point estimate ≤ 0.10.
* **Specificity:** holdout HN pooled ≤ 0.10, and no category > 0.20.
* **Real known conditions:** night detection ≥ 0.80. Rain, snow and dawn/dusk are reported with their label
  caveat.
* **Advantage:** KCM at least as good as the strongest baseline at comparable PFA on real-condition detection,
  and not worse on hard negatives.
* **Robustness to backbone:** the qualitative pattern (KCM > MEWMA on real known conditions at equal or
  lower HN) holds on at least 2 of 3 backbones.
* **Classification A** additionally requires independent verified real evidence for each claimed condition,
  including fog. If that is missing, the best possible classification is B, with the gap named.
