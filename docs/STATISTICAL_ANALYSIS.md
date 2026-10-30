# Statistical analysis (stage 5, BDD holdout, DINOv2; α = 0.05 conformal for every detector)

**Units:**
* the holdout pre-change drive (62) for synthetic streams and hard negatives;
* the **condition drive** for real conditions (night 59, dawn/dusk 24, rain 10, snow 14 with valid streams).

Frames are never replicates. Results include the drive-cluster bootstrap 95% CI of the mean paired
difference (2000 resamples), the Wilcoxon signed-rank test, the rank-biserial effect size r, and Holm
adjustment within each endpoint over the 7 comparators.

**Full table:** `results/v5/paired_statistics.csv`. Per-recording values: `CONFIRMATORY_RESULTS_PER_RECORDING.csv`.

## KCM vs the strongest comparators
| endpoint | comparator | n | mean paired difference (KCM − comp.) [95% CI] | Wilcoxon p (Holm) | r | KCM better on |
|---|---|---|---|---|---|---|
| synthetic delay (frames; no detection = 61) | MEWMA multilayer | 58 | **−10.7 [−13.3, −8.0]** | < 1e-8 (< 1e-8) | −0.82 | 88% of drives |
| synthetic delay | block-2 k-NN EWMA | 54 | **−4.9 [−7.5, −2.2]** | 6e-4 (6e-4) | −0.54 | 67% |
| synthetic delay | deep-kNN EWMA (final) | 48 | −25.9 [−30.0, −21.5] | < 1e-8 | −0.97 | 92% |
| real, pooled (detection) | MEWMA multilayer | 106 | −0.02 [−0.09, +0.05] | 0.46 (0.85) | −0.19 | n.s. |
| real, pooled | block-2 k-NN EWMA | 102 | +0.05 [−0.02, +0.13] | 0.19 (0.56) | 0.34 | n.s. |
| real, pooled | deep-kNN EWMA (final) | 95 | −0.05 [−0.15, +0.06] | 0.42 (0.85) | −0.16 | n.s. |
| real night (detection) | MEWMA multilayer | 58 | 0 (both 1.00) | — | — | tie |
| *real night, delay (exploratory)* | MEWMA multilayer | 58 | **−4.7 frames [−6.3, −3.2]** | 9e-9 | — | 86% |
| real dawn/dusk | MEWMA multilayer | 24 | −0.08 [−0.29, +0.08] | 0.41 | −0.33 | n.s. |
| real rain | MEWMA multilayer | 10 | +0.26 [0.00, +0.51] | 0.25 (1.0) | 1.0 | n.s. (n = 10) |
| real snow | MEWMA multilayer | 14 | −0.21 [−0.57, +0.14] | 0.26 (0.79) | −0.43 | n.s. (n = 14) |
| hard-negative alarm rate | MEWMA multilayer | 58 | −0.010 [−0.024, 0.000] | 0.08 (0.08) | −1.0 | n.s. |
| hard-negative alarm rate | block-2 k-NN EWMA | 54 | **−0.063 [−0.111, −0.019]** | 0.014 (0.027) | −0.53 | — |
| hard-negative alarm rate | deep-kNN EWMA (final) | 48 | **−0.188 [−0.242, −0.133]** | < 1e-6 | −0.92 | — |

Against the final-layer MEWMA, T², CUSUM and MMD monitors, KCM is better on every detection endpoint
(p < 1e-6). Those monitors barely react at all (synthetic 0.03–0.05; night 0.02–0.03), and so have
slightly lower HN rates (0.00–0.02 vs 0.05).

## What is and is not statistically supported
* **Supported:**
  1. **Faster detection** of known synthetic conditions than every baseline, with large effects.
  2. **Faster** detection of real night than multilayer MEWMA (exploratory endpoint).
  3. **Fewer scene-change alarms** than k-NN monitors.
  4. **Calibrated PFA.**
* **Not supported:**
  1. Any detection-rate advantage over multilayer MEWMA on BDD real conditions. Night is saturated for both
     (1.00), and dawn/dusk, rain and snow show no significant difference.
  2. Any claim about real rain or snow from BDD: n = 10 and 14 drives, and only 3 and 1 of them are
     visibly adverse (E1b).
