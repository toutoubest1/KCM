# Data split — stage 3 (ACDC as development data)

Script: `scripts/v3/prepare_acdc.py`. Config: `configs/v3/acdc.json`. Outputs: `data/v3/acdc_index.json`,
`acdc_partitions.json`, `acdc_units.json`, `acdc_streams.json`. Everything below was fixed **before any ACDC
feature was computed**.

## Source
* ACDC `rgb_anon_trainvaltest.zip`: 4006 adverse images (fog 1000, night 1006, rain 1000, snow 1000), each
  with its corresponding normal-condition image (`*_ref`), 1920×1080.
* Downloaded by the PI on 2026-10-02 and resized on the PI's Mac exactly like BDD (480×270, INTER_AREA,
  JPEG q90; script `acdc_pack.py`). All 8012 images were packed.
* Encoder input 182×322, same as stage 2. No ACDC annotation is used.

## Structure discovered
* Images are sparse frames of 26 GoPro recordings: fog 6, night 7, rain 5, snow 8. Median gap is 3–13
  frame numbers.
* A recording is split into chapters (GOPRxxxx, GP01xxxx, …), which are separate video files covering
  successive road segments.
* The official ACDC train/val/test split cuts across recordings, so it is ignored.
* The normal-condition images are stored under the adverse recording's name, in the same order. The identity
  of the underlying normal-condition drive is not provided.

## Partition (unit and rule)
* **dev:** recording-level. Per condition, recordings are taken in stable-hash order into dev until dev
  holds ≥ 30% of that condition's frames. Dev recordings are disjoint from everything else.
* **In-control pool:** all non-dev recordings, split at the **chapter** level into ref / calib / norm. Each
  chapter, in hash order, goes to the role with the largest remaining deficit against the target shares
  ref 0.40, calib 0.45, norm 0.15. Chapters of one recording that share a role are concatenated in order.
* Within the pool, ref, calib and norm never share a chapter, but may share a recording.

**Deviation from the V3-0 plan (made before any feature, recorded honestly):**
* V3-0 planned recording-level units for all partitions. With only 26 large recordings, recording-level
  greedy filling either left the reference bank empty or gave fewer than 19 calibration sequences, which
  makes a conformal threshold at α = 0.05 impossible.
* Hence only dev, which is used for evaluation, is strictly recording-disjoint.
* Calibration chapters can come from the same recording as reference-bank chapters. This could make
  calibration slightly anti-conservative for novel dev recordings. The dev clean PFA tests exactly this.

| role | frames (normal) | units | content |
|---|---|---|---|
| dev | 1507 | 12 recordings | fog 0478, 0479, 0480; night 0351, 0397, 0594; rain 0400, 0571; snow 0122, 0604, 0605, 0607 |
| ref | 970 | 8 recording-role units | k-NN bank, MEWMA reference (normal frames only) |
| calib | 1140 | 12 recording-role units | conformal calibration (normal frames only) |
| norm | 389 | 6 recording-role units | evidence normalisation, score standardisation, joint normalisers |

The fog dev set is dominated by one recording (0478, 335 of 371 frames). All dev results are reported per
recording and with recording-cluster bootstrap CIs.

## Streams (τ = 40 normal frames, monitored from t = 20, H = 30)
The streams are shorter than BDD's (τ = 100) because ACDC sequences are short. The protocol is the same for
calibration and evaluation.

| type | construction | dev count |
|---|---|---|
| clean | non-overlapping 40-frame chunks of a unit's normal sequence | dev 32 (PFA); calib 23 (threshold); norm 7 |
| condition | dev recording, blocks j = 0, 20, …: normal[j, j+40) → adverse[j+40, j+70); the route continues and the condition changes | fog 14, night 11, rain 13, snow 9 |
| hard negative | same pre-change block → 30 normal frames of a different dev recording (cyclic order); scene changes, condition stays normal | 47 |

* Condition and hard-negative streams overlap within a recording (stride 20), so they are not independent.
  Inference is clustered by recording.
* Calibration uses 23 sequences, so the conformal threshold at α = 0.05 is the 23rd order statistic, i.e. the
  maximum.

## Frame-level diagnostic set (D1)
All dev adverse frames paired with their normal frames: fog 371, night 413, rain 374, snow 349. A secondary
replication uses the calib-pool adverse frames, which no fitted quantity sees.

## Untouched
The v1 test split (closed) and the v2 BDD final holdout: no features are extracted from either, and neither
is read.
