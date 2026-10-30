# RADIATE external validation — pre-registration (stage 5, E10)

**Written 2026-10-03, before any RADIATE file was downloaded or any detector output was seen.** The frozen
method (FROZEN_METHOD.md, tag `rmrm-frozen-kcm-v5`) is not modified. This file is committed before the
download.

## 1. Files and roles
The file list was read from the PI's Dropbox screenshots of the RADIATE share.
* **Unit of independence:** the recording **session**, i.e. the filename prefix `<scene>_<session>`. For
  example `fog_8_0`, `fog_8_1` and `fog_8_2` are segments of one session.
* **Session disjointness:** roles are session-disjoint wherever the data allow.
* **Camera:** only `zed_left` is used. The zips must be downloaded whole, because they bundle radar, lidar
  and the stereo pair.

| role | files | size (GB) | use |
|---|---|---|---|
| **R — reference / normalisation** | `city_2_0.zip`, `city_5_0.zip`, `junction_3_0.zip` | 1.85 + 1.61 + 1.88 = **5.34** | KCM normalisation (median / MAD of d); baseline reference bank and MEWMA in-control fit |
| **C — calibration** | `city_1_1.zip`, `city_3_0.zip`, `city_6_0.zip`, `junction_1_2.zip`, `junction_2_0.zip`, `motorway_2_1.zip` | 1.95 + 2.11 + 1.88 + 2.32 + 2.14 + 1.71 = **12.11** | conformal thresholds (6 sessions) |
| **T0 — clean test** | `city_7_0.zip`, `motorway_1_0.zip`, `rural_1_1.zip` | 1.92 + 1.72 + 1.93 = **5.57** | clean PFA; pre-change segments; hard negatives (3 sessions) |
| **Fog test** | `fog_6_0.zip`, `fog_8_0.zip`, `fog_8_1.zip`, `fog_8_2.zip` | ≈ 1.2 (size not visible) + 1.15 + 1.14 + 0.71 ≈ **4.2** | 2 sessions |
| **Rain test** | `rain_4_0.zip`, `rain_4_1.zip` | 1.84 + 1.74 = **3.58** | 1 session |
| **Snow test** | `snow_1_0.zip` | **1.15** | 1 session |
| **Night test** | `night_1_0.zip`, `night_1_2.zip`, `night_1_4.zip` | 1.64 + 1.84 + 1.89 = **5.37** | 1 session; alternate segments |
| **total** | **22 files** | **≈ 37.3 GB** | — |

**Rule for files not visible in the screenshots:**
* Any `fog_*`, `rain_*` or `snow_*` file that exists in the share but was not visible (for example
  `rain_1_0`, `rain_2_0` or `rain_3_0`) **is also to be downloaded and assigned to its condition's test
  role**.
* Night is capped at the three listed segments, because all six come from one session.
* No other file is used. `tiny_foggy.zip` is a sample and is excluded.

## 2. Pre-processing (identical to BDD)
* Take `zed_left` frames at 15 Hz in timestamp order and keep every third frame, giving **5 Hz**, the BDD
  rate of the frozen protocol.
* Resize 672×376 → 480×270 (INTER_AREA, JPEG q90), then to the frozen encoder input of 182×322.
* No cropping and no rectification.

## 3. Frozen-method re-instantiation (frozen procedure, section 3)
* The direction w is **unchanged** (trained on BDD normT + ACDC). No RADIATE frame trains anything.
* **Normalisation:** median / 1.4826·MAD of d over all R frames.
* **Calibration:**
  - Calibration chunks are non-overlapping 100-frame chunks of every C sequence. Each chunk's maximum is taken
    over [20, 100), and the threshold is the split-conformal threshold at α = 0.05.
  - **Feasibility rule:** ≥ 30 chunks is the target. With 19–29 chunks the experiment proceeds and the limitation is
    reported. With fewer than 19 no finite threshold exists, so the experiment is reported as infeasible and the
    method is not changed.
* **Baselines**, with the same roles:
  - multilayer MEWMA, with its reference and normalisation fitted on R;
  - block-2 k-NN EWMA, using an R bank and Evidence normalisation on R;
  - deep-kNN EWMA on the final layer.

  Every baseline's threshold is set on C.

## 4. Test streams (stage-5 protocol: τ = 100, monitored from 20, H = 60)
* **Clean:** non-overlapping 100-frame chunks of T0 sequences. PFA is counted per chunk and reported per
  sequence.
* **Condition:**
  - The pre-change segment is a T0 chunk, taken in round-robin over T0 sequences.
  - The post-change segment is a consecutive 60-frame window of a condition sequence; windows are
    non-overlapping with stride 60.
  - Every condition window is used exactly once.
* **Hard negatives:** a T0 chunk of one session followed by a 60-frame window of a *different* T0 session
  (clear → clear scene change). Every ordered pair of T0 sessions is used, with windows in round-robin.
* **Pre-results label audit:** three frames per condition sequence are rated V / A / N, exactly as in E1b,
  before any score is computed.

## 5. Endpoints and reading (fixed now)
* **Endpoints** (KCM vs MEWMA and block-2 k-NN, α = 0.05):
  - clean PFA with Clopper–Pearson CI;
  - detection ≤ 60 frames and median delay for fog, rain, snow and night;
  - hard-negative alarm rate.
* **Units:** results are given per sequence and per session. The bootstrap resamples **sequences**.
* **Supports the claim on RADIATE for a condition** if all of the following hold:
  1. The clean-PFA CI contains 0.05, or lies below it.
  2. HN ≤ 0.10.
  3. That condition's detection ≥ HN + 0.50.
* **Secondary:** if compute allows, CLIP ViT-B/16 and DeiT-S with the identical recipe.

## 6. Known limitations, stated in advance
* **Few independent sessions:** fog 2, rain 1, snow 1, night 1.
  - RADIATE confirms platform independence (country, camera and frame rate differ).
  - It cannot establish per-condition population-level rates.
  - Results will be reported as per-session case evidence.
* **Scene mismatch:** RADIATE adverse sequences are suburban (fog, rain, snow) or motorway (night), while the
  clean roles mix city, junction, motorway and rural. Splices therefore also contain scene changes. The
  hard-negative arm measures exactly that confound.
* **Different sensor:** the ZED camera's low resolution and lens-borne raindrops and snow differ from BDD.
  This tests real transfer of the frozen direction; it was not chosen to favour it.
