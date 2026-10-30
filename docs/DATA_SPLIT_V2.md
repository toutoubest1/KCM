# Stage-2 data split (fixed before any stage-2 detector result)

Code: `scripts/v2/prepare_data_v2.py`, config `configs/v2/base.json`; outputs `data/v2/drive_table.csv`
(one row per drive: source, attributes, key-frame object counts, motion proxy, partition),
`data/v2/partitions.json`, `data/v2/streams.json`.

## Sources
* **BDD100K MOT-2020 train parts 1–4** (`images20-track-train-{1..4}.zip`, official server
  http://128.32.162.150/bdd100k/mot20/, 19.4 GB, 800 videos). On the Mac, only the drives needed were resized
  (480×270 JPEG) and transferred: all clean daytime drives and all dawn/dusk / daytime-rain / daytime-snow / fog
  drives, plus at most 40 night drives per part (cap fixed a priori to bound transfer; night is ~45% of BDD):
  **574 train drives, 114,245 frames**.
* **BDD100K MOT val** (stage-1 data, 200 drives) **excluding the closed v1 test drives** (old roles `test`, and
  `test_post` — 17 clean + 62 condition drives are excluded entirely).
* Attributes per drive: official 100k key-frame labels (`bdd100k_labels.zip`): weather, scene, time of day and
  key-frame object counts (`scripts/data_prep` / `rmrm_prep/attrs_v2.py`).

## Partition rule (drive level; no frame of a drive in two partitions)
* **Clean** = daytime and weather ∈ {clear, partly cloudy, overcast}. Hash of the video name assigns each clean
  drive to ref 30% / norm 15% / calib 20% / dev 17.5% / holdout 17.5%. Stage-1 val drives (seen during v1
  development) may only enter ref/norm/calib/dev — **the new holdout contains only train-part drives never used
  before**.
* **Condition drives** (post-change segments of natural shifts): night, dawn/dusk, daytime rain, daytime snow
  (fog: none available in these parts). Train-part condition drives are split 50/50 by hash into
  cond_dev / cond_holdout; stage-1 val condition drives (old dev_post) only into cond_dev.

| partition | clean drives | from train / val |
|---|---|---|
| ref (reference banks, patch PCA basis) | 128 | 105 / 23 |
| norm (evidence normalisation, model fitting) | 52 | 43 / 9 |
| calib (thresholds only) | 66 | 52 / 14 |
| **dev** (method development) | 59 (58 long enough for streams) | 55 / 4 |
| **holdout** (final, untouched until a freeze tag) | 62 | 62 / 0 |

Condition drives: cond_dev night 125, dawn/dusk 30, rain 15, snow 20; cond_holdout night 81, dawn/dusk 24,
rain 10, snow 14. Unused val drives (non-clean, non-condition): 9.

## Streams (per dev or holdout clean drive A; τ = 100, H = 60, monitoring from t = 20)
* **clean**: A frames [0, 200) (PFA uses alarms in [20, 100)).
* **synthetic** (10): A[0,100) clean → A[100,160) corrupted; corruption types fog, low-light, rain, snow,
  motion blur, defocus blur, Gaussian noise, contrast, brightness, occlusion; severity cycled
  ((drive index + corruption index) mod 3) + 1 so every severity appears for every corruption.
* **natural** (4): A[0,100) → B[0,60) for B a drive of the partition's condition pool (night, dawn/dusk, rain,
  snow), B chosen by a fixed hash rotation.
* **hard negatives** (5; clean → clean, no environmental degradation): A[0,100) → B[0,60) with B a different
  clean drive of the SAME partition selected by a controlled factor:
  * `hn_random_scene` — any other drive;
  * `hn_road_geometry` — different scene type (city street / highway / residential);
  * `hn_traffic_density` — key-frame car count from the bottom tercile to the top tercile (or vice versa);
  * `hn_object_composition` — ≥ 3 pedestrians ↔ none;
  * `hn_camera_motion` — motion-proxy (median frame-difference over the drive) high ↔ low tercile.
* Dev: 58 clean, 580 synthetic, 232 natural, 290 hard-negative streams. Holdout: 62 / 620 / 248 / 310.

## Known limitations
* Condition labels are per-video (from one key frame); hard-negative factors come from key-frame annotations
  and a coarse motion proxy, not frame-accurate labels.
* Natural shifts and hard negatives are splices (abrupt cuts); they test condition-vs-content specificity, not
  the gradual onset of real weather changes.
* No foggy drives in these parts; fog is evaluated only synthetically.
