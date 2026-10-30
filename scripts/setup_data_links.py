"""Create data/raw/<name> symlinks from configs/paths.json (central dataset-path configuration).
Expected layout of each linked folder (produced by the scripts in scripts/data_prep/):
  bdd_track_train/<video_id>/<video_id>-0000001.jpg ...   (480x270 JPEG, BDD100K MOT-2020 train parts 1-4)
  bdd_track_val/<video_id>/<video_id>-0000001.jpg ...     (480x270 JPEG, BDD100K MOT-2020 val)
  acdc/<fog|night|rain|snow>/<train|val|test>[_ref]/<GOPRxxxx>/<name>.jpg (480x270 JPEG)
  radiate/<sequence>/<00000>.jpg + meta.json             (zed_left, 5 Hz, 480x270 JPEG)
Weights: set RMRM_WEIGHTS (or keep the default <repo>/weights) - see weights/README.md."""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); os.chdir(ROOT)
cfg = json.load(open("configs/paths.json")); os.makedirs("data/raw", exist_ok=True)
for k in ["bdd_track_train", "bdd_track_val", "acdc", "radiate"]:
    src = cfg[k] if os.path.isabs(cfg[k]) else os.path.join(ROOT, cfg[k]); dst = os.path.join("data/raw", k)
    if not os.path.exists(src): print(f"not found (skipped): {k} -> {src}"); continue
    if os.path.lexists(dst): print(f"exists: {dst}"); continue
    os.symlink(src, dst); print(f"linked {dst} -> {src}")
