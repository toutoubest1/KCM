"""Pre-freeze, DEV-ONLY selection of the 'best single early/mid layer' ablation (b* in {2,4,6,8}):
rule fixed in advance = highest BDD-dev known synthetic detection within 60 frames (ties -> earlier block).
Uses only training (normT, ACDC), normalisation (normS), calibration (calib) and BDD dev streams."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "scripts/v2"); sys.path.insert(0, "src")
from kcm import KCM, phi, drive_max, conformal  # noqa
from run_v2 import Store, assemble  # noqa
from rmrm.data import load_streams, load_index  # noqa
from rmrm.evaluation import stream_outcome  # noqa
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
store = Store(cfg["feature_root"]); fd = os.path.join(cfg["feature_root"], "clips")
corr = [f for f in sorted(os.listdir(fd)) if any(f.startswith(k.replace("/", "__") + "__") for k in ns["normT"]) and not f.endswith("__clean.npz")]
G_pos = np.concatenate([np.load(os.path.join(fd, f))["glob"] for f in corr]); G_neg = np.concatenate([store.clip(f"{k}__clean")["glob"] for k in ns["normT"]])
A_pos, A_neg = [], []
for s in load_index("data/v3/acdc_index.json"):
    g = np.load(os.path.join(fd, f"{s.seq_id}__clean".replace("/", "__") + ".npz"))["glob"]
    (A_pos if s.seq_id.endswith("_adv") else A_neg).append(g)
A_pos, A_neg = np.concatenate(A_pos), np.concatenate(A_neg)
normS = [store.clip(f"{k}__clean")["glob"] for k in ns["normS"]]; cal = [store.clip(f"{k}__clean")["glob"][:100] for k in sorted(k for k, v in part.items() if v == "calib")]
dev = [s for s in load_streams(cfg["streams"]) if s.split == "dev" and s.kind == "synthetic"]
DG = [assemble(store, None, s, "clip")["glob"] for s in dev]
res = {}
for b, i in [(2, 0), (4, 1), (6, 2), (8, 3)]:
    m = KCM(idx=(i,)).fit([(G_pos, 1, False), (G_neg, 0, False), (A_pos, 1, False), (A_neg, 0, False)]).normalise(normS)
    h = conformal([drive_max(m.stat(g)) for g in cal])
    det = []
    for s, g in zip(dev, DG):
        r = stream_outcome(m.stat(g), h, s.tau)
        if not r["false_alarm"]: det.append(np.isfinite(r["delay"]))
    res[b] = float(np.mean(det)); print(b, res[b], flush=True)
best = max(sorted(res), key=lambda b: (res[b], -b))
json.dump({"dev_synthetic_detection": res, "selected_block": best}, open("results/v5/single_layer_selection_dev.json", "w"), indent=1)
print("selected", best)
