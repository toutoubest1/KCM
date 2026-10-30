"""Fit the stage-2 system on BDD exactly as in scripts/v2/run_v2.py (ref/norm partitions; LR on corrupted
norm frames) and pickle it for stage 3 (results/v3/bdd_system.pkl)."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v2"))
from system3 import fit_bdd  # noqa: E402
from run_v2 import Store  # noqa: E402

cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"]))
store = Store(cfg["feature_root"]); ids = lambda p: sorted(k for k, v in part.items() if v == p)

def lr_data_fn(ev, scorer):
    pos = [f[:-4].replace("__", "/", 1) for f in sorted(os.listdir(os.path.join(cfg["feature_root"], "clips")))
           if any(f.startswith(k.replace("/", "__") + "__") for k in ids("norm")) and not f.endswith("__clean.npz")]
    Xp = np.concatenate([ev.z(store.scores(c, scorer)) for c in pos]); Xn = ev.norm_Z
    print("LR data", len(Xp), len(Xn), flush=True)
    return np.concatenate([Xp, Xn]), np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]

sysm, ref_glob = fit_bdd(cfg, store, part, lr_data_fn)
os.makedirs("results/v3", exist_ok=True)
pickle.dump(sysm, open("results/v3/bdd_system.pkl", "wb"))
for n, d in sysm.dets.items():
    if hasattr(d, "lr"):
        print(n, "coef", dict(zip([sysm.ev.names[i] for i in d.ic], np.round(d.lr.coef_[0], 3))), "beta", getattr(d, "beta", None))
print("joint normalisers", sysm.joint_s)
