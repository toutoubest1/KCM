"""Requires extracted BDD100K/ACDC features (data/features_v2/dinov2_s14). Re-fits the frozen KCM direction with the
frozen training procedure (exactly the data selection of scripts/v5/run_confirmatory.py) and checks that it equals the
stored frozen model results/v5/dino/kcm_model.pkl (coefficients, normalisation) and reproduces h = 8.565."""
import json, os, pickle, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); os.chdir(ROOT)
for p in ["scripts/v5", "scripts/v2", "scripts/v3", "src"]: sys.path.insert(0, p)
from kcm import KCM, drive_max, conformal  # noqa
from run_v2 import Store  # noqa
from rmrm.data import load_index  # noqa
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
store = Store(cfg["feature_root"]); FD = os.path.join(cfg["feature_root"], "clips")
corr = [f for f in sorted(os.listdir(FD)) if any(f.startswith(k.replace("/", "__") + "__") for k in ns["normT"]) and not f.endswith("__clean.npz")]
pos = np.concatenate([np.load(os.path.join(FD, f))["glob"] for f in corr])
neg = np.concatenate([store.clip(f"{k}__clean")["glob"] for k in ns["normT"]])
apos, aneg = [], []
for s in load_index("data/v3/acdc_index.json"):
    g = np.load(os.path.join(FD, f"{s.seq_id}__clean".replace("/", "__") + ".npz"))["glob"]
    (apos if s.seq_id.endswith("_adv") else aneg).append(g)
k = KCM().fit([(pos, 1, False), (neg, 0, False), (np.concatenate(apos), 1, False), (np.concatenate(aneg), 0, False)])
k.normalise([store.clip(f"{d}__clean")["glob"] for d in ns["normS"]])
ref = pickle.load(open("results/v5/dino/kcm_model.pkl", "rb"))["KCM"]
dc = float(np.abs(k.lr.coef_ - ref.lr.coef_).max()); dn = (abs(k.mu - ref.mu), abs(k.sd - ref.sd))
cal = [drive_max(k.stat(store.clip(f"{d}__clean")["glob"][:100])) for d in sorted(x for x, v in part.items() if v == "calib")]
h = conformal(cal, 0.05)
print("n_train", k.n_train, "max|coef diff|", dc, "norm diff", dn, "h", h)
ok = k.n_train == ref.n_train and dc < 1e-6 and max(dn) < 1e-6 and abs(h - 8.56533953362844) < 1e-6
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
