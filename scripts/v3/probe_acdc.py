"""Exploratory diagnostic D1b (added after D1, before any method change): is real-condition information LINEARLY
present in the frozen representation at all? Logistic probes (C=0.1, fixed a priori) on the L2-normalised
[CLS, mean-patch] token of each block, trained on NON-dev ACDC recordings (adverse frames of condition c vs the
normal frames of all non-dev units) and evaluated on dev recordings (adverse vs their paired normal frames).
No dev frame is used for fitting."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")
from rmrm.ccd import l2n  # noqa: E402
cfg = json.load(open("configs/v3/acdc.json")); part = json.load(open(cfg["partitions"]))
def g(sid):
    z = np.load(os.path.join(cfg["feature_root"], "clips", f"{sid}__clean".replace("/", "__") + ".npz"))["glob"].astype(np.float32)
    return z
F = {}
for u in part:
    for k in ("adv", "ref"): F[f"{u}_{k}"] = g(f"{u}_{k}")
def feat(x, b):   # x [T,6,2,384]; b = block index 0..5 or 'all'
    if b == "all": return np.concatenate([np.concatenate([l2n(x[:, i, 0]), l2n(x[:, i, 1])], 1) for i in range(6)], 1)
    return np.concatenate([l2n(x[:, b, 0]), l2n(x[:, b, 1])], 1)
blocks = [2, 4, 6, 8, 10, 12]
dev = [u for u, r in part.items() if r == "dev"]; nondev = [u for u, r in part.items() if r != "dev"]
rows = []
for bi, b in list(enumerate(blocks)) + [("all", "all")]:
    Xn = np.concatenate([feat(F[f"{u}_ref"], bi) for u in nondev])
    for c in cfg["conditions"]:
        Xa = np.concatenate([feat(F[f"{u}_adv"], bi) for u in nondev if u.split("/")[1].startswith(c)])
        X = np.concatenate([Xa, Xn]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xn))]
        sc = StandardScaler().fit(X); m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        du = [u for u in dev if u.split("/")[1].startswith(c)]
        A = np.concatenate([feat(F[f"{u}_adv"], bi) for u in du]); N = np.concatenate([feat(F[f"{u}_ref"], bi) for u in du])
        sa, sn = m.decision_function(sc.transform(A)), m.decision_function(sc.transform(N))
        # specificity check: probe score on normal frames of OTHER dev recordings (scene change only)
        No = np.concatenate([feat(F[f"{u}_ref"], bi) for u in dev if u not in du]); so = m.decision_function(sc.transform(No))
        per = {}
        for u in du:
            a_ = m.decision_function(sc.transform(feat(F[f"{u}_adv"], bi))); n_ = m.decision_function(sc.transform(feat(F[f"{u}_ref"], bi)))
            per[u.split("/")[1]] = round(roc_auc_score(np.r_[np.ones(len(a_)), np.zeros(len(n_))], np.r_[a_, n_]), 3)
        rows.append({"block": b, "condition": c, "n_train_adv": len(Xa), "n_train_norm": len(Xn),
                     "dev_auroc_paired": roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn]),
                     "dev_auroc_vs_other_normal": roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(so))], np.r_[sa, so]),
                     "per_recording": json.dumps(per)})
df = pd.DataFrame(rows); os.makedirs("results/v3/diag", exist_ok=True); df.to_csv("results/v3/diag/d1b_linear_probe.csv", index=False)
pd.set_option("display.width", 250)
print(df.pivot(index="block", columns="condition", values="dev_auroc_paired").round(3).to_string())
print(df.pivot(index="block", columns="condition", values="dev_auroc_vs_other_normal").round(3).to_string())
for _, r in df[df.block.astype(str).isin(["2", "6", "all"])].iterrows(): print(r.block, r.condition, r.per_recording)

# cross-condition transfer of the probes (block 6 and block 2): AUROC of probe(c_train) on dev adverse frames of c_test vs dev normal
xr = []
for bi, b in [(0, 2), (2, 6)]:
    Xn = np.concatenate([feat(F[f"{u}_ref"], bi) for u in nondev]); Ndev = np.concatenate([feat(F[f"{u}_ref"], bi) for u in dev])
    for c in cfg["conditions"]:
        Xa = np.concatenate([feat(F[f"{u}_adv"], bi) for u in nondev if u.split("/")[1].startswith(c)])
        X = np.concatenate([Xa, Xn]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xn))]
        sc = StandardScaler().fit(X); m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        sn = m.decision_function(sc.transform(Ndev))
        for c2 in cfg["conditions"]:
            A = np.concatenate([feat(F[f"{u}_adv"], bi) for u in dev if u.split("/")[1].startswith(c2)]); sa = m.decision_function(sc.transform(A))
            xr.append({"block": b, "train": c, "test": c2, "auroc": roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])})
xr = pd.DataFrame(xr); xr.to_csv("results/v3/diag/d1b_probe_cross_condition.csv", index=False)
for b in (2, 6): print("block", b); print(xr[xr.block == b].pivot(index="train", columns="test", values="auroc").round(3).to_string())
