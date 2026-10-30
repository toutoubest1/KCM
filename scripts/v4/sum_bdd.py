import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from rmrm.ccd import clopper_pearson
D = sys.argv[1] if len(sys.argv) > 1 else "results/v4"
d = pd.read_csv(f"{D}/bdd_per_stream.csv", low_memory=False)
d["fa"] = d.false_alarm.astype(str) == "True"; d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
K = d[d.protocol == "known"]
rows = []
for m, g in K.groupby("method"):
    c = g[g.kind == "clean"]; x, n = int(c.fa.sum()), len(c); lo, hi = clopper_pearson(x, n)
    sy = g[(g.kind == "synthetic") & ~g.fa]; hn = g[(g.kind == "hard_negative") & ~g.fa]; na = g[(g.kind == "natural") & ~g.fa]
    r = {"method": m, "pfa": x / n, "pfa_lo": lo, "pfa_hi": hi, "syn_det": sy.det.mean(),
         "syn_median_delay": np.median(np.where(sy.det, sy.dl, np.inf)), "hn": hn.det.mean(), "hn_worst": hn.groupby("shift").det.mean().max()}
    for c_ in ["dawn_dusk", "night", "rain", "snow"]: r[f"nat_{c_}"] = na[na["shift"] == c_].det.mean()
    for f in sorted(g.family.dropna().unique()): r[f"known_{f}"] = sy[sy.family == f].det.mean()
    rows.append(r)
KS = pd.DataFrame(rows).set_index("method"); print("== BDD KNOWN"); print(KS.round(3).to_string())
L = d[d.protocol == "lofo"]
lo = L[(L.kind == "synthetic") & ~L.fa].groupby(["fold", "method"]).det.mean().unstack()
print("== BDD LOFO held-out family detection"); print(lo.round(3).to_string()); print("mean:"); print(lo.mean().round(3).to_string())
lh = L[(L.kind == "hard_negative") & ~L.fa].groupby(["fold", "method"]).det.mean().unstack(); print("== LOFO HN"); print(lh.round(3).to_string())
lp = L[L.kind == "clean"].groupby(["fold", "method"]).fa.mean().unstack(); print("== LOFO PFA"); print(lp.round(3).to_string())
ld = L[(L.kind == "synthetic") & ~L.fa].groupby("method").apply(lambda g: np.median(np.where(g.det, g.dl, np.inf)))
print("== LOFO pooled median delay"); print(ld.to_string())
KS.to_csv(f"{D}/bdd_known_summary.csv"); lo.to_csv(f"{D}/bdd_lofo_family.csv")
