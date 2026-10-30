import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from rmrm.ccd import clopper_pearson
d = pd.read_csv("results/v4/acdc_per_stream.csv"); d["fa"] = d.false_alarm.astype(bool); d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
pd.set_option("display.width", 250)
out = []
for (p, m), g in d.groupby(["protocol", "method"]):
    c = g[g.kind == "clean"]; x, n = int(c.fa.sum()), len(c); lo, hi = clopper_pearson(x, n)
    hn = g[(g.kind == "hard_negative") & ~g.fa]; cd = g[(g.kind == "condition") & ~g.fa]
    r = {"protocol": p, "method": m, "pfa": x / n, "pfa_lo": lo, "pfa_hi": hi, "n_clean": n, "hn": hn.det.mean(), "n_hn": len(hn)}
    for s in ["fog", "night", "rain", "snow"]:
        z = cd[cd["shift"] == s]
        if len(z): r[s] = z.det.mean(); r[f"{s}_n"] = len(z); r[f"{s}_rec"] = z.unit.nunique()
    perrec = cd.groupby("unit").det.mean(); r["frac_rec_ge_.5"] = (perrec >= .5).mean(); r["n_rec"] = len(perrec)
    r["median_delay"] = np.median(np.where(cd.det, cd.dl, np.inf)) if len(cd) else np.nan
    out.append(r)
S = pd.DataFrame(out); S.to_csv("results/v4/acdc_summary.csv", index=False)
for p in ["K", "F1_illumination", "F4_weather"]:
    print("==", p); print(S[S.protocol == p].drop(columns="protocol").set_index("method").round(3).to_string())
k = d[(d.protocol == "K") & (d.kind == "condition") & ~d.fa]
print(k[k.method.isin(["V1", "M2", "M1", "g2", "MEWMA"])].groupby(["method", "unit"]).det.mean().unstack().round(2).to_string())
