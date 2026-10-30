import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from rmrm.ccd import clopper_pearson
rows = []
srcs = {"dinov2_s14": ("results/v5/dino/holdout_per_stream.csv", {"KCM": "KCM", "KCM_block12": "KCM_block12", "MEWMA_multilayer": "MEWMA_multilayer", "MEWMA_final": "MEWMA_final", "deepkNN_EWMA_final": "deepkNN_EWMA_final"})}
for bb in ["deit_s16", "clip_b16_openai"]:
    f = f"results/v5/backbone_{bb}/holdout_per_stream.csv"
    if os.path.exists(f): srcs[bb] = (f, None)
for bb, (f, mp) in srcs.items():
    d = pd.read_csv(f, low_memory=False); d = d[d.alpha == 0.05]
    if mp: d = d[d.method.isin(mp)]
    d["fa"] = d.false_alarm.astype(str) == "True"; d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
    for m, g in d.groupby("method"):
        c = g[g.kind == "clean"]; x, n = int(c.fa.sum()), len(c); lo, hi = clopper_pearson(x, n)
        sy = g[(g.kind == "synthetic") & ~g.fa]; na = g[(g.kind == "natural") & ~g.fa]; hn = g[(g.kind == "hard_negative") & ~g.fa]
        r = {"backbone": bb, "method": m, "pfa": x / n, "pfa_lo": lo, "pfa_hi": hi, "syn_det": sy.det.mean(),
             "syn_median_delay": float(np.median(np.where(sy.det, sy.dl, np.inf))), "hn_pooled": hn.det.mean(), "hn_worst": hn.groupby("shift").det.mean().max()}
        for cnd in ["night", "dawn_dusk", "rain", "snow"]:
            r[f"real_{cnd}"] = na[na["shift"] == cnd].groupby("cond_drive").det.mean().mean()
        r["real_pooled"] = na.groupby("cond_drive").det.mean().mean()
        for cor in ["fog", "rain", "snow", "occlusion"]: r[f"syn_{cor}"] = sy[sy["shift"] == cor].det.mean()
        rows.append(r)
B = pd.DataFrame(rows); B.to_csv("results/BACKBONE_COMPARISON.csv", index=False)
pd.set_option("display.width", 250); print(B.round(3).to_string(index=False))
