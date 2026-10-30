"""Stage-5 auxiliary result files derived from stored per-stream outputs (no re-fitting, no features needed):
  results/v5/hard_negative_per_post_drive.csv   per (HN category, post-change drive) alarm rate, KCM / MEWMA / block-2 k-NN
  results/v5/night_delay_<comparator>.csv       describe() of per-condition-drive mean night delays, KCM vs comparator (paired)
  results/v5/calibration_pool_ablation.json     ablation A4c: conformal thresholds on 200 random calibration subsets of
                                                12/20/33/66 drives, applied to the 62 holdout clean drives
  results/v5/acdc_dev_reused_per_recording.csv  ACDC development-reused (stage-4 cross-fitted, known protocol) per recording
NOTE: these four files were originally produced by short inline commands during stage 5 that were not saved as scripts.
This file re-implements them from their documented definitions; the clean-copy audit confirmed that all four
regenerate byte-identically to the stored originals (scripts/verify_results.py checks this).
usage: python scripts/v5/aux_results.py [--out results/v5]"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT)
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/v5")
from rmrm.data import load_streams  # noqa
from kcm import conformal  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/v5"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv("results/v5/dino/holdout_per_stream.csv", low_memory=False); d = d[d.alpha == 0.05].copy()
d["fa"] = d.false_alarm.astype(str) == "True"; d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
M3 = ["KCM", "MEWMA_multilayer", "g2_kNN_EWMA"]

# ---- hard negatives per post-change drive
post = {s.stream_id: s.segments[1].seq_id for s in load_streams("data/v2/streams.json") if s.split == "holdout" and s.kind == "hard_negative"}
hn = d[(d.kind == "hard_negative") & d.method.isin(M3)]  # streams with a pre-change alarm count as det=False here (descriptive file)
hn = hn.assign(post=lambda z: z.stream_id.map(post))
t = hn.groupby(["shift", "post", "method"]).det.mean().unstack("method")[M3].round(2).reset_index()
t.to_csv(f"{a.out}/hard_negative_per_post_drive.csv", index=False)

# ---- paired night delays (per condition drive, detected streams only)
for comp in ["MEWMA_multilayer", "g2_kNN_EWMA"]:
    z = d[(d.kind == "natural") & (d["shift"] == "night") & ~d.fa & d.method.isin(["KCM", comp])]
    z = z.assign(c=np.where(z.det, z.dl, 61.0))           # undetected night streams censored at 61 frames
    v = z.groupby(["cond_drive", "method"]).c.mean().unstack()[["KCM", comp]].dropna()
    v.describe().round(1).to_csv(f"{a.out}/night_delay_{comp}.csv")

# ---- A4c calibration-pool ablation
cal = json.load(open("results/v5/dino/calibration.json"))
out = {}; rng = np.random.default_rng(0)                  # one generator, methods in order KCM, MEWMA, block-2 k-NN
for m in M3:
    cm = np.asarray(cal["calib_maxima"][m]); hm = d[(d.method == m) & (d.kind == "clean")].pre_max.values
    out[m] = {}
    for n in [12, 20, 33, 66]:
        th, pf = [], []
        for _ in range(200 if n < len(cm) else 1):
            idx = rng.choice(len(cm), n, replace=False) if n < len(cm) else np.arange(len(cm))
            h = conformal(cm[idx], 0.05); th.append(h); pf.append(float(np.mean(hm > h)))
        th = np.asarray(th); fin = th[np.isfinite(th)]
        out[m][str(n)] = {"threshold_median": float(np.median(fin)) if len(fin) else None, "frac_infinite_threshold": float(np.mean(~np.isfinite(th))),
                          "threshold_iqr": [float(np.percentile(fin, 25)), float(np.percentile(fin, 75))] if len(fin) > 1 else None,
                          "holdout_pfa_mean": float(np.mean(pf)), "holdout_pfa_p90": float(np.percentile(pf, 90)), "holdout_pfa_max": float(np.max(pf))}
json.dump(out, open(f"{a.out}/calibration_pool_ablation.json", "w"), indent=1)

# ---- ACDC development-reused per recording (stage-4 cross-fitted known protocol; M2 == frozen KCM formulation)
ac = pd.read_csv("results/v4/acdc_per_stream.csv"); ac = ac[(ac.protocol == "K") & (ac.kind == "condition") & ac.method.isin(["M2", "MEWMA", "g2"])].copy()
ac["fa"] = ac.false_alarm.astype(str) == "True"; ac["dl"] = ac.delay.astype(float); ac["det"] = ~ac.fa & np.isfinite(ac.dl)
ac = ac[~ac.fa]
rows = []
for (m, sh, u), z in ac.groupby(["method", "shift", "unit"]):
    rows.append({"method": "KCM (stage-4 M2, identical formulation)" if m == "M2" else m, "shift": sh, "unit": u, "det": z.det.mean(),
                 "median_delay": float(np.median(np.where(z.det, z.dl, np.inf))), "n_streams": len(z)})
order = {"KCM (stage-4 M2, identical formulation)": 0, "MEWMA": 1, "g2": 2}
pd.DataFrame(rows).sort_values(["method", "shift", "unit"], key=lambda s: s.map(order) if s.name == "method" else s).to_csv(f"{a.out}/acdc_dev_reused_per_recording.csv", index=False)
print("written to", a.out)
