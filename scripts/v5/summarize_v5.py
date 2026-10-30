"""Stage-5 summaries, per-recording tables and statistics from results/v5/dino/holdout_per_stream.csv (+ backbones).
Writes the deliverable CSVs in the repository root and results/v5/stats.json."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, "src")
from rmrm.ccd import clopper_pearson  # noqa
rng = np.random.default_rng(0)
d = pd.read_csv("results/v5/dino/holdout_per_stream.csv", low_memory=False)
d["fa"] = d.false_alarm.astype(str) == "True"; d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
lab = {x["drive"]: x["rating"] for x in json.load(open("data/annotations/bdd_ratings.json"))["ratings"]}
d["label_audit"] = d.cond_drive.map(lab)
A = d[d.alpha == 0.05].copy()
CONDS = ["night", "dawn_dusk", "rain", "snow"]
HN = ["hn_random_scene", "hn_road_geometry", "hn_traffic_density", "hn_object_composition", "hn_camera_motion"]

def clusterboot(vals_by_unit, n=2000):
    u = list(vals_by_unit.values()); u = np.array(u, float)
    if len(u) == 0: return (np.nan, np.nan)
    bs = [u[rng.integers(0, len(u), len(u))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def summary(g):
    r = {}
    c = g[g.kind == "clean"]; x, n = int(c.fa.sum()), len(c); lo, hi = clopper_pearson(x, n)
    r.update(pfa=x / n, pfa_lo=lo, pfa_hi=hi, n_clean_drives=n)
    sy = g[(g.kind == "synthetic") & ~g.fa]
    r.update(syn_det=sy.det.mean(), syn_det10=(sy.det & (sy.dl <= 10)).mean(), syn_det20=(sy.det & (sy.dl <= 20)).mean(),
             syn_median_delay=float(np.median(np.where(sy.det, sy.dl, np.inf))) if len(sy) else np.nan, n_syn=len(sy))
    for sv in (1, 2, 3): r[f"syn_det_s{sv}"] = sy[sy.severity == sv].det.mean()
    for cor in sorted(sy["shift"].unique()): r[f"syn_{cor}"] = sy[sy["shift"] == cor].det.mean()
    na = g[(g.kind == "natural") & ~g.fa]
    for cnd in CONDS:
        z = na[na["shift"] == cnd]; per = z.groupby("cond_drive").det.mean()
        r[f"real_{cnd}"] = per.mean(); r[f"real_{cnd}_ci"] = clusterboot(per.to_dict()); r[f"real_{cnd}_n_drives"] = len(per)
        r[f"real_{cnd}_median_delay"] = float(np.median(np.where(z.det, z.dl, np.inf))) if len(z) else np.nan
        r[f"real_{cnd}_det20"] = (z.det & (z.dl <= 20)).groupby(z.cond_drive).mean().mean()
        zv = z[z.label_audit == "V"]; pv = zv.groupby("cond_drive").det.mean()
        r[f"real_{cnd}_visible"] = pv.mean() if len(pv) else np.nan; r[f"real_{cnd}_visible_n"] = len(pv)
    per = na.groupby("cond_drive").det.mean(); r["real_pooled"] = per.mean(); r["real_pooled_ci"] = clusterboot(per.to_dict())
    hn = g[(g.kind == "hard_negative") & ~g.fa]
    r["hn_pooled"] = hn.det.mean(); r["hn_pooled_ci"] = clusterboot(hn.groupby("unit").det.mean().to_dict())
    for h in HN: r[h] = hn[hn["shift"] == h].det.mean()
    r["hn_worst"] = max(r[h] for h in HN)
    return r

S = {m: summary(g) for m, g in A.groupby("method") if m not in ("KCM_LOFO", "KCM_windowcal")}
SS = pd.DataFrame(S).T
# windowed-calibration ablation (clean only) and LOFO stress test
w = A[A.method == "KCM_windowcal"]; x = int(w.fa.sum()); lo, hi = clopper_pearson(x, len(w))
SS.loc["KCM_windowcal", ["pfa", "pfa_lo", "pfa_hi", "n_clean_drives"]] = [x / len(w), lo, hi, len(w)]
lofo = A[(A.method == "KCM_LOFO") & ~A.fa].groupby("family").det.mean()
SS.to_csv("results/CONFIRMATORY_SUMMARY_RESULTS.csv")
base = ["MEWMA_multilayer", "g2_kNN_EWMA", "MEWMA_final", "deepkNN_EWMA_final", "T2_EWMA_final", "T2_CUSUM_final", "MMD_final"]
cols = ["pfa", "pfa_lo", "pfa_hi", "syn_det", "syn_median_delay", "real_night", "real_dawn_dusk", "real_rain", "real_snow", "real_pooled", "hn_pooled", "hn_worst"]
SS.loc[["KCM"] + base, cols].to_csv("results/BASELINE_COMPARISON.csv")
abl = ["KCM_block12", "KCM_block4", "KCM", "KCM+ST", "KCM_windowcal", "KCM_block2", "KCM_block6", "KCM_block8", "KCM_block10", "ST"]
SS.loc[abl, cols + ["syn_occlusion"]].to_csv("results/ABLATION_RESULTS.csv")
hn = A[(A.kind == "hard_negative") & ~A.fa]
H = hn.groupby(["method", "shift"]).det.agg(["mean", "size"]).unstack("shift")
H.to_csv("results/HARD_NEGATIVE_RESULTS.csv")
# matched-FA curves
cur = []
for (m, a), g in d[~d.method.isin(["KCM_LOFO", "KCM_windowcal"])].groupby(["method", "alpha"]):
    s = summary(g); cur.append({"method": m, "alpha": a, "holdout_pfa": s["pfa"], "real_pooled": s["real_pooled"], "real_night": s["real_night"],
                                "syn_det": s["syn_det"], "syn_median_delay": s["syn_median_delay"], "hn_pooled": s["hn_pooled"]})
pd.DataFrame(cur).to_csv("results/v5/matched_fa_curves.csv", index=False)
# per-recording table (alpha = 0.05)
pr = []
for m, g in A.groupby("method"):
    for u, z in g[g.kind == "clean"].groupby("unit"): pr.append({"method": m, "unit_type": "clean_drive", "unit": u, "metric": "false_alarm", "value": float(z.fa.any())})
    sy = g[(g.kind == "synthetic") & ~g.fa]
    for u, z in sy.groupby("unit"):
        pr.append({"method": m, "unit_type": "pre_change_drive", "unit": u, "metric": "synthetic_detection", "value": z.det.mean()})
        pr.append({"method": m, "unit_type": "pre_change_drive", "unit": u, "metric": "synthetic_mean_censored_delay", "value": np.where(z.det, z.dl, 61).mean()})
    for (u, c), z in g[(g.kind == "natural") & ~g.fa].groupby(["cond_drive", "shift"]):
        pr.append({"method": m, "unit_type": f"condition_drive:{c}:{lab.get(u)}", "unit": u, "metric": "real_detection", "value": z.det.mean()})
        dd = z.dl[z.det]; pr.append({"method": m, "unit_type": f"condition_drive:{c}:{lab.get(u)}", "unit": u, "metric": "real_median_delay", "value": float(dd.median()) if len(dd) else np.inf})
    for u, z in g[(g.kind == "hard_negative") & ~g.fa].groupby("unit"):
        pr.append({"method": m, "unit_type": "pre_change_drive", "unit": u, "metric": "hard_negative_alarm_rate", "value": z.det.mean()})
pd.DataFrame(pr).to_csv("results/CONFIRMATORY_RESULTS_PER_RECORDING.csv", index=False)
# ---------------- paired statistics: KCM vs baselines (units = drives)
def paired(m, ref, endpoint):
    g = A[A.method.isin([m, ref])]
    if endpoint == "synthetic_delay":
        z = g[(g.kind == "synthetic")]; bad = z[z.fa].groupby("unit").method.apply(set)
        z = z[~z.unit.isin(bad.index)]; v = z.assign(c=np.where(z.det, z.dl, 61)).groupby(["unit", "method"]).c.mean().unstack()
    elif endpoint.startswith("real"):
        z = g[(g.kind == "natural") & ~g.fa]
        if endpoint != "real_pooled": z = z[z["shift"] == endpoint[5:]]
        v = z.groupby(["cond_drive", "method"]).det.mean().unstack()
    elif endpoint == "hard_negative":
        z = g[(g.kind == "hard_negative") & ~g.fa]; v = z.groupby(["unit", "method"]).det.mean().unstack()
    v = v.dropna(); diff = (v[m] - v[ref]).values
    if len(diff) < 5 or np.allclose(diff, 0): return {"n": len(diff), "mean_diff": float(diff.mean()) if len(diff) else np.nan}
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    try: p = float(wilcoxon(diff).pvalue)
    except ValueError: p = np.nan
    nz = diff[diff != 0]; rk = pd.Series(np.abs(nz)).rank().values
    rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else 0.0
    return {"n": len(diff), "mean_diff": float(diff.mean()), "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
            "wilcoxon_p": p, "rank_biserial": rb, "frac_kcm_better": float(np.mean(diff < 0) if endpoint == "synthetic_delay" else np.mean(diff > 0))}
ST = []
for ep in ["synthetic_delay", "real_pooled", "real_night", "real_dawn_dusk", "real_rain", "real_snow", "hard_negative"]:
    for ref in base:
        ST.append({"endpoint": ep, "comparator": ref, **paired("KCM", ref, ep)})
st = pd.DataFrame(ST)
for ep, g in st.groupby("endpoint"):                       # Holm within endpoint
    p = g.wilcoxon_p.fillna(1).values; o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    st.loc[g.index, "holm_p"] = adj
st.to_csv("results/v5/paired_statistics.csv", index=False)
json.dump({"lofo_family_detection_KCM": lofo.to_dict()}, open("results/v5/lofo_holdout.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(SS.loc[["KCM"] + base + abl[:2] + ["KCM+ST", "KCM_windowcal"], cols].round(3).to_string())
print("LOFO (holdout, frozen procedure):", lofo.round(3).to_dict())
print(st.round(4).to_string())
