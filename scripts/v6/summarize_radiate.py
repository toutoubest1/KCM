"""Summaries of the RADIATE validation (reads results/v6/<backbone>/per_stream.csv; no re-fitting).
Writes RADIATE_SUMMARY_RESULTS.csv, RADIATE_PER_SEQUENCE_RESULTS.csv, RADIATE_PER_SESSION_RESULTS.csv,
RADIATE_HARD_NEGATIVE_RESULTS.csv, RADIATE_SUPPORT_RULE.csv, results/v6/summary.json"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT); sys.path.insert(0, "src")
from rmrm.ccd import clopper_pearson  # noqa
BK = [b for b in ["dinov2_s14", "deit_s16", "clip_b16_openai"] if os.path.exists(f"results/v6/{b}/per_stream.csv")]
d = pd.concat([pd.read_csv(f"results/v6/{b}/per_stream.csv") for b in BK])
d["fa"] = d.false_alarm.astype(str) == "True"; d["dl"] = d.delay.astype(float); d["det"] = ~d.fa & np.isfinite(d.dl)
SES = lambda s: s.rsplit("_", 1)[0]
CONDS = ["fog", "rain", "snow", "night"]; rng = np.random.default_rng(0)
def seqboot(z):
    per = z.groupby("post_seq").det.agg(["sum", "size"])
    if len(per) < 2: return (np.nan, np.nan)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(per), len(per)); bs.append(per["sum"].values[i].sum() / per["size"].values[i].sum())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
summ, seq, ses, hnr, sup = [], [], [], [], []
for (b, m, a), g in d.groupby(["backbone", "method", "alpha"]):
    c = g[g.kind == "clean"]; x, n = int(c.fa.sum()), len(c); lo, hi = clopper_pearson(x, n)
    r = dict(backbone=b, method=m, alpha=a, pfa=x / n, pfa_lo=lo, pfa_hi=hi, n_clean_chunks=n, n_clean_fa=x,
             n_clean_sessions_with_fa=c[c.fa].pre_session.nunique())
    hn = g[(g.kind == "hard_negative") & ~g.fa]; xh = int(hn.det.sum()); lo2, hi2 = clopper_pearson(xh, len(hn))
    r.update(hn=xh / max(len(hn), 1), hn_lo=lo2, hn_hi=hi2, n_hn=len(hn), n_hn_prefa=int(((g.kind == "hard_negative") & g.fa).sum()))
    for cnd in CONDS:
        z = g[(g.kind == "condition") & (g["shift"] == cnd) & ~g.fa]
        r[f"{cnd}_det"] = z.det.mean(); r[f"{cnd}_n"] = len(z); r[f"{cnd}_n_prefa"] = int(((g.kind == "condition") & (g["shift"] == cnd) & g.fa).sum())
        r[f"{cnd}_ci"] = seqboot(z); r[f"{cnd}_det20"] = (z.det & (z.dl <= 20)).mean()
        r[f"{cnd}_median_delay"] = float(np.median(np.where(z.det, z.dl, np.inf))) if len(z) else np.nan
        r[f"{cnd}_seqs_ge_half"] = int((z.groupby("post_seq").det.mean() >= 0.5).sum())
    zc = g[(g.kind == "condition") & ~g.fa]; r["pooled_det"] = zc.det.mean(); r["pooled_ci"] = seqboot(zc)
    summ.append(r)
    for (cnd, s), z in g[(g.kind == "condition") & ~g.fa].groupby(["shift", "post_seq"]):
        seq.append(dict(backbone=b, method=m, alpha=a, condition=cnd, sequence=s, session=SES(s), n_windows=len(z), detection=z.det.mean(),
                        median_delay=float(np.median(np.where(z.det, z.dl, np.inf))), det_within_20=(z.det & (z.dl <= 20)).mean(),
                        post_max_median=float(z.post_max.median())))
    for s, z in c.groupby(c.pre.str.split("#").str[0]):
        seq.append(dict(backbone=b, method=m, alpha=a, condition="clean_T0", sequence=s, session=SES(s), n_windows=len(z), false_alarm_rate=z.fa.mean()))
    for (cnd, s), z in g[(g.kind == "condition") & ~g.fa].groupby(["shift", "post_session"]):
        ses.append(dict(backbone=b, method=m, alpha=a, condition=cnd, session=s, n_windows=len(z), n_sequences=z.post_seq.nunique(),
                        detection=z.det.mean(), median_delay=float(np.median(np.where(z.det, z.dl, np.inf)))))
    for sh, z in hn.groupby("shift"):
        hnr.append(dict(backbone=b, method=m, alpha=a, pair=sh, n=len(z), alarm_rate=z.det.mean()))
    if a == 0.05:
        cal_ok = (lo <= 0.05)  # CI contains 0.05 or lies below it
        for cnd in CONDS:
            sup.append(dict(backbone=b, method=m, condition=cnd, pfa=x / n, pfa_ci=f"[{lo:.3f}, {hi:.3f}]", crit1_calibration=bool(cal_ok),
                            hn=r["hn"], crit2_hn_le_0p10=bool(r["hn"] <= 0.10), detection=r[f"{cnd}_det"],
                            crit3_det_ge_hn_plus_0p50=bool(r[f"{cnd}_det"] >= r["hn"] + 0.50),
                            supported=bool(cal_ok and r["hn"] <= 0.10 and r[f"{cnd}_det"] >= r["hn"] + 0.50)))
S = pd.DataFrame(summ); S.to_csv("results/RADIATE_SUMMARY_RESULTS.csv", index=False)
pd.DataFrame(seq).to_csv("results/RADIATE_PER_SEQUENCE_RESULTS.csv", index=False)
pd.DataFrame(ses).to_csv("results/RADIATE_PER_SESSION_RESULTS.csv", index=False)
pd.DataFrame(hnr).to_csv("results/RADIATE_HARD_NEGATIVE_RESULTS.csv", index=False)
U = pd.DataFrame(sup); U.to_csv("results/RADIATE_SUPPORT_RULE.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["backbone", "method", "pfa", "pfa_lo", "pfa_hi", "n_clean_fa", "hn", "n_hn"] + sum([[f"{c}_det", f"{c}_n", f"{c}_median_delay"] for c in CONDS], []) + ["pooled_det"]
print(S[S.alpha == 0.05][cols].round(3).to_string(index=False))
print(U[U.method == "KCM"].round(3).to_string(index=False))
