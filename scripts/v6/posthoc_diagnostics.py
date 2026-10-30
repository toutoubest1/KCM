"""POST-HOC, DESCRIPTIVE diagnostics of the RADIATE result (computed after the results were seen; they do NOT
change any pre-registered outcome or the classification). Reads stored per-stream results and calibration maxima."""
import json, os, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT); out = {}
for b in ["dinov2_s14", "deit_s16", "clip_b16_openai"]:
    cal = json.load(open(f"results/v6/{b}/calibration.json")); h = cal["thresholds"]["KCM"]["0.05"]
    m = pd.Series(cal["calib_maxima"]["KCM"], index=[u.split("#")[0] for u in cal["calib_units"]])
    d = pd.read_csv(f"results/v6/{b}/per_stream.csv"); d = d[(d.alpha == 0.05) & (d.method == "KCM")]
    d["fa"] = d.false_alarm.astype(str) == "True"; d["det"] = ~d.fa & np.isfinite(d.delay.astype(float))
    c = d[d.kind == "clean"]; t0 = c.groupby(c.pre.str.split("#").str[0]).pre_max.agg(["max", "median"])
    hn = d[(d.kind == "hard_negative") & ~d.fa]; nr = hn[~hn.pre.str.startswith("rural") & ~hn.post.str.startswith("rural")]
    cc = c[~c.pre.str.startswith("rural")]; cond = d[(d.kind == "condition") & ~d.fa]
    out[b] = {"threshold": h, "calibration_session_max": m.groupby(level=0).max().round(2).to_dict(),
              "calibration_session_median": m.groupby(level=0).median().round(2).to_dict(),
              "T0_session_max": t0["max"].round(2).to_dict(), "T0_session_median": t0["median"].round(2).to_dict(),
              "excl_rural_1_1": {"pfa": f"{int(cc.fa.sum())}/{len(cc)}", "hn": f"{int(nr.det.sum())}/{len(nr)}",
                                 "condition_det_with_non_rural_prechange": cond[~cond.pre.str.startswith("rural")].groupby("shift").det.mean().round(3).to_dict()}}
from scipy.stats import f_oneway
for b in out:
    cal = json.load(open(f"results/v6/{b}/calibration.json"))
    m = pd.Series(cal["calib_maxima"]["KCM"], index=[u.split("#")[0] for u in cal["calib_units"]]); g = m.groupby(level=0)
    w, bt = float(g.var().mean()), float(g.mean().var())
    out[b]["calibration_variance_decomposition"] = {"within_session_var": w, "between_session_mean_var": bt, "icc_approx": bt / (bt + w),
                                                     "anova_p": float(f_oneway(*[v.values for _, v in g]).pvalue)}
json.dump(out, open("results/v6/posthoc_diagnostics.json", "w"), indent=1); print(json.dumps(out, indent=1))
