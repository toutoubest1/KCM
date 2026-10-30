"""Threshold calibration on clean streams and online-detection metrics.

False-alarm criterion (primary)

PFA_M(h) = P( sup_{t<M} stat_t > h ) for a detector started on a clean stream,
M = pre-change length of the evaluation streams.  h is calibrated EXCLUSIVELY on
clean calibration sequences (disjoint from the reference, normalisation and
evaluation sequences): every calibration sequence contributes windows of length M
starting at offsets 0, s, 2s, ... (detector restarted at each window start);
h_alpha = the (1-alpha) empirical quantile of the per-window maxima (with the
conservative "higher" interpolation).  Windows from one sequence are correlated;
this only affects the precision of the estimate, not its unbiasedness.

Delay convention: d = t_alarm - tau + 1 = number of post-change frames observed
at alarm (d >= 1).  No alarm within the post-change horizon H -> miss (d = inf).
An alarm at t < tau is a false alarm; such streams are excluded from conditional
delays (CADD = E[d | no false alarm]) and reported through PFA.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def window_maxima(stat_fn, bundles, M, stride):
    """stat_fn(bundle_slice) -> stat trajectory. Returns array of per-window maxima."""
    out = []
    for b in bundles:
        T = len(b["glob"])
        starts = list(range(0, max(T - M, 0) + 1, stride))
        for s in starts:
            sl = {k: v[s:s + M] for k, v in b.items()}
            st = stat_fn(sl)
            st = st[np.isfinite(st)]
            out.append(st.max() if len(st) else -np.inf)
    return np.array(out)


def calibrate_threshold(maxima, alpha):
    return float(np.quantile(maxima, 1 - alpha, method="higher"))


def first_alarm(stat, h, start=0):
    idx = np.nonzero(stat[start:] > h)[0]
    return int(idx[0] + start) if len(idx) else None


def stream_outcome(stat, h, tau, horizons=(5, 10, 20)):
    T = len(stat)
    H = T - tau
    ta = first_alarm(stat, h)
    fa = ta is not None and ta < tau
    if fa:
        d = np.nan
    elif ta is None:
        d = np.inf
    else:
        d = ta - tau + 1
    res = {"t_alarm": ta if ta is not None else -1, "false_alarm": bool(fa), "delay": d,
           "detected": bool((not fa) and np.isfinite(d)), "horizon": H}
    for hz in horizons:
        res[f"det_within_{hz}"] = bool((not fa) and np.isfinite(d) and d <= hz)
    # secondary framewise metrics
    y = np.r_[np.zeros(tau), np.ones(H)]
    s = np.where(np.isfinite(stat), stat, np.nanmin(stat[np.isfinite(stat)]) if np.isfinite(stat).any() else 0)
    try:
        res["auroc"] = roc_auc_score(y, s)
        res["auprc"] = average_precision_score(y, s)
    except ValueError:
        res["auroc"] = res["auprc"] = np.nan
    pred = s > h
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
    res["precision"] = tp / (tp + fp) if tp + fp else np.nan
    res["recall"] = tp / (tp + fn) if tp + fn else np.nan
    res["f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return res


def summarize(df, group_cols):
    """df: per-stream outcomes. Returns grouped summary with PFA, detection, delays."""
    import pandas as pd

    def agg(g):
        nfa = g[~g.false_alarm]
        d = nfa.delay.values
        dc = np.where(np.isfinite(d), d, nfa.horizon.values + 1)  # censored at H+1
        det = np.isfinite(d)
        out = {
            "n_streams": len(g),
            "pfa": g.false_alarm.mean(),
            "det_rate": det.mean() if len(d) else np.nan,
            "cadd_detected": d[det].mean() if det.any() else np.nan,
            "mean_delay_censored": dc.mean() if len(dc) else np.nan,
            "median_delay": np.median(np.where(np.isfinite(d), d, np.inf)) if len(d) else np.nan,
            "det_within_5": g.det_within_5.mean(),
            "det_within_10": g.det_within_10.mean(),
            "det_within_20": g.det_within_20.mean(),
            "auroc": g.auroc.mean(),
            "auprc": g.auprc.mean(),
            "f1": g.f1.mean(),
        }
        return pd.Series(out)

    return df.groupby(group_cols).apply(agg, include_groups=False).reset_index()
