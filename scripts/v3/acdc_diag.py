"""Stage-3 diagnostic of the UNCHANGED stage-2 system on ACDC (dev only).

D1 frame-level decomposition (paired adverse vs corresponding normal frames) and D2 sequential detection,
for instantiations U0 (all BDD-fitted) and U1 (in-control population statistics from ACDC normal data,
learned LR / beta unchanged). See METHOD_DEVELOPMENT_LOG_V3.md, Entry V3-0.
usage: python scripts/v3/acdc_diag.py [--inst U0 U1] [--tag diag]
"""
import argparse, json, os, pickle, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "src")
from system3 import System, reinstantiate, DETS  # noqa: E402
from rmrm.ccd import stillness_maps, drive_max, conformal_threshold, clopper_pearson  # noqa: E402
from rmrm.data import load_index, load_streams  # noqa: E402
from rmrm.evaluation import stream_outcome  # noqa: E402

cfg = json.load(open("configs/v3/acdc.json")); bcfg = json.load(open("configs/v2/base.json"))
part = json.load(open(cfg["partitions"])); seqs = {s.seq_id: s for s in load_index(cfg["index"])}
TAU, MS, H, ALPHA = cfg["tau"], cfg["monitor_start"], cfg["H"], cfg["alpha"]
CL = {}
def clip(sid):
    if sid not in CL:
        z = np.load(os.path.join(cfg["feature_root"], "clips", f"{sid}__clean".replace("/", "__") + ".npz"))
        CL[sid] = {k: z[k] for k in z.files}
    return CL[sid]
units = lambda r: sorted(u for u, v in part.items() if v == r)
NAMES = [d["name"] for d in DETS]


def scores_all(S):
    out = {}
    for sid in seqs:
        c = clip(sid); s = S.scorer.score_clip(c); s.update(stillness_maps(c)); out[sid] = s
    return out


def chunks(sc, us):
    """non-overlapping TAU-frame normal chunks of the given units: list of (scores, glob)"""
    res = []
    for u in us:
        s = sc[f"{u}_ref"]; g = clip(f"{u}_ref")["glob"]; n = len(g)
        for j in range(0, n - TAU + 1, TAU):
            res.append(({k: v[j:j + TAU] for k, v in s.items()}, g[j:j + TAU]))
    return res


def build(inst, S_bdd):
    if inst == "U0":
        S = S_bdd; S.ms, S.tau = MS, TAU
        sc = scores_all(S)
        # U0 keeps the BDD joint normalisers (computed on BDD norm drives, window [20,100)); only the conformal
        # threshold is set on ACDC calib chunks.
        return S, sc
    S = reinstantiate(S_bdd, [clip(f"{u}_ref") for u in units("ref")],
                      lambda scorer: [dict(**scorer.score_clip(clip(f"{u}_ref")), **stillness_maps(clip(f"{u}_ref"))) for u in units("norm")],
                      [clip(f"{u}_ref")["glob"] for u in units("norm")],
                      {**bcfg, "acdc_patch_ref_stride": 1})
    S.ms, S.tau = MS, TAU
    sc = scores_all(S)
    S.fit_joint(chunks(sc, units("norm")), window=(MS, TAU))
    return S, sc


def d1(inst, S, sc):
    rows = []
    cal = [S.components(sc[f"{u}_ref"]) for u in units("calib")]
    cal = {k: np.concatenate([c[k] for c in cal]) for k in cal[0]}
    rsd = {k: 1.4826 * np.median(np.abs(v - np.median(v))) for k, v in cal.items()}
    for setname, us in [("dev", units("dev")), ("calib_adverse_replication", units("calib"))]:
        for cond in cfg["conditions"]:
            uu = [u for u in us if u.split("/")[1].startswith(cond)]
            A = [S.components(sc[f"{u}_adv"]) for u in uu]; N = [S.components(sc[f"{u}_ref"]) for u in uu]
            for k in cal:
                a = np.concatenate([x[k] for x in A]); n = np.concatenate([x[k] for x in N])
                rows.append({"inst": inst, "set": setname, "condition": cond, "quantity": k, "n_pairs": len(a), "n_rec": len(uu),
                             "paired_effect_sd": float(np.median(a - n) / rsd[k]),
                             "auroc": float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(n))], np.r_[a, n])),
                             "paired_win": float(np.mean(a > n)),
                             "per_rec_effect": json.dumps({u.split("/")[1]: round(float(np.median(x[k] - y[k]) / rsd[k]), 2) for u, x, y in zip(uu, A, N)})})
    return rows


def seg(sc, sid, a, b, what):
    if what == "glob": return clip(sid)["glob"][a:b]
    return {k: v[a:b] for k, v in sc[sid].items()}


def d2(inst, S, sc, streams):
    H_ = {}
    calm = {n: [] for n in NAMES}
    for s_, g_ in chunks(sc, units("calib")):
        for n in NAMES: calm[n].append(drive_max(S.stat(n, s_, g_), MS, TAU))
    for n in NAMES: H_[n] = conformal_threshold(calm[n], ALPHA)
    rows = []
    for st in streams:
        parts = [(g.seq_id, g.start, g.end) for g in st.segments]
        s_ = {k: np.concatenate([seg(sc, *p, "s")[k] for p in parts]) for k in sc[parts[0][0]]}
        g_ = np.concatenate([seg(sc, *p, "glob") for p in parts])
        for n in NAMES:
            stat = S.stat(n, s_, g_); h = H_[n]
            if st.tau is None:
                t = np.nonzero(stat[:TAU] > h)[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
            else:
                r = stream_outcome(stat, h, st.tau)
            rows.append({"inst": inst, "method": n, "h": h, "stream_id": st.stream_id, "kind": st.kind, "shift": st.shift,
                         "unit": st.unit, "false_alarm": r["false_alarm"], "delay": r.get("delay", np.nan)})
    return rows, {n: {"h": H_[n], "calib_maxima": list(map(float, calm[n]))} for n in NAMES}


def summarize(df):
    out = []
    for (inst, m), g in df.groupby(["inst", "method"]):
        c = g[g.kind == "clean"]; x, n = int(c.false_alarm.sum()), len(c); lo, hi = clopper_pearson(x, n)
        r = {"inst": inst, "method": m, "pfa": x / n, "pfa_lo": lo, "pfa_hi": hi, "n_clean": n}
        hn = g[(g.kind == "hard_negative") & ~g.false_alarm.astype(bool)]
        r["hn_alarm"] = float(np.isfinite(hn.delay.astype(float)).mean()); r["n_hn"] = len(hn)
        for cond in cfg["conditions"]:
            k = g[(g.kind == "condition") & (g["shift"] == cond) & ~g.false_alarm.astype(bool)]
            d = k.delay.astype(float)
            r[f"{cond}_det"] = float(np.isfinite(d).mean()) if len(k) else np.nan
            r[f"{cond}_median_delay"] = float(np.median(np.where(np.isfinite(d), d, np.inf))) if len(k) else np.nan
            r[f"{cond}_n"] = len(k)
        r["mean_real_gap"] = np.nanmean([r[f"{c}_det"] for c in cfg["conditions"]]) - r["hn_alarm"]
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--inst", nargs="+", default=["U1", "U0"]); p.add_argument("--tag", default="diag")
    a = p.parse_args(); od = f"results/v3/{a.tag}"; os.makedirs(od, exist_ok=True)
    streams = [s for s in load_streams(cfg["streams"]) if s.split == "dev"]
    D1, D2, meta = [], [], {}
    for inst in a.inst:
        t0 = time.time(); S_bdd = pickle.load(open("results/v3/bdd_system.pkl", "rb"))
        S, sc = build(inst, S_bdd); print(inst, "built", f"{time.time()-t0:.0f}s", "joint", S.joint_s, flush=True)
        D1 += d1(inst, S, sc); r, meta[inst] = d2(inst, S, sc, streams); D2 += r
        pickle.dump({"scores": sc}, open(f"{od}/scores_{inst}.pkl", "wb"))
        del S, S_bdd, sc
    d1df = pd.DataFrame(D1); d1df.to_csv(f"{od}/d1_frame_decomposition.csv", index=False)
    d2df = pd.DataFrame(D2); d2df.to_csv(f"{od}/d2_per_stream.csv", index=False)
    json.dump(meta, open(f"{od}/d2_calibration.json", "w"), indent=1)
    sm = summarize(d2df); sm.to_csv(f"{od}/d2_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for inst in a.inst:
        x = d1df[(d1df.inst == inst) & (d1df.set == "dev")]
        print(f"== D1 {inst} dev: paired AUROC"); print(x.pivot(index="quantity", columns="condition", values="auroc").round(3).to_string())
        print(f"== D1 {inst} dev: paired effect (in-control SD)"); print(x.pivot(index="quantity", columns="condition", values="paired_effect_sd").round(2).to_string())
    print(sm.round(3).to_string(index=False))
