"""Stage-5 confirmatory evaluation on the BDD holdout (DINOv2): frozen KCM, baselines, ablations, LOFO stress test.
Everything is fitted on non-holdout roles exactly as specified in FROZEN_METHOD.md / CONFIRMATORY_BENCHMARK_PLAN.md
and evaluated ONCE on the holdout. Writes results/v5/dino/*."""
import json, os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "scripts/v2"); sys.path.insert(0, "scripts/v4"); sys.path.insert(0, "scripts/v3"); sys.path.insert(0, "src")
from kcm import KCM, drive_max, conformal, MS  # noqa: E402
from run_v2 import Store  # noqa: E402
from rmrm.ccd import Evidence, Detector, DetSpec  # noqa: E402
from rmrm.baselines import Baseline  # noqa: E402
from rmrm.data import load_streams, load_index  # noqa: E402
from rmrm.evaluation import stream_outcome, calibrate_threshold  # noqa: E402
from v4lib import FAMILIES, REAL_CLOSE  # noqa: E402

OUT = "results/v5/dino"; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
ids = lambda r: sorted(k for k, v in part.items() if v == r)
store = Store(cfg["feature_root"]); FD = os.path.join(cfg["feature_root"], "clips"); HD = "data/features_v5/dinov2_s14/holdout"
TAU, H = cfg["tau"], cfg["H"]
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
S = pickle.load(open("results/v3/bdd_system.pkl", "rb")); scorer = S.scorer; del S

# ------------------------------------------------------------------ training / normalisation / calibration data
corr = [f for f in sorted(os.listdir(FD)) if any(f.startswith(k.replace("/", "__") + "__") for k in ns["normT"]) and not f.endswith("__clean.npz")]
cname = {f: f.split("__")[-1].rsplit("_s", 1)[0] for f in corr}
Gc = {f: np.load(os.path.join(FD, f))["glob"] for f in corr}
G_neg = np.concatenate([store.clip(f"{k}__clean")["glob"] for k in ns["normT"]])
A_pos, A_neg = {}, []
for s in load_index("data/v3/acdc_index.json"):
    g = np.load(os.path.join(FD, f"{s.seq_id}__clean".replace("/", "__") + ".npz"))["glob"]
    if s.seq_id.endswith("_adv"): A_pos.setdefault(s.attrs["condition"], []).append(g)
    else: A_neg.append(g)
A_pos = {c: np.concatenate(v) for c, v in A_pos.items()}; A_neg = np.concatenate(A_neg)
normS_g = [store.clip(f"{k}__clean")["glob"] for k in ns["normS"]]
normS_sc = [store.scores(f"{k}__clean", scorer) for k in ns["normS"]]
CAL = []
for k in ids("calib"):
    s = store.scores(f"{k}__clean", scorer)
    CAL.append(({kk: v[:TAU] for kk, v in s.items() if kk != "frame_idx"}, store.clip(f"{k}__clean")["glob"][:TAU], store.clip(f"{k}__clean")["glob"]))
P("data loaded")

def fit_kcm(idx=(0, 1, 2, 3), excl_corr=(), excl_real=()):
    pos = np.concatenate([Gc[f] for f in corr if cname[f] not in excl_corr])
    apos = np.concatenate([v for c, v in A_pos.items() if c not in excl_real])
    return KCM(idx=idx).fit([(pos, 1, False), (G_neg, 0, False), (apos, 1, False), (A_neg, 0, False)]).normalise(normS_g)

# ------------------------------------------------------------------ detectors
ev = Evidence(q_top=0.1).fit(normS_sc)
DET = {}
DET["KCM"] = fit_kcm()
for b, i in [(2, 0), (4, 1), (6, 2), (8, 3), (10, 4), (12, 5)]:
    DET[f"KCM_block{b}"] = fit_kcm(idx=(i,))
P("KCM + single-block variants fitted", DET["KCM"].n_train)
ref_glob = {"glob": np.concatenate([store.clip(f"{k}__clean")["glob"][::cfg["ref_stride"]] for k in ids("ref")])}
norm_glob = [{"glob": store.clip(f"{k}__clean")["glob"]} for k in ids("norm")]
BASE = {}
for n in ["mewma_multilayer", "mewma", "knn_ewma", "ewma", "cusum", "mmd"]:
    BASE[n] = Baseline(n).fit(ref_glob, norm_glob); P("baseline fitted", n)
g2det = Detector(DetSpec("single_g2", "single", cond=("g2",)), ev, MS).fit(ev.norm_Z)
stdet = Detector(DetSpec("still", "still_scan", extra={"blocks": [6, 8]}), ev, MS).fit(ev.norm_Z)
BNAME = {"mewma_multilayer": "MEWMA_multilayer", "mewma": "MEWMA_final", "knn_ewma": "deepkNN_EWMA_final",
         "ewma": "T2_EWMA_final", "cusum": "T2_CUSUM_final", "mmd": "MMD_final"}

def stats(sc, glob):
    """all detector trajectories for one stream / sequence"""
    out = {m: d.stat(glob) for m, d in DET.items()}
    for n, b in BASE.items():
        out[BNAME[n]] = np.r_[np.full(MS, -np.inf), b.score_stream({"glob": glob[MS:]})[0]]
    Z = ev.z(sc); out["g2_kNN_EWMA"] = g2det.stat(Z, sc); out["ST"] = stdet.stat(Z, sc)
    return out
# KCM + ST (rule C1, normalisers = 95% quantile of normS sequence maxima)
NSI = []
for k, s in zip(ns["normS"], normS_sc):
    g = store.clip(f"{k}__clean")["glob"][:TAU]; NSI.append(stats({kk: v[:TAU] for kk, v in s.items() if kk != "frame_idx"}, g))
qK = float(np.quantile([drive_max(t["KCM"]) for t in NSI], .95)); qS = float(np.quantile([drive_max(t["ST"]) for t in NSI], .95))
def add_combo(tr): tr["KCM+ST"] = np.maximum(tr["KCM"] / qK, tr["ST"] / qS); return tr
CALT = [add_combo(stats(sc, g)) for sc, g, _ in CAL]
METHODS = list(CALT[0])
ALPHAS = [0.02, 0.05, 0.10, 0.20]
TH = {m: {a: conformal([drive_max(t[m]) for t in CALT], a) for a in ALPHAS} for m in METHODS}
# calibration ablation A4b: stage-1 overlapping-window quantile (windows of 100 frames, step 20, within each calib drive)
win = {m: [] for m in ["KCM"]}
for sc, g, gfull in CAL:
    T = len(gfull)
    for s0 in range(0, T - TAU + 1, 20):
        win["KCM"].append(drive_max(DET["KCM"].stat(gfull[s0:s0 + TAU])))
TH_WIN = calibrate_threshold(np.array(win["KCM"]), 0.05)
P("thresholds (alpha=0.05):", {m: round(TH[m][0.05], 3) for m in METHODS}, "KCM windows:", round(TH_WIN, 3))
json.dump({"thresholds": TH, "kcm_window_threshold": TH_WIN, "combo_normalisers": {"KCM": qK, "ST": qS},
           "calib_maxima": {m: [float(drive_max(t[m])) for t in CALT] for m in METHODS}}, open(f"{OUT}/calibration.json", "w"), indent=1)

# ------------------------------------------------------------------ LOFO stress-test models (frozen training procedure per family)
LOFO = {}
for fam, cs in FAMILIES.items():
    m = fit_kcm(excl_corr=set(cs), excl_real=set(REAL_CLOSE.get(fam, [])))
    LOFO[fam] = (m, {a: conformal([drive_max(m.stat(g)) for _, g, _ in CAL], a) for a in ALPHAS})
P("LOFO models fitted")

# ------------------------------------------------------------------ holdout evaluation (single pass)
HC = {}
def hclip(cid):
    if cid not in HC:
        if len(HC) > 400: HC.clear()
        z = np.load(os.path.join(HD, cid.replace("/", "__") + ".npz")); HC[cid] = {k: z[k] for k in z.files}
    return HC[cid]
def assemble(st):
    parts = []
    for g in st.segments:
        c = hclip(g.clip_id); fi = c["frame_idx"]; a = int(np.searchsorted(fi, g.start)); b = int(np.searchsorted(fi, g.end))
        assert b - a == g.end - g.start, (g, fi[:3], len(fi))
        parts.append({k: v[a:b] for k, v in c.items() if k not in ("frame_idx", "pmap12")})
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    glob = d.pop("glob").astype(np.float32)
    return {k: v.astype(np.float32) for k, v in d.items()}, glob

rows = []
streams = [s for s in load_streams(cfg["streams"]) if s.split == "holdout"]
for i, st in enumerate(streams):
    sc, glob = assemble(st); tr = add_combo(stats(sc, glob))
    fam = next((f for f, cs in FAMILIES.items() if st.shift in cs), None) if st.kind == "synthetic" else None
    cond_drive = st.segments[1].seq_id if st.kind == "natural" else None
    extra = {}
    if fam is not None:
        m, th = LOFO[fam]; extra["KCM_LOFO"] = (m.stat(glob), th)
    if st.kind == "clean":
        extra["KCM_windowcal"] = (tr["KCM"], {0.05: TH_WIN})
    for meth in METHODS + list(extra):
        x, ths = (tr[meth], TH[meth]) if meth in tr else extra[meth]
        for a, h in ths.items():
            if st.tau is None:
                t = np.nonzero(x[:TAU] > h)[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
            else:
                r = stream_outcome(x, h, st.tau)
            rows.append({"method": meth, "alpha": a, "stream_id": st.stream_id, "kind": st.kind, "shift": st.shift, "family": fam,
                         "severity": st.severity, "unit": st.unit, "cond_drive": cond_drive, "false_alarm": r["false_alarm"],
                         "delay": r["delay"], "threshold": h, "pre_max": float(drive_max(x, MS, TAU))})
    if i % 100 == 0: P("stream", i, "/", len(streams))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/holdout_per_stream.csv", index=False)
pickle.dump({"KCM": DET["KCM"]}, open(f"{OUT}/kcm_model.pkl", "wb"))
P("done", len(df))
