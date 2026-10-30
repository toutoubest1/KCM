"""RADIATE external validation (RADIATE_PREREGISTRATION.md; implementation details RADIATE_RUN_LOG.md R-2).
Frozen KCM direction (DINOv2: stored stage-5 model; CLIP/DeiT: frozen stage-5 E4 recipe re-fitted on the same BDD/ACDC
features, reproduction of the stored BDD thresholds asserted). Only normalisation (R) and threshold (C) are
re-estimated. Single pass. usage: run_radiate.py dinov2_s14|clip_b16_openai|deit_s16"""
import copy, json, os, pickle, sys, time
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT)
sys.path.insert(0, "scripts/v5"); sys.path.insert(0, "src")
from kcm import KCM, drive_max, conformal, MS, l2n  # noqa
from rmrm.baselines import Baseline  # noqa
from rmrm.data import load_index  # noqa
from rmrm.evaluation import stream_outcome  # noqa
name = sys.argv[1]; FD = f"data/features_v6/{name}"; OUT = f"results/v6/{name}"; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
TAU, H, ALPHAS = 100, 60, (0.02, 0.05, 0.10, 0.20)
ROLES = {"R": ["city_2_0", "city_5_0", "junction_3_0"],
         "C": ["city_1_1", "city_3_0", "city_6_0", "junction_1_2", "junction_2_0", "motorway_2_1"],
         "T0": ["city_7_0", "motorway_1_0", "rural_1_1"],
         "fog": ["fog_6_0", "fog_8_0", "fog_8_1", "fog_8_2"], "rain": ["rain_4_0", "rain_4_1"], "snow": ["snow_1_0"],
         "night": ["night_1_0", "night_1_2", "night_1_4"]}
SESSION = lambda s: s.rsplit("_", 1)[0]
G = {s: np.load(f"{FD}/{s}.npz")["glob"].astype(np.float32) for r in ROLES.values() for s in r}

# ------------------------------------------------------------------ frozen direction
if name == "dinov2_s14":
    kcm = copy.deepcopy(pickle.load(open("results/v5/dino/kcm_model.pkl", "rb"))["KCM"])
    w_hash = float(np.abs(kcm.lr.coef_).sum())
else:
    BD = f"data/features_v5/{name}"; L = lambda cid: np.load(os.path.join(BD, cid.replace("/", "__") + ".npz"))
    cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
    pos = np.concatenate([np.load(os.path.join(BD, f))["glob"] for f in sorted(os.listdir(BD)) if f.startswith("normT_corr__")])
    neg = np.concatenate([L(f"norm/{k}")["glob"] for k in ns["normT"]])
    apos = np.concatenate([L(f"acdc/{s.seq_id}")["glob"] for s in load_index("data/v3/acdc_index.json") if s.seq_id.endswith("_adv")])
    aneg = np.concatenate([L(f"acdc/{s.seq_id}")["glob"] for s in load_index("data/v3/acdc_index.json") if s.seq_id.endswith("_ref")])
    kcm = KCM().fit([(pos, 1, False), (neg, 0, False), (apos, 1, False), (aneg, 0, False)]).normalise([L(f"norm/{k}")["glob"] for k in ns["normS"]])
    hb = conformal([drive_max(kcm.stat(L(f"calib/{k}")["glob"].astype(np.float32))) for k in sorted(k for k, v in part.items() if v == "calib")], 0.05)
    href = json.load(open(f"results/v5/backbone_{name}/thresholds.json"))["KCM"]["0.05"]
    assert abs(hb - href) < 1e-6, (hb, href); P("frozen E4 direction reproduced; BDD threshold", hb)
    w_hash = float(np.abs(kcm.lr.coef_).sum())
bdd_norm = (float(kcm.mu), float(kcm.sd))
kcm.normalise([G[s] for s in ROLES["R"]])           # re-instantiation: normalisation on R (frozen procedure sec. 3)
P("KCM normalisation BDD", bdd_norm, "-> RADIATE R", (kcm.mu, kcm.sd))

# ------------------------------------------------------------------ baselines fitted on R
R_all = np.concatenate([G[s] for s in ROLES["R"]])
B = {n: Baseline(n).fit({"glob": R_all[::2]}, [{"glob": G[s]} for s in ROLES["R"]]) for n in ["mewma_multilayer", "knn_ewma"]}
BN = {"mewma_multilayer": "MEWMA_multilayer", "knn_ewma": "deepkNN_EWMA_final"}
u2 = lambda g: l2n(g[:, 0].reshape(len(g), -1))      # block-2 [CLS, mean-patch], l2-normalised (ChannelScorer convention)
def knn_logd2(Q, Bk, k=10):
    d2 = (Q * Q).sum(1)[:, None] + (Bk * Bk).sum(1)[None] - 2 * Q @ Bk.T
    return np.log(np.partition(np.maximum(d2, 1e-12), k - 1, axis=1)[:, k - 1])
bank2 = u2(R_all)
loso = np.concatenate([knn_logd2(u2(G[s]), u2(np.concatenate([G[o] for o in ROLES["R"] if o != s]))) for s in ROLES["R"]])
g2_med = np.median(loso); g2_sd = 1.4826 * np.median(np.abs(loso - g2_med))
LAM = 0.1; FAC = np.sqrt(LAM / (2 - LAM))
def ewma(x):
    out = np.full(len(x), -np.inf); e = 0.0
    for t in range(MS, len(x)): e = (1 - LAM) * e + LAM * x[t]; out[t] = e
    return out
def stats(g):
    out = {"KCM": kcm.stat(g)}
    for n, b in B.items(): out[BN[n]] = np.r_[np.full(MS, -np.inf), b.score_stream({"glob": g[MS:]})[0]]
    out["g2_kNN_EWMA"] = ewma((knn_logd2(u2(g), bank2) - g2_med) / g2_sd) / FAC
    return out

# ------------------------------------------------------------------ calibration on C chunks
chunks = lambda s: [(s, i) for i in range(len(G[s]) // 100)]
cal = [c for s in ROLES["C"] for c in chunks(s)]
CT = [stats(G[s][i * 100:(i + 1) * 100]) for s, i in cal]
METHODS = list(CT[0])
TH = {m: {a: conformal([drive_max(t[m]) for t in CT], a) for a in ALPHAS} for m in METHODS}
P("calibration chunks", len(cal), "thresholds a=0.05", {m: round(TH[m][0.05], 3) for m in METHODS})
json.dump({"n_calibration_chunks": len(cal), "thresholds": TH, "calib_maxima": {m: [float(drive_max(t[m])) for t in CT] for m in METHODS},
           "kcm_norm_bdd": bdd_norm, "kcm_norm_radiate": [float(kcm.mu), float(kcm.sd)], "w_abs_sum": w_hash,
           "g2_norm": [float(g2_med), float(g2_sd)], "calib_units": [f"{s}#{i}" for s, i in cal]}, open(f"{OUT}/calibration.json", "w"), indent=1)

# ------------------------------------------------------------------ streams
t0c = [c for s in ROLES["T0"] for c in chunks(s)]
inter = []; per = {s: chunks(s) for s in ROLES["T0"]}
for i in range(max(len(v) for v in per.values())):
    for s in ROLES["T0"]:
        if i < len(per[s]): inter.append(per[s][i])
windows = lambda s: [(s, j) for j in range(len(G[s]) // 60)]
streams = []
for s, i in t0c:
    streams.append(dict(kind="clean", shift="none", pre=f"{s}#{i}", post=None, g=G[s][i * 100:(i + 1) * 100], tau=None))
k = 0
for cond in ["fog", "rain", "snow", "night"]:
    for s in ROLES[cond]:
        for _, j in windows(s):
            ps, pi = inter[k % len(inter)]; k += 1
            streams.append(dict(kind="condition", shift=cond, pre=f"{ps}#{pi}", post=f"{s}@{j}", post_seq=s,
                                g=np.concatenate([G[ps][pi * 100:(pi + 1) * 100], G[s][j * 60:(j + 1) * 60]]), tau=TAU))
use = {s: 0 for s in ROLES["T0"]}
for a in ROLES["T0"]:
    for b in ROLES["T0"]:
        if a == b: continue
        for _, i in chunks(a):
            wb = windows(b); _, j = wb[use[b] % len(wb)]; use[b] += 1
            streams.append(dict(kind="hard_negative", shift=f"hn_{SESSION(a)}->{SESSION(b)}", pre=f"{a}#{i}", post=f"{b}@{j}", post_seq=b,
                                g=np.concatenate([G[a][i * 100:(i + 1) * 100], G[b][j * 60:(j + 1) * 60]]), tau=TAU))
P("streams", pd.Series([s["kind"] for s in streams]).value_counts().to_dict())
rows = []
for st in streams:
    tr = stats(st["g"])
    for m in METHODS:
        x = tr[m]
        for a in ALPHAS:
            h = TH[m][a]
            if st["tau"] is None:
                t = np.nonzero(x[:TAU] > h)[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
            else:
                r = stream_outcome(x, h, st["tau"])
            rows.append({"backbone": name, "method": m, "alpha": a, "kind": st["kind"], "shift": st["shift"], "pre": st["pre"],
                         "pre_session": SESSION(st["pre"].split("#")[0]), "post": st["post"], "post_seq": st.get("post_seq"),
                         "post_session": SESSION(st["post_seq"]) if st.get("post_seq") else None,
                         "false_alarm": r["false_alarm"], "delay": r["delay"], "threshold": h,
                         "pre_max": float(drive_max(x, MS, TAU)), "post_max": float(np.max(x[TAU:])) if st["tau"] else np.nan})
pd.DataFrame(rows).to_csv(f"{OUT}/per_stream.csv", index=False)
P("done", len(rows))
