"""Stage-5 E4: frozen KCM recipe + baselines on an alternative frozen encoder (same frames, roles, layer mapping:
blocks 2,4,6,8 for KCM; final block for final-layer baselines). usage: run_backbone.py clip_b16_openai|deit_s16"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "src")
from kcm import KCM, drive_max, conformal, MS  # noqa
from rmrm.baselines import Baseline  # noqa
from rmrm.data import load_streams, load_index  # noqa
from rmrm.evaluation import stream_outcome  # noqa
name = sys.argv[1]; FD = f"data/features_v5/{name}"; OUT = f"results/v5/backbone_{name}"; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
ids = lambda r: sorted(k for k, v in part.items() if v == r); TAU = cfg["tau"]
L = lambda cid: np.load(os.path.join(FD, cid.replace("/", "__") + ".npz"))
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
pos = np.concatenate([np.load(os.path.join(FD, f))["glob"] for f in sorted(os.listdir(FD)) if f.startswith("normT_corr__")])
neg = np.concatenate([L(f"norm/{k}")["glob"] for k in ns["normT"]])
apos = np.concatenate([L(f"acdc/{s.seq_id}")["glob"] for s in load_index("data/v3/acdc_index.json") if s.seq_id.endswith("_adv")])
aneg = np.concatenate([L(f"acdc/{s.seq_id}")["glob"] for s in load_index("data/v3/acdc_index.json") if s.seq_id.endswith("_ref")])
kcm = KCM().fit([(pos, 1, False), (neg, 0, False), (apos, 1, False), (aneg, 0, False)]).normalise([L(f"norm/{k}")["glob"] for k in ns["normS"]])
kcm_f = KCM(idx=(5,)).fit([(pos, 1, False), (neg, 0, False), (apos, 1, False), (aneg, 0, False)]).normalise([L(f"norm/{k}")["glob"] for k in ns["normS"]])
P("KCM fitted", kcm.n_train)
ref = {"glob": np.concatenate([L(f"ref/{k}")["glob"] for k in ids("ref")])}
normg = [{"glob": L(f"norm/{k}")["glob"]} for k in ids("norm")]
B = {n: Baseline(n).fit(ref, normg) for n in ["mewma_multilayer", "mewma", "knn_ewma"]}
BN = {"mewma_multilayer": "MEWMA_multilayer", "mewma": "MEWMA_final", "knn_ewma": "deepkNN_EWMA_final"}
def stats(g):
    g = g.astype(np.float32); out = {"KCM": kcm.stat(g), "KCM_block12": kcm_f.stat(g)}
    for n, b in B.items(): out[BN[n]] = np.r_[np.full(MS, -np.inf), b.score_stream({"glob": g[MS:]})[0]]
    return out
CT = [stats(L(f"calib/{k}")["glob"]) for k in ids("calib")]
TH = {m: {a: conformal([drive_max(t[m]) for t in CT], a) for a in (0.02, 0.05, 0.10, 0.20)} for m in CT[0]}
P("thresholds", {m: round(v[0.05], 3) for m, v in TH.items()})
HC = {}
def seg(g):
    cid = f"holdout/{g.clip_id}"
    if cid not in HC:
        if len(HC) > 400: HC.clear()
        z = L(cid); HC[cid] = (z["glob"], z["frame_idx"])
    G, fi = HC[cid]; a = int(np.searchsorted(fi, g.start)); return G[a:a + g.end - g.start]
rows = []
for st in [s for s in load_streams(cfg["streams"]) if s.split == "holdout"]:
    tr = stats(np.concatenate([seg(g) for g in st.segments]))
    for m, x in tr.items():
        for a, h in TH[m].items():
            if st.tau is None:
                t = np.nonzero(x[:TAU] > h)[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
            else:
                r = stream_outcome(x, h, st.tau)
            rows.append({"backbone": name, "method": m, "alpha": a, "stream_id": st.stream_id, "kind": st.kind, "shift": st.shift,
                         "severity": st.severity, "unit": st.unit, "cond_drive": st.segments[1].seq_id if st.kind == "natural" else None,
                         "false_alarm": r["false_alarm"], "delay": r["delay"]})
pd.DataFrame(rows).to_csv(f"{OUT}/holdout_per_stream.csv", index=False); json.dump(TH, open(f"{OUT}/thresholds.json", "w"), indent=1)
P("done")
