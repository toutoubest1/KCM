"""Stage-4 BDD evaluation: known protocol (all families) + leave-one-family-out (F1..F5), dev split only.
Roles: training normT, normalisation normS, threshold calib (66), evaluation dev. Holdout never loaded."""
import json, os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v2")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3")); sys.path.insert(0, "src")
from v4lib import (Dir, N1, N0, Wrap, Combo, g2_branch, st_branch, threshold, phi, FAMILIES, REAL_CLOSE, FAM_OF)  # noqa: E402
from run_v2 import Store, assemble  # noqa: E402
from rmrm.ccd import Evidence  # noqa: E402
from rmrm.data import load_streams, load_index  # noqa: E402
from rmrm.evaluation import stream_outcome  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert SPLIT in ("dev", "holdout")
if SPLIT == "holdout": assert os.path.exists("results/v4/FROZEN.json"), "holdout only after freeze"
OUT = "results/v4" if SPLIT == "dev" else "results/v4/holdout"; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
MS, TAU = cfg["monitor_start"], cfg["tau"]
store = Store(cfg["feature_root"]); S = pickle.load(open("results/v3/bdd_system.pkl", "rb")); scorer = S.scorer
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
# ---------------- normalisation role: evidence on normS
ev = Evidence(q_top=0.1).fit([store.scores(f"{k}__clean", scorer) for k in ns["normS"]])
names = ev.names
# ---------------- training role: normT
fd = os.path.join(cfg["feature_root"], "clips")
corr = [f for f in sorted(os.listdir(fd)) if any(f.startswith(k.replace("/", "__") + "__") for k in ns["normT"]) and not f.endswith("__clean.npz")]
cname = {f: f.split("__")[-1].rsplit("_s", 1)[0] for f in corr}
Zc = {f: ev.z(store.scores(f[:-4].replace("__", "/", 1), scorer)) for f in corr}
Pc = {f: phi(np.load(os.path.join(fd, f))["glob"]) for f in corr}
ZT = np.concatenate([ev.z(store.scores(f"{k}__clean", scorer)) for k in ns["normT"]])
PT = np.concatenate([phi(store.clip(f"{k}__clean")["glob"]) for k in ns["normT"]])
# ACDC (all frames; never used for BDD normalisation / calibration)
acdc = {"adv": {}, "ref": []}
for s in load_index("data/v3/acdc_index.json"):
    g = np.load(os.path.join(fd, f"{s.seq_id}__clean".replace("/", "__") + ".npz"))["glob"]
    if s.seq_id.endswith("_adv"): acdc["adv"].setdefault(s.attrs["condition"], []).append(phi(g))
    else: acdc["ref"].append(phi(g))
A_neg = np.concatenate(acdc["ref"]); A_pos = {c: np.concatenate(v) for c, v in acdc["adv"].items()}
P("training data ready", len(corr), "corrupted normT clips;", ZT.shape, "clean normT frames; ACDC", {c: len(v) for c, v in A_pos.items()}, len(A_neg))
folds = ["K"] + list(FAMILIES)
D, NN = {}, {}
for fo in folds:
    held = set(FAMILIES.get(fo, [])); realx = set(REAL_CLOSE.get(fo, []))
    keep = [f for f in corr if cname[f] not in held]
    D[fo] = Dir([(np.concatenate([Pc[f] for f in keep]), 1), (PT, 0), (np.concatenate([v for c, v in A_pos.items() if c not in realx]), 1), (A_neg, 0)])
    NN[fo] = N1(names, np.concatenate([Zc[f] for f in keep]), ZT)
    P("trained fold", fo, "excluded", sorted(held | realx), "D sizes", D[fo].sizes)
del Pc, PT, A_pos, A_neg
# ---------------- standardise on normS
normS_glob = [store.clip(f"{k}__clean")["glob"] for k in ns["normS"]]
for fo in folds: D[fo].standardise(normS_glob); NN[fo].standardise(ev.norm_Z)
n0 = N0(names).standardise(ev.norm_Z); g2 = g2_branch(ev, MS); st = st_branch(ev, MS); mew = Wrap(S.bl["mewma_multilayer"], "baseline")

def trajs(sc, glob):
    Z = ev.z(sc); out = {"N0": n0.stat(sc, Z, glob, MS), "g2": g2.stat(sc, Z, glob, MS), "ST": st.stat(sc, Z, glob, MS),
                         "MEWMA": mew.stat(sc, Z, glob, MS)}
    for fo in folds:
        out[f"D|{fo}"] = D[fo].stat(sc, Z, glob, MS); out[f"N1|{fo}"] = NN[fo].stat(sc, Z, glob, MS)
    return out

def seq_items(ids, T=TAU):
    r = []
    for k in ids:
        s = store.scores(f"{k}__clean", scorer); g = store.clip(f"{k}__clean")["glob"]
        r.append(trajs({kk: v[:T] for kk, v in s.items() if kk != "frame_idx"}, g[:T]))
    return r
NORM = seq_items(ns["normS"]); CAL = seq_items(sorted(k for k, v in part.items() if v == "calib"))
P("normS / calib trajectories done")
DETS = {"V1": ("C1", ["D", "N1", "ST"]), "V2": ("C2", ["D", "N1", "ST"]), "V3": ("C1", ["D", "N0", "ST"]),
        "M1": (None, ["N1"]), "M2": (None, ["D"]), "M1+ST": ("C1", ["N1", "ST"]), "N0": (None, ["N0"]),
        "g2": (None, ["g2"]), "ST": (None, ["ST"]), "MEWMA": (None, ["MEWMA"])}
def key(b, fo): return f"{b}|{fo}" if b in ("D", "N1") else b
combos, H, audit = {}, {}, {}
for fo in folds:
    for m, (rule, br) in DETS.items():
        sel = lambda it: {b: it[key(b, fo)] for b in br}
        if rule is None:
            stat = lambda it, b=br[0]: it[key(b, fo)]
        else:
            cb = Combo(br, rule).fit([sel(it) for it in NORM], MS, TAU); combos[(fo, m)] = cb
            stat = lambda it, cb=cb: cb.stat(sel(it))
            # S4-G audit on calibration drives: 95% quantile of each branch's normalised maxima; argmax shares
            pr = [cb.parts(sel(it)) for it in CAL]
            mx = {b: [np.nanmax(np.where(np.isfinite(p[b][MS:TAU]), p[b][MS:TAU], np.nan)) for p in pr] for b in br}
            arg = [max(br, key=lambda b: mx[b][i]) for i in range(len(pr))]
            if rule == "C1":
                normq = {b: float(np.quantile(mx[b], 0.95)) for b in br}           # normS value is 1 by construction
            else:
                normq = {b: float(np.quantile(mx[b], 0.95) / np.quantile([np.nanmax(cb.parts(sel(it))[b][MS:TAU]) for it in NORM], 0.95)) for b in br}
            audit[(fo, m)] = {"calib_q95_ratio": normq, "argmax_share": {b: arg.count(b) / len(arg) for b in br}}
        H[(fo, m)] = threshold([stat(it) for it in CAL], MS, TAU)
        DETS[m] = (rule, br)
P("thresholds", {m: round(H[("K", m)], 3) for m in DETS})
rows = []
streams = [s for s in load_streams(cfg["streams"]) if s.split == SPLIT]
for i, s in enumerate(streams):
    sc = assemble(store, scorer, s, "scores"); g = assemble(store, scorer, s, "clip")["glob"]; tr = trajs(sc, g)
    fam = FAM_OF.get(s.shift) if s.kind == "synthetic" else None
    for fo in folds:
        if fo != "K" and s.kind == "synthetic" and fam != fo: continue
        if fo != "K" and s.kind == "natural": continue
        for m, (rule, br) in DETS.items():
            x = tr[key(br[0], fo)] if rule is None else combos[(fo, m)].stat({b: tr[key(b, fo)] for b in br})
            h = H[(fo, m)]
            if s.tau is None:
                t = np.nonzero(x[:TAU] > h)[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
            else:
                r = stream_outcome(x, h, s.tau)
            rows.append({"protocol": "known" if fo == "K" else "lofo", "fold": fo, "method": m, "stream_id": s.stream_id, "kind": s.kind,
                         "shift": s.shift, "family": fam, "severity": s.severity, "unit": s.unit, "false_alarm": r["false_alarm"], "delay": r["delay"]})
    if i % 200 == 0: P("stream", i, "/", len(streams))
pd.DataFrame(rows).to_csv(f"{OUT}/bdd_per_stream.csv", index=False)
json.dump({"thresholds": {f"{fo}|{m}": v for (fo, m), v in H.items()},
           "combo": {f"{fo}|{m}": (cb.q if cb.rule == "C1" else {b: len(r) for b, r in cb.ref.items()}) for (fo, m), cb in combos.items()},
           "audit": {f"{fo}|{m}": v for (fo, m), v in audit.items()}}, open(f"{OUT}/bdd_meta.json", "w"), indent=1)
pickle.dump({"D": D, "N1": NN, "n0": n0}, open(f"{OUT}/bdd_models.pkl", "wb"))
P("done")
