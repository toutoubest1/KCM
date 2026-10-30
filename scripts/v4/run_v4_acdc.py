"""Stage-4 ACDC evaluation: recording-level 4-fold cross-fitting (data/v4/acdc_folds.json); per fold the in-control
statistics are re-estimated on that fold's ACDC roles (a_ref bank, a_norm normalisation, a_calib threshold), the
directional branch is trained on BDD normT synthetic + ACDC a_train(fold). Protocols: K (known), F1 (illumination held
out -> evaluate night), F4 (weather held out -> evaluate fog/rain/snow). Stillness is disabled (ACDC is not video)."""
import json, os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v2")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3")); sys.path.insert(0, "src")
from v4lib import Dir, N0, Wrap, Combo, g2_branch, threshold, phi, FAMILIES, REAL_CLOSE  # noqa: E402
from run_v2 import Store  # noqa: E402
from rmrm.ccd import Evidence  # noqa: E402
from rmrm.ccd import ChannelScorer  # noqa: E402
from rmrm.data import stable_hash  # noqa: E402
from rmrm.baselines import Baseline  # noqa: E402
from rmrm.evaluation import stream_outcome  # noqa: E402

cfg = json.load(open("configs/v2/base.json")); acfg = json.load(open("configs/v3/acdc.json")); ns = json.load(open("data/v4/bdd_norm_split.json"))
TAU, MS, H = acfg["tau"], acfg["monitor_start"], acfg["H"]
store = Store(cfg["feature_root"]); fd = os.path.join(cfg["feature_root"], "clips")
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
BM = pickle.load(open("results/v4/bdd_models.pkl", "rb"))          # N1 weights per protocol (trained on BDD normT only)
T = pd.read_csv("data/v4/acdc_frames.csv", dtype={"rec": str}); FO = json.load(open("data/v4/acdc_folds.json"))
CL = {}
for c in set(T.adv_clip) | set(T.ref_clip):
    z = np.load(os.path.join(fd, f"{c}__clean".replace("/", "__") + ".npz")); CL[c] = {k: z[k] for k in ("glob", "pu6", "pu8", "pr6", "pr8")}
def vclip(rows, kind):
    c, r = rows[f"{kind}_clip"].values, rows[f"{kind}_row"].values
    return {k: np.stack([CL[a][k][b] for a, b in zip(c, r)]) for k in ("glob", "pu6", "pu8", "pr6", "pr8")}
def sub(c, a, b): return {k: v[a:b] for k, v in c.items()}
# BDD training data for D (same as BDD run)
corr = [f for f in sorted(os.listdir(fd)) if any(f.startswith(k.replace("/", "__") + "__") for k in ns["normT"]) and not f.endswith("__clean.npz")]
cname = {f: f.split("__")[-1].rsplit("_s", 1)[0] for f in corr}
Pc = {f: phi(np.load(os.path.join(fd, f))["glob"]) for f in corr}
PT = np.concatenate([phi(store.clip(f"{k}__clean")["glob"]) for k in ns["normT"]])
P("loaded")
PROTOS = {"K": None, "F1_illumination": ["night"], "F4_weather": ["fog", "rain", "snow"]}
rows, audit, thr = [], {}, {}
for k in sorted(FO):
    fo = FO[k]; roles = fo["roles"]
    def role_rows(role):
        out = []
        for u, r in roles.items():
            if r != role: continue
            c, rec, ch = u.split("_"); out.append(T[(T.cond == c) & (T.rec == rec) & (T.chap == int(ch[2:]))])
        d = pd.concat(out); return {rk: g.sort_values(["chap", "frame"]) for rk, g in d.groupby(["cond", "rec"])}
    R = {r: role_rows(r) for r in ("a_ref", "a_train", "a_norm", "a_calib")}
    scorer = ChannelScorer(cfg["stored_layers"], k=10, patch_ref_stride=1, glob_ref_stride=1).fit([vclip(g, "ref") for g in R["a_ref"].values()])
    def chunks(role):
        res = []
        for g in R[role].values():
            c = vclip(g, "ref"); n = len(g)
            for j in range(0, n - TAU + 1, TAU): res.append(sub(c, j, j + TAU))
        return res
    NC, CC = chunks("a_norm"), chunks("a_calib")
    norm_full = [vclip(g, "ref") for g in R["a_norm"].values()]
    ev = Evidence(q_top=0.1).fit([scorer.score_clip(c) for c in norm_full])
    n0 = N0(ev.names).standardise(ev.norm_Z); g2 = g2_branch(ev, MS)
    mew = Baseline("mewma_multilayer").fit({"glob": np.concatenate([vclip(g, "ref")["glob"] for g in R["a_ref"].values()])}, [{"glob": c["glob"]} for c in norm_full])
    mewb = Wrap(mew, "baseline")
    tr_adv = {c: np.concatenate([phi(vclip(g, "adv")["glob"]) for (cc, _), g in R["a_train"].items() if cc == c] or [np.zeros((0, 3072))]) for c in ["fog", "night", "rain", "snow"]}
    tr_neg = np.concatenate([phi(vclip(g, "ref")["glob"]) for g in R["a_train"].values()])
    eval_recs = sorted(fo["eval"], key=lambda u: stable_hash(u))
    EV = {u: T[(T.cond == u.split("_")[0]) & (T.rec == u.split("_")[1])].sort_values(["chap", "frame"]) for u in eval_recs}
    EVC = {u: {"adv": vclip(g, "adv"), "ref": vclip(g, "ref")} for u, g in EV.items()}
    streams = []
    for i, u in enumerate(eval_recs):
        n = len(EV[u]); oth = eval_recs[(i + 1) % len(eval_recs)]; no = len(EV[oth]); cnd = u.split("_")[0]
        for q, j in enumerate(range(0, n - TAU + 1, TAU)):
            streams.append(("clean", "none", u, [(u, "ref", j, j + TAU)], None))
        for q, j in enumerate(range(0, n - TAU - H + 1, 20)):
            streams.append(("condition", cnd, u, [(u, "ref", j, j + TAU), (u, "adv", j + TAU, j + TAU + H)], TAU))
            j2 = (j * 7) % max(no - H, 1)
            streams.append(("hard_negative", "hn_scene", u, [(u, "ref", j, j + TAU), (oth, "ref", j2, j2 + H)], TAU))
    SC = {}
    def seq_scores(segs):
        cl = {kk: np.concatenate([EVC[u][kd][kk][a:b] for (u, kd, a, b) in segs]) for kk in ("glob", "pu6", "pu8", "pr6", "pr8")}
        return scorer.score_clip(cl), cl["glob"]
    for proto, realx in PROTOS.items():
        held = set(FAMILIES.get(proto, [])); realx = set(realx or [])
        keep = [f for f in corr if cname[f] not in held]
        pos_a = np.concatenate([v for c, v in tr_adv.items() if c not in realx and len(v)])
        Dm = Dir([(np.concatenate([Pc[f] for f in keep]), 1), (PT, 0), (pos_a, 1), (tr_neg, 0)]).standardise([c["glob"] for c in norm_full])
        N1m = BM["N1"][proto]; import copy; N1m = copy.deepcopy(N1m).standardise(ev.norm_Z)
        def trajs(sc, glob):
            Z = ev.z(sc)
            return {"D": Dm.stat(sc, Z, glob, MS), "N1": N1m.stat(sc, Z, glob, MS), "N0": n0.stat(sc, Z, glob, MS),
                    "g2": g2.stat(sc, Z, glob, MS), "MEWMA": mewb.stat(sc, Z, glob, MS)}
        NT = [trajs(scorer.score_clip(c), c["glob"]) for c in NC]; CT = [trajs(scorer.score_clip(c), c["glob"]) for c in CC]
        DETS = {"V1": ("C1", ["D", "N1"]), "V2": ("C2", ["D", "N1"]), "V3": ("C1", ["D", "N0"]), "M1": (None, ["N1"]),
                "M2": (None, ["D"]), "N0": (None, ["N0"]), "g2": (None, ["g2"]), "MEWMA": (None, ["MEWMA"])}
        cbs, Hh = {}, {}
        for m, (rule, br) in DETS.items():
            if rule:
                cbs[m] = Combo(br, rule).fit([{b: t[b] for b in br} for t in NT], MS, TAU)
                Hh[m] = threshold([cbs[m].stat({b: t[b] for b in br}) for t in CT], MS, TAU)
                pr = [cbs[m].parts({b: t[b] for b in br}) for t in CT]
                mx = {b: [np.nanmax(np.where(np.isfinite(p[b][MS:TAU]), p[b][MS:TAU], np.nan)) for p in pr] for b in br}
                audit[f"{k}|{proto}|{m}"] = {"calib_q95": {b: float(np.quantile(mx[b], .95)) for b in br},
                                              "argmax_share": {b: float(np.mean([max(br, key=lambda bb: mx[bb][i]) == b for i in range(len(pr))])) for b in br}}
            else:
                Hh[m] = threshold([t[br[0]] for t in CT], MS, TAU)
        thr[f"{k}|{proto}"] = {**Hh, "n_calib_chunks": len(CT), "n_norm_chunks": len(NT)}
        for kind, shift, u, segs, tau in streams:
            if proto != "K" and kind == "condition" and shift not in realx: continue
            key = tuple(segs)
            if key not in SC: SC[key] = seq_scores(segs)
            sc, glob = SC[key]; tr = trajs(sc, glob)
            for m, (rule, br) in DETS.items():
                x = tr[br[0]] if rule is None else cbs[m].stat({b: tr[b] for b in br})
                if tau is None:
                    t = np.nonzero(x[:TAU] > Hh[m])[0]; r = {"false_alarm": bool(len(t)), "delay": np.nan}
                else:
                    r = stream_outcome(x, Hh[m], tau)
                rows.append({"fold": int(k), "protocol": proto, "method": m, "kind": kind, "shift": shift, "unit": u,
                             "false_alarm": r["false_alarm"], "delay": r["delay"]})
        P("fold", k, proto, "done; calib chunks", len(CT), "norm chunks", len(NT), "thresholds", {m: round(v, 2) for m, v in Hh.items()})
pd.DataFrame(rows).to_csv("results/v4/acdc_per_stream.csv", index=False)
json.dump({"thresholds": thr, "audit": audit}, open("results/v4/acdc_meta.json", "w"), indent=1)
P("done")
