"""Stage-2 runner: channel scores (cached per clip), evidence normalisation (norm partition), detectors,
drive-level conformal calibration (calib partition), evaluation on dev (or, after freeze, holdout) streams.

usage: python scripts/v2/run_v2.py --config configs/v2/base.json --detectors configs/v2/det_phase1.json --tag p1 [--split dev]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rmrm.data import load_streams  # noqa: E402
from rmrm.ccd import ChannelScorer, Evidence, Detector, DetSpec, drive_max, conformal_threshold, clopper_pearson, stillness_maps  # noqa: E402
from rmrm.baselines import Baseline  # noqa: E402
from rmrm.evaluation import stream_outcome, calibrate_threshold  # noqa: E402


class Store:
    def __init__(self, fdir):
        self.fdir = fdir; self.c = {}; self.sc = {}
        os.makedirs(os.path.join(fdir, "scores"), exist_ok=True)

    def clip(self, cid):
        if cid not in self.c:
            if len(self.c) > 60: self.c.clear()
            z = np.load(os.path.join(self.fdir, "clips", cid.replace("/", "__") + ".npz"))
            self.c[cid] = {k: z[k] for k in z.files}
        return self.c[cid]

    def scores(self, cid, scorer):
        if cid in self.sc: return self.sc[cid]
        f = os.path.join(self.fdir, "scores", cid.replace("/", "__") + ".npz")
        if os.path.exists(f):
            z = np.load(f); s = {k: z[k] for k in z.files}
            if "st6" not in s:                       # stillness maps added in iteration 2 (pure function of the clip)
                s.update(stillness_maps(self.clip(cid))); np.savez(f, **s)
        else:
            c = self.clip(cid); s = scorer.score_clip(c); s.update(stillness_maps(c)); s["frame_idx"] = c["frame_idx"]
            np.savez(f, **s)
        if len(self.sc) > 3000: self.sc.clear()
        self.sc[cid] = s
        return s


def seg_slice(d, fi, start, end):
    a = int(np.searchsorted(fi, start)); b = int(np.searchsorted(fi, end))
    assert b - a == end - start
    return {k: v[a:b] for k, v in d.items() if k != "frame_idx"}


def assemble(store, scorer, spec, what):
    parts = []
    for g in spec.segments:
        cid = g.clip_id
        src = store.scores(cid, scorer) if what == "scores" else store.clip(cid)
        fi = store.scores(cid, scorer)["frame_idx"] if what == "scores" else src["frame_idx"]
        parts.append(seg_slice(src, fi, g.start, g.end))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main(a):
    cfg = json.load(open(a.config)); dets = json.load(open(a.detectors))
    part = json.load(open(cfg["partitions"]))
    tau, ms, alpha = cfg["tau"], cfg["monitor_start"], cfg["alpha"]
    store = Store(cfg["feature_root"])
    ids = lambda p: sorted(k for k, v in part.items() if v == p)
    t0 = time.time()
    ref_clips = [store.clip(f"{k}__clean") for k in ids("ref")]
    scorer = ChannelScorer(cfg["stored_layers"], k=cfg.get("knn_k", 10)).fit(ref_clips)
    ref_glob = {"glob": np.concatenate([c["glob"][::cfg["ref_stride"]] for c in ref_clips])}
    del ref_clips; store.c.clear()
    print(f"scorer fitted {time.time()-t0:.0f}s", flush=True)
    norm_s = [store.scores(f"{k}__clean", scorer) for k in ids("norm")]
    ev = Evidence(q_top=cfg.get("q_top", 0.1)).fit(norm_s)
    print("channels:", ev.names, flush=True)
    # optional training data for learned evidence ratio (norm-drive corruptions vs clean norm frames)
    lr_data = None
    if any(d["kind"] in ("ccd_lr", "ccd_lr_resid") for d in dets):
        pos = [f[:-4].replace("__", "/", 1) for f in sorted(os.listdir(os.path.join(cfg["feature_root"], "clips")))
               if any(f.startswith(k.replace("/", "__") + "__") for k in ids("norm")) and not f.endswith("__clean.npz")]
        Xp = np.concatenate([ev.z(store.scores(c, scorer)) for c in pos])
        Xn = ev.norm_Z
        lr_data = (np.concatenate([Xp, Xn]), np.r_[np.ones(len(Xp)), np.zeros(len(Xn))])
        print(f"LR training: {len(Xp)} corrupted norm frames, {len(Xn)} clean norm frames", flush=True)
    # detectors
    D = {}
    unions = [d for d in dets if d["kind"] == "union"]
    joints = [d for d in dets if d["kind"] == "joint"]
    dets = [d for d in dets if d["kind"] not in ("union", "joint")]
    for d in dets:
        if d["kind"] == "baseline":
            b = Baseline(d["baseline"], **cfg.get("baseline_params", {}))
            b.fit(ref_glob, [{"glob": store.clip(f"{k}__clean")["glob"]} for k in ids("norm")])
            D[d["name"]] = ("raw", (lambda bb: (lambda x: np.r_[np.full(ms, -np.inf), bb.score_stream({"glob": x["glob"][ms:]})[0]]))(b))
        else:
            spec = DetSpec(**d)
            det = Detector(spec, ev, ms).fit(ev.norm_Z, lr_data)
            D[d["name"]] = ("z", (lambda dt: (lambda Z, sc: dt.stat(Z, sc)))(det))
    # calibration: drive-level conformal on calib drives (first tau frames, monitored from ms) + legacy windows
    def stat_of(name, src):
        typ, fn = D[name]
        if typ in ("union", "joint"):
            comps, hs = fn
            return np.max(np.stack([stat_of(c, src) / h for c, h in zip(comps, hs)]), 0)
        return fn(ev.z(src["scores"]), src["scores"]) if typ == "z" else fn(src["raw"])
    def clean_src(k, end):
        s = store.scores(f"{k}__clean", scorer); c = store.clip(f"{k}__clean")
        return {"scores": {kk: v[:end] for kk, v in s.items() if kk != "frame_idx"}, "raw": {"glob": c["glob"][:end]}}
    # joint detectors: G = max_i stat_i / s_i with s_i = 95% quantile of component maxima on NORM drives
    # (scale normalisers estimated outside the calibration set); G is then conformally calibrated like any detector
    for jd in joints:
        sc_ = []
        for c in jd["components"]:
            mx = []
            for k in ids("norm"):
                s_ = store.scores(f"{k}__clean", scorer); cl = store.clip(f"{k}__clean")
                mx.append(drive_max(stat_of(c, {"scores": {kk: v[:tau] for kk, v in s_.items() if kk != "frame_idx"},
                                                  "raw": {"glob": cl["glob"][:tau]}}), ms, tau))
            sc_.append(float(np.quantile(mx, 0.95)))
        D[jd["name"]] = ("joint", (jd["components"], sc_))
        print(f"joint {jd['name']}: normalisers {sc_}", flush=True)
    cal = {n: [] for n in D}; cal_win = {n: [] for n in D}
    for k in ids("calib"):
        src = clean_src(k, 10 ** 6)
        T = len(src["raw"]["glob"])
        for n in D:
            st = stat_of(n, {"scores": {kk: v[:tau] for kk, v in src["scores"].items()}, "raw": {"glob": src["raw"]["glob"][:tau]}})
            cal[n].append(drive_max(st, ms, tau))
            for s0 in range(0, T - tau + 1, 20):
                st = stat_of(n, {"scores": {kk: v[s0:s0 + tau] for kk, v in src["scores"].items()}, "raw": {"glob": src["raw"]["glob"][s0:s0 + tau]}})
                cal_win[n].append(drive_max(st, ms, tau))
    H = {n: {"conformal": conformal_threshold(cal[n], alpha), "windows": calibrate_threshold(np.array(cal_win[n]), alpha)} for n in D}
    # union detectors: G = max_i stat_i / h_i(alpha/k) with threshold 1 (Bonferroni over k conformal components)
    for u in unions:
        comps = u["components"]; k = len(comps)
        hs = [conformal_threshold(cal[c], alpha / k) for c in comps]
        D[u["name"]] = ("union", (comps, hs))
        cal[u["name"]] = list(np.max(np.stack([np.array(cal[c]) / h for c, h in zip(comps, hs)]), 0))
        H[u["name"]] = {"conformal": 1.0, "windows": 1.0}
    print(f"calibrated on {len(ids('calib'))} drives {time.time()-t0:.0f}s", flush=True)
    # evaluation
    streams = [s for s in load_streams(cfg["streams"]) if s.split == a.split]
    rows, traj = [], {n: {} for n in D}
    for i, s in enumerate(streams):
        src = {"scores": assemble(store, scorer, s, "scores"), "raw": assemble(store, scorer, s, "clip")}
        for n in D:
            st = stat_of(n, src); traj[n][s.stream_id] = st.astype(np.float32)
            for mode, h in H[n].items():
                if s.tau is None:
                    t = np.nonzero(st > h)[0]
                    r = {"false_alarm": bool(len(t) and t[0] < tau), "fa_full": bool(len(t)), "t_alarm": int(t[0]) if len(t) else -1}
                else:
                    r = stream_outcome(st, h, s.tau)
                rows.append({"method": n, "calib": mode, "h": h, "stream_id": s.stream_id, "split": s.split, "kind": s.kind,
                             "shift": s.shift, "severity": s.severity, "unit": s.unit, "tau": s.tau, **r})
        if i % 200 == 0:
            print(f"  stream {i}/{len(streams)} {time.time()-t0:.0f}s", flush=True)
    out = os.path.join("results", "v2", a.tag); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_stream.csv"), index=False)
    for n in D:
        np.savez_compressed(os.path.join(out, f"traj_{n}.npz"), **{k.replace("/", "__"): v for k, v in traj[n].items()})
    json.dump({"thresholds": H, "calib_drive_maxima": {n: list(map(float, v)) for n, v in cal.items()},
               "n_calib_drives": len(ids("calib")), "channels": ev.names, "detectors": dets}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    # quick calibration summary on unseen clean drives
    df = pd.DataFrame(rows); c = df[df.kind == "clean"]
    for (n, mode), g in c.groupby(["method", "calib"]):
        x, m = int(g.false_alarm.sum()), len(g); lo, hi = clopper_pearson(x, m)
        print(f"{n:28s} {mode:9s} PFA {x}/{m} = {x/m:.3f} [{lo:.3f}, {hi:.3f}]")
    print("done", time.time() - t0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--detectors", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--split", default="dev")
    main(p.parse_args())
