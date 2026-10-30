"""Stage-4 partitions (V4-0 section 2 + addendum): BDD normT/normS split; ACDC recording-level 4-fold cross-fitting.
Per fold and condition: a_calib = whole non-eval recordings (hash order) up to its share (never the last remaining
recording); the other non-eval recordings are split at chapter level into a_ref / a_train / a_norm by deficit.
Deterministic, feature-free."""
import json, os, sys, collections, glob
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from rmrm.data import load_index, stable_hash  # noqa: E402
# ---- BDD
part = json.load(open("data/v2/partitions.json"))
norm = sorted([k for k, v in part.items() if v == "norm"], key=lambda k: stable_hash("v4|" + k))
bdd = {"normT": sorted(norm[: len(norm) // 2]), "normS": sorted(norm[len(norm) // 2:])}
json.dump(bdd, open("data/v4/bdd_norm_split.json", "w"), indent=1)
# ---- ACDC frame table (maps every frame to its row in the stage-3 feature clips)
rows = []
for s in load_index("data/v3/acdc_index.json"):
    kind = "adv" if s.seq_id.endswith("_adv") else "ref"
    for i, p in enumerate(s.frames):
        c, sp, seq, f = p.split("/")[-4:]
        rows.append({"cond": c, "rec": seq[-4:], "chap": 0 if seq.startswith("GOPR") else int(seq[2:4]),
                     "frame": int(f.split("_frame_")[1][:6]), "kind": kind, "clip": s.seq_id, "row": i, "path": p})
F = pd.DataFrame(rows)
A = F[F.kind == "adv"].set_index(["cond", "rec", "chap", "frame"]); R = F[F.kind == "ref"].set_index(["cond", "rec", "chap", "frame"])
T = A[["clip", "row"]].rename(columns={"clip": "adv_clip", "row": "adv_row"}).join(R[["clip", "row"]].rename(columns={"clip": "ref_clip", "row": "ref_row"}), how="inner").reset_index()
assert len(T) == 4006, len(T)
T = T.sort_values(["cond", "rec", "chap", "frame"]).reset_index(drop=True)
T.to_csv("data/v4/acdc_frames.csv", index=False)
# ---- folds
shares = {"a_ref": 0.24, "a_train": 0.21, "a_norm": 0.15, "a_calib": 0.40}
recs = T.groupby(["cond", "rec"]).size()
fold_of = {}
for c in sorted(T.cond.unique()):
    rr = sorted([r for (cc, r) in recs.index if cc == c], key=lambda r: stable_hash(f"v4|{c}|{r}"))
    for i, r in enumerate(rr): fold_of[(c, r)] = i % 4
folds = {}
for k in range(4):
    roles = {}
    for c in sorted(T.cond.unique()):
        pool = T[(T.cond == c) & T.apply(lambda x: fold_of[(x.cond, x.rec)] != k, axis=1)]
        tot = len(pool); acc = collections.Counter()
        # a_calib: WHOLE recordings (hash order) until >= its share -> long calibration sequences and calibration
        # recordings disjoint from every fitted role; remaining recordings split at chapter level by deficit
        rr = sorted(pool.rec.unique(), key=lambda r: stable_hash(f"v4cal|{k}|{c}|{r}"))
        cal = set()
        for r in rr:
            if acc["a_calib"] < shares["a_calib"] * tot and len(rr) - len(cal) > 1:
                cal.add(r); acc["a_calib"] += int((pool.rec == r).sum())
        for r in sorted(cal):
            for ch in sorted(pool[pool.rec == r].chap.unique()): roles[f"{c}_{r}_ch{ch}"] = "a_calib"
        rest = pool[~pool.rec.isin(cal)]
        sh = {q: v for q, v in shares.items() if q != "a_calib"}; rtot = len(rest); racc = collections.Counter()
        chaps = sorted(rest.groupby(["rec", "chap"]).size().items(), key=lambda t: stable_hash(f"v4|{k}|{c}|{t[0][0]}|{t[0][1]}"))
        for (r, ch), n in chaps:
            role = max(sh, key=lambda q: sh[q] / sum(sh.values()) * rtot - racc[q]); racc[role] += n
            roles[f"{c}_{r}_ch{ch}"] = role
    folds[k] = {"eval": sorted(f"{c}_{r}" for (c, r), f in fold_of.items() if f == k), "roles": roles}
json.dump(folds, open("data/v4/acdc_folds.json", "w"), indent=1)
for k, v in folds.items():
    cnt = collections.Counter()
    for u, r in v["roles"].items():
        c, rec, ch = u.split("_"); cnt[r] += int(((T.cond == c) & (T.rec == rec) & (T.chap == int(ch[2:]))).sum())
    ev = sum(int(recs[(e.split("_")[0], e.split("_")[1])]) for e in v["eval"])
    print(k, "eval", v["eval"], ev, dict(cnt))
