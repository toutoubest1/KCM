"""Stage-2 data: drive-level partitions, positive streams and hard-negative streams.

usage: python scripts/v2/prepare_data_v2.py --config configs/v2/base.json
Writes: data/v2/index.json, data/v2/partitions.json, data/v2/streams.json, data/v2/drive_table.csv

Rules (fixed before any stage-2 detector result, see DATA_SPLIT_V2.md):
* eligible drives: BDD100K MOT train parts (any partition) + BDD MOT val drives that were NOT in the closed v1
  test split (old roles test / test_post are excluded entirely; old ref_fit/norm/calib/dev/dev_post drives may
  only enter ref/norm/calib/dev, never the new final holdout).
* clean = daytime & weather in {clear, partly cloudy, overcast}; condition drives = night, dawn/dusk, rainy,
  snowy, foggy (daytime rain/snow; any-time fog).
* one partition per DRIVE (hash of the video name); no frame of a drive appears in two partitions.
"""
import argparse, collections, glob, json, os, sys
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rmrm.data import Sequence, save_index, stable_hash, Segment, StreamSpec, save_streams  # noqa: E402

IMG = (".jpg", ".png")


def frames_in(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG))


def motion_proxy(frames, step=5, n=40):
    """Mean absolute grey-level difference between frames `step` apart (ego-motion/scene-motion proxy)."""
    idx = list(range(0, min(len(frames) - step, n * step), step))
    v = []
    for i in idx:
        a = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE); b = cv2.imread(frames[i + step], cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (120, 68)).astype(np.float32); b = cv2.resize(b, (120, 68)).astype(np.float32)
        v.append(np.abs(a - b)[:45].mean())   # upper 2/3 of the image (exclude hood/dashboard)
    return float(np.median(v)) if v else np.nan


def main(cfg):
    attrs = pd.read_csv(cfg["attrs_csv"]).set_index("name")
    old_roles = json.load(open(cfg["old_roles"]))
    rows = []
    for root, src in [(cfg["val_root"], "val"), (cfg["train_root"], "train")]:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            vid = os.path.basename(d); fr = frames_in(d)
            if len(fr) < cfg["min_len"] or vid not in attrs.index:
                continue
            a = attrs.loc[vid]
            rows.append(dict(vid=vid, src=src, n_frames=len(fr), weather=a.weather, scene=a.scene, timeofday=a.timeofday,
                             n_car=a.n_car, n_person=a.n_person, n_truck=a.n_truck, n_bus=a.n_bus,
                             old_role=old_roles.get(f"bdd/{vid}", "") if src == "val" else "", dir=d))
    T = pd.DataFrame(rows)
    T = T[~T.old_role.isin(["test", "test_post"])].copy()      # closed v1 test drives
    clean = (T.timeofday == "daytime") & T.weather.isin(["clear", "partly cloudy", "overcast"])
    cond = pd.Series("", index=T.index)
    cond[T.timeofday == "night"] = "night"
    cond[T.timeofday == "dawn/dusk"] = "dawn_dusk"
    cond[(T.timeofday == "daytime") & (T.weather == "rainy")] = "rain"
    cond[(T.timeofday == "daytime") & (T.weather == "snowy")] = "snow"
    cond[T.weather == "foggy"] = "fog"
    T["clean"] = clean; T["condition"] = cond
    # partitions
    fr = cfg["clean_fractions"]           # ref, norm, calib, dev, holdout
    names = list(fr)
    def assign(vid, allowed):
        w = np.array([fr[n] for n in allowed]); w = np.cumsum(w / w.sum())
        u = (stable_hash("v2part|" + vid) % 100000) / 100000
        return allowed[int(np.searchsorted(w, u, side="right"))]
    part = []
    for _, r in T.iterrows():
        if r.clean:
            allowed = names if r.src == "train" else [n for n in names if n != "holdout"]
            part.append(assign(r.vid, allowed))
        elif r.condition:
            if r.src == "val":
                part.append("cond_dev")
            else:
                u = (stable_hash("v2cond|" + r.vid) % 100000) / 100000
                part.append("cond_dev" if u < 0.5 else "cond_holdout")
        else:
            part.append("unused")
    T["partition"] = part
    # motion proxy for clean dev/holdout drives (hard negative construction)
    T["motion"] = np.nan
    for i, r in T[T.partition.isin(["dev", "holdout"])].iterrows():
        T.loc[i, "motion"] = motion_proxy(frames_in(r.dir))
    os.makedirs("data/v2", exist_ok=True)
    T.drop(columns=["dir"]).to_csv("data/v2/drive_table.csv", index=False)
    seqs = [Sequence(f"bdd/{r.vid}", f"bdd100k_track_{r.src}", frames_in(r.dir),
                     {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()
                      if k in ("weather", "scene", "timeofday", "n_car", "n_person", "src", "partition", "condition")}, 5.0)
            for _, r in T[T.partition != "unused"].iterrows()]
    save_index(seqs, "data/v2/index.json")
    json.dump({f"bdd/{r.vid}": r.partition for _, r in T.iterrows() if r.partition != "unused"}, open("data/v2/partitions.json", "w"), indent=1)

    # ------------------------------------------------------------- streams
    tau, H = cfg["tau"], cfg["H"]
    need = tau + H
    corrs = cfg["corruptions"]
    streams = []
    for split in ("dev", "holdout"):
        P = T[(T.partition == split) & (T.n_frames >= need)].sort_values("vid").reset_index(drop=True)
        C = T[(T.partition == f"cond_{split}") & (T.n_frames >= H)]
        for i, a in P.iterrows():
            A = f"bdd/{a.vid}"
            streams.append(StreamSpec(f"{A}|clean", split, "clean", "none", None, [Segment(A, 0, min(a.n_frames, cfg["max_clean_len"]))], None, A))
            for j, c in enumerate(corrs):
                sev = (i + j) % 3 + 1
                streams.append(StreamSpec(f"{A}|{c}|s{sev}", split, "synthetic", c, sev,
                                          [Segment(A, 0, tau), Segment(A, tau, need, c, sev)], tau, A))
            for cname in ["night", "dawn_dusk", "rain", "snow", "fog"]:
                pool = C[C.condition == cname].sort_values("vid")
                if len(pool) == 0:
                    continue
                pool = pool.iloc[np.argsort([stable_hash(cname + v) for v in pool.vid])]
                b = pool.iloc[i % len(pool)]
                streams.append(StreamSpec(f"{A}|nat:{cname}", split, "natural", cname, None,
                                          [Segment(A, 0, tau), Segment(f"bdd/{b.vid}", 0, H)], tau, A))
            # hard negatives: clean -> different clean drive of the same partition, by controlled factor
            others = P[P.vid != a.vid]
            def pick(df, key):
                if len(df) == 0:
                    return None
                return df.iloc[np.argsort([stable_hash(key + a.vid + v) for v in df.vid])].iloc[0]
            q_lo, q_hi = P.n_car.quantile(1 / 3), P.n_car.quantile(2 / 3)
            m_lo, m_hi = P.motion.quantile(1 / 3), P.motion.quantile(2 / 3)
            cand = {
                "hn_random_scene": others,
                "hn_road_geometry": others[others.scene != a.scene],
                "hn_traffic_density": others[(others.n_car >= q_hi)] if a.n_car <= q_lo else others[(others.n_car <= q_lo)],
                "hn_object_composition": others[(others.n_person >= 3) & (others.n_car <= q_hi)] if a.n_person < 3 else others[(others.n_person == 0)],
                "hn_camera_motion": others[others.motion <= m_lo] if a.motion >= np.median(P.motion) else others[others.motion >= m_hi],
            }
            for hn, df in cand.items():
                b = pick(df, hn)
                if b is None:
                    continue
                streams.append(StreamSpec(f"{A}|{hn}", split, "hard_negative", hn, None,
                                          [Segment(A, 0, tau), Segment(f"bdd/{b.vid}", 0, H)], tau, A))
    save_streams(streams, "data/v2/streams.json")
    print(T.groupby(["partition", "src"]).size())
    print(collections.Counter((s.split, s.kind) for s in streams))
    print(collections.Counter((s.split, s.shift) for s in streams if s.kind in ("natural", "hard_negative")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(json.load(open(p.parse_args().config)))
