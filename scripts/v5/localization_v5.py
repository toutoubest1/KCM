"""Stage-5 E7 / A6: occlusion localisation on the BDD holdout (62 streams; exact occluder mask pooled to the 13x23 grid,
cell occluded if >= 50% covered). Map = mean over [tau, tau+19] of: block-6 / block-8 patch k-NN evidence (per-position
normS standardisation), block-12 patch k-NN evidence (final layer; normS statistics), block-8 stillness. Overlays chosen
by a fixed rule: best / median / worst block-6 AUROC + one night (natural) and one hard-negative stream."""
import json, os, pickle, sys
import numpy as np, pandas as pd, cv2
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/v2"); sys.path.insert(0, "scripts/v3")
from rmrm.data import load_streams, load_index, stable_hash, resize, load_frame  # noqa
from rmrm.corruptions import apply_corruption  # noqa
from rmrm.ccd import Evidence  # noqa
from run_v2 import Store  # noqa
cfg = json.load(open("configs/v2/base.json")); ns = json.load(open("data/v4/bdd_norm_split.json"))
store = Store(cfg["feature_root"]); S = pickle.load(open("results/v3/bdd_system.pkl", "rb")); scorer = S.scorer; del S
ev = Evidence(q_top=0.1).fit([store.scores(f"{k}__clean", scorer) for k in ns["normS"]])
Z12 = dict(np.load("data/features_v5/dinov2_s14/block12_loc.npz")); HD = "data/features_v5/dinov2_s14/holdout"
H0, W0 = cfg["size_hw"]; gh, gw = H0 // 14, W0 // 14
seqs = {s.seq_id: s for s in load_index(cfg["index"])}
L = lambda cid: dict(np.load(os.path.join(HD, cid.replace("/", "__") + ".npz")))
rows, maps = [], {}
for s in load_streams(cfg["streams"]):
    if s.split != "holdout" or s.shift != "occlusion": continue
    g = s.segments[1]; white = np.full((H0, W0, 3), 255, np.uint8)
    occ = apply_corruption(white, "occlusion", g.severity, stable_hash(g.seq_id + "|occlusion"), g.start)
    mask = cv2.resize((occ.mean(2) < 128).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA).ravel() >= 0.5
    if mask.sum() == 0 or mask.all(): continue
    c = L(g.clip_id); sc = {k: c[k][:20].astype(np.float32) for k in ("pmap6", "pmap8", "st6", "st8")}
    m = {"block6_kNN": ev.zmap(sc, 6).mean(0), "block8_kNN": ev.zmap(sc, 8).mean(0),
         "block12_kNN": ((c["pmap12"][:20].astype(np.float32) - Z12["med"]) / Z12["mad"]).mean(0), "block8_stillness": ev.zstill(sc, 8).mean(0)}
    r = {"stream_id": s.stream_id, "severity": s.severity, "frac_occluded": float(mask.mean())}
    for k, v in m.items():
        r[f"auroc_{k}"] = roc_auc_score(mask, v); r[f"auprc_{k}"] = average_precision_score(mask, v)
    rows.append(r); maps[s.stream_id] = (m, mask, g)
df = pd.DataFrame(rows); os.makedirs("results/v5/localization", exist_ok=True); df.to_csv("results/v5/localization/holdout_localization.csv", index=False)
print(df.drop(columns="stream_id").mean().round(3).to_string()); print(df.drop(columns="stream_id").groupby("severity").mean().round(3).to_string())
# overlays
os.makedirs("figures/v5_localization", exist_ok=True)
def overlay(seq_id, frame, mp, mask, title, fn):
    im = resize(load_frame(seqs[seq_id].frames[frame], (H0, W0)), (H0, W0)).copy()
    hm = cv2.resize(mp.reshape(gh, gw).astype(np.float32), (W0, H0), interpolation=cv2.INTER_CUBIC)
    hm = np.clip((hm - np.percentile(hm, 5)) / (np.percentile(hm, 99) - np.percentile(hm, 5) + 1e-6), 0, 1)
    col = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    out = (0.55 * im + 0.45 * col).astype(np.uint8)
    if mask is not None:
        mk = cv2.resize(mask.reshape(gh, gw).astype(np.uint8) * 255, (W0, H0), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cnts, -1, (0, 255, 0), 1)
    big = cv2.resize(out, (W0 * 2, H0 * 2)); cv2.putText(big, title, (6, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.imwrite(fn, big[..., ::-1])
o = df.sort_values("auroc_block6_kNN"); pick = {"best": o.iloc[-1], "median": o.iloc[len(o) // 2], "worst": o.iloc[0]}
for tag, r in pick.items():
    m, mask, g = maps[r.stream_id]
    for k in ("block6_kNN", "block12_kNN", "block8_stillness"):
        corr = apply_corruption(resize(load_frame(seqs[g.seq_id].frames[g.start + 10], (H0, W0)), (H0, W0)), "occlusion", g.severity, stable_hash(g.seq_id + "|occlusion"), g.start + 10)
        cv2.imwrite(f"/tmp/occ.png", corr[..., ::-1])
        seqs["_tmp"] = type(seqs[g.seq_id])(seq_id="_tmp", dataset="tmp", frames=["/tmp/occ.png"])
        overlay("_tmp", 0, m[k], mask, f"{tag} occlusion s{g.severity}: {k} AUROC {r[f'auroc_{k}']:.2f}", f"figures/v5_localization/{tag}_{k}.png")
for kind, shift in [("natural", "night"), ("hard_negative", "hn_object_composition")]:
    s = next(x for x in load_streams(cfg["streams"]) if x.split == "holdout" and x.kind == kind and x.shift == shift)
    g = s.segments[1]; c = L(f"{g.seq_id}__clean"); sc = {k: c[k][:20].astype(np.float32) for k in ("pmap6", "pmap8", "st6", "st8")}
    overlay(g.seq_id, 10, ev.zmap(sc, 6).mean(0), None, f"{shift}: block6_kNN map (no spatial ground truth)", f"figures/v5_localization/{shift}_block6_kNN.png")
print("overlays written")
