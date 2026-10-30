"""Stage-5 backbone-generalisation extraction (glob tokens only: CLS and mean-patch of blocks 2..12) for a frozen
alternative encoder. Same frames as the DINOv2 pipeline: BDD ref (stride 2), norm (52), normT corrupted clips,
calib, ACDC (all), BDD holdout clips. Corruptions are applied at 182x322 exactly as for DINOv2, then the image is
resized (INTER_AREA) to the encoder input 176x320. Resumable.  usage: extract_backbone_glob.py clip_b16_openai|deit_s16"""
import json, os, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "src")
from rmrm.data import load_index, load_streams, stable_hash, resize, load_frame  # noqa
from rmrm.backbones import build_backbone  # noqa
from rmrm.extract_v2 import run_backbone_raw  # noqa
from rmrm.corruptions import apply_corruption  # noqa
torch.set_num_threads(2)
name = sys.argv[1]; cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
OUT = f"data/features_v5/{name}"; os.makedirs(OUT, exist_ok=True)
bb = build_backbone(name, [2, 4, 6, 8, 10, 12]); SZ = (182, 322); IN = (176, 320)
seqs = {s.seq_id: s for s in load_index(cfg["index"])}; seqs.update({s.seq_id: s for s in load_index("data/v3/acdc_index.json")})
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
jobs = {}
for k, v in part.items():
    n = len(seqs[k].frames) if k in seqs else 0
    if v == "ref": jobs[f"ref/{k}"] = (k, list(range(0, min(n, 200), 2)), None, None)
    if v == "norm": jobs[f"norm/{k}"] = (k, list(range(0, min(n, 200))), None, None)
    if v == "calib": jobs[f"calib/{k}"] = (k, list(range(0, 100)), None, None)
    if v == "holdout": jobs[f"holdout/{k}__clean"] = (k, list(range(0, min(n, 200))), None, None)
    if v == "cond_holdout": jobs[f"holdout/{k}__clean"] = (k, list(range(0, 60)), None, None)
for i, k in enumerate(sorted(k for k, v in part.items() if v == "norm")):
    if k not in ns["normT"]: continue
    n = len(seqs[k].frames)
    for j, c in enumerate(cfg["corruptions"]):
        sev = (i + j) % 3 + 1; st = min(100, n - 30); jobs[f"normT_corr/{k}__{c}_s{sev}"] = (k, list(range(st, st + 30)), c, sev)
for s in load_index("data/v3/acdc_index.json"): jobs[f"acdc/{s.seq_id}"] = (s.seq_id, list(range(len(s.frames))), None, None)
for s in load_streams(cfg["streams"]):
    if s.split == "holdout" and s.kind == "synthetic":
        g = s.segments[1]; jobs[f"holdout/{g.clip_id}"] = (g.seq_id, list(range(g.start, g.end)), g.corruption, g.severity)
order = ["normT_corr", "norm", "calib", "ref", "acdc", "holdout"]
items = sorted(jobs.items(), key=lambda kv: (order.index(kv[0].split("/")[0]), kv[0]))
P(name, len(items), "clips,", sum(len(v[1]) for v in jobs.values()), "frames")
for n_, (cid, (sid, idx, c, sv)) in enumerate(items):
    f = os.path.join(OUT, cid.replace("/", "__") + ".npz")
    if os.path.exists(f): continue
    seed = stable_hash(sid + "|" + str(c)); ims = []
    for i in idx:
        im = resize(load_frame(seqs[sid].frames[i], SZ), SZ)
        if c is not None: im = apply_corruption(im, c, sv, seed, i)
        ims.append(cv2.resize(im, (IN[1], IN[0]), interpolation=cv2.INTER_AREA))
    G = []
    for j in range(0, len(ims), 16):
        _, g, _ = run_backbone_raw(bb, ims[j:j + 16]); G.append(g.astype(np.float16))
    np.savez(f + ".tmp.npz", glob=np.concatenate(G), frame_idx=np.array(idx)); os.replace(f + ".tmp.npz", f)
    if n_ % 100 == 0: P(n_, cid)
P("done")
