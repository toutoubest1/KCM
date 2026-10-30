"""Stage-5 DINOv2 extraction for the confirmatory BDD holdout (runs once; resumable).
Per clip saves: glob [T,6,2,384] fp16, k-NN channel scores (g2..g12, pmap6, pmap8 vs BDD ref bank), stillness maps
(st6, st8), frame_idx; for occlusion clips also pmap12 (block-12 patch k-NN, localisation ablation A6).
Also builds the block-12 patch PCA / bank (BDD ref) and normS per-position statistics for pmap12."""
import json, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/v3")
from rmrm.data import load_index, load_streams, stable_hash  # noqa
from rmrm.backbones import build_backbone  # noqa
from rmrm.extract_v2 import run_backbone_raw, project_unit  # noqa
from rmrm.data import resize, load_frame  # noqa
from rmrm.corruptions import apply_corruption  # noqa
from rmrm.ccd import stillness_maps  # noqa
torch.set_num_threads(2); torch.manual_seed(0)
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); ns = json.load(open("data/v4/bdd_norm_split.json"))
OUT = "data/features_v5/dinov2_s14/holdout"; os.makedirs(OUT, exist_ok=True)
seqs = {s.seq_id: s for s in load_index(cfg["index"])}
tr = dict(np.load(os.path.join(cfg["feature_root"], "transforms_v2.npz")))
bb = build_backbone("dinov2_s14", [2, 4, 6, 8, 10, 12]); SZ = tuple(cfg["size_hw"])
t0 = time.time(); P = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

def frames(seq, a, b, corr=None, sev=None):
    seed = stable_hash(seq.seq_id + "|" + str(corr)); out = []
    for i in range(a, b):
        im = resize(load_frame(seq.frames[i], SZ), SZ)
        if corr is not None: im = apply_corruption(im, corr, sev, seed, i)
        out.append(im)
    return out

def run(ims):
    G, PU, PR, P12 = [], {6: [], 8: []}, {6: [], 8: []}, []
    for j in range(0, len(ims), 16):
        layers, g, pat = run_backbone_raw(bb, ims[j:j + 16]); G.append(g.astype(np.float16))
        for b in (6, 8):
            p, r = project_unit(pat[b], tr, b); PU[b].append(p.astype(np.float16)); PR[b].append(r.astype(np.float16))
        P12.append(pat[12])
    c = {"glob": np.concatenate(G)}
    for b in (6, 8): c[f"pu{b}"] = np.concatenate(PU[b]); c[f"pr{b}"] = np.concatenate(PR[b])
    return c, np.concatenate(P12)

# ---- block-12 patch PCA / bank / normS statistics (localisation ablation A6) - streamed, low memory
T12 = "data/features_v5/dinov2_s14/block12_loc.npz"
if not os.path.exists(T12):
    ref = sorted(k for k, v in part.items() if v == "ref")
    rng = np.random.default_rng(0)
    def pat12(ims):
        _, _, pat = run_backbone_raw(bb, ims); x = pat[12].astype(np.float32); return x / np.linalg.norm(x, axis=-1, keepdims=True)
    pf = [(k, i) for k in ref[:40] for i in range(0, min(len(seqs[k].frames), 200), 10)]
    sample = []
    for j in range(0, len(pf), 16):
        U = pat12([resize(load_frame(seqs[k].frames[i], SZ), SZ) for k, i in pf[j:j + 16]])
        sample += [u[rng.choice(U.shape[1], 64, replace=False)] for u in U]
    sample = np.concatenate(sample); mu = sample.mean(0); ev, V = np.linalg.eigh(np.cov((sample - mu).T)); W = V[:, ::-1][:, :64].astype(np.float32)
    P("block-12 PCA var", float(ev[::-1][:64].sum() / ev.sum()))
    def proj(U):
        p = (U - mu) @ W; r = ((U - mu) ** 2).sum(-1) - (p ** 2).sum(-1); return np.concatenate([p, np.sqrt(np.maximum(r, 0))[..., None]], -1).astype(np.float16)
    bf = [(k, i) for k in ref for i in range(0, min(len(seqs[k].frames), 200), 10)]
    bank = []
    for j in range(0, len(bf), 16):
        bank.append(proj(pat12([resize(load_frame(seqs[k].frames[i], SZ), SZ) for k, i in bf[j:j + 16]])))
    bank = np.concatenate(bank).transpose(1, 0, 2).astype(np.float32)
    P("block-12 bank", bank.shape)
    def pm(U):
        Q = proj(U).astype(np.float32).transpose(1, 0, 2)
        d2 = (Q * Q).sum(-1)[..., None] + (bank * bank).sum(-1)[:, None, :] - 2 * np.matmul(Q, bank.transpose(0, 2, 1))
        return np.log(np.partition(np.maximum(d2, 1e-12), 9, axis=-1)[..., 9]).T
    NS = []
    for k in ns["normS"]:
        ims = [resize(load_frame(seqs[k].frames[i], SZ), SZ) for i in range(0, min(len(seqs[k].frames), 200), 2)]
        for j in range(0, len(ims), 16): NS.append(pm(pat12(ims[j:j + 16])))
    NS = np.concatenate(NS); med = np.median(NS, 0); mad = 1.4826 * np.median(np.abs(NS - med), 0)
    np.savez(T12, mu=mu, W=W, bank=bank.astype(np.float16), med=med, mad=np.maximum(mad, 1e-6))
    P("block-12 localisation reference done", NS.shape); del bank, NS
S = pickle.load(open("results/v3/bdd_system.pkl", "rb")); scorer = S.scorer; del S
Z12 = dict(np.load(T12)); bank12 = Z12["bank"].astype(np.float32)
def pmap12(P12):
    U = P12 / np.linalg.norm(P12, axis=-1, keepdims=True); p = (U - Z12["mu"]) @ Z12["W"]; r = ((U - Z12["mu"]) ** 2).sum(-1) - (p ** 2).sum(-1)
    Q = np.concatenate([p, np.sqrt(np.maximum(r, 0))[..., None]], -1).transpose(1, 0, 2).astype(np.float32)
    d2 = (Q * Q).sum(-1)[..., None] + (bank12 * bank12).sum(-1)[:, None, :] - 2 * np.matmul(Q, bank12.transpose(0, 2, 1))
    return np.log(np.partition(np.maximum(d2, 1e-12), 9, axis=-1)[..., 9]).T
# ---- holdout clips
jobs = {}
for k, v in part.items():
    if v == "holdout": jobs[f"{k}__clean"] = (k, 0, min(len(seqs[k].frames), 200), None, None)
    if v == "cond_holdout": jobs[f"{k}__clean"] = (k, 0, 60, None, None)
for s in load_streams(cfg["streams"]):
    if s.split == "holdout" and s.kind == "synthetic":
        g = s.segments[1]; jobs[g.clip_id] = (g.seq_id, g.start, g.end, g.corruption, g.severity)
P(len(jobs), "holdout clips")
for n, (cid, (sid, a, b, c, sv)) in enumerate(sorted(jobs.items())):
    f = os.path.join(OUT, cid.replace("/", "__") + ".npz")
    if os.path.exists(f): continue
    cl, P12 = run(frames(seqs[sid], a, b, c, sv))
    sc = scorer.score_clip(cl); sc.update(stillness_maps(cl))
    out = {"glob": cl["glob"], "frame_idx": np.arange(a, b), **{k: np.asarray(v, np.float16) for k, v in sc.items()}}
    if c == "occlusion": out["pmap12"] = pmap12(P12).astype(np.float16)
    np.savez(f + ".tmp.npz", **out); os.replace(f + ".tmp.npz", f)
    if n % 50 == 0: P(n, cid, b - a, "frames")
P("done")
