"""Stage-2 (v2) feature cache: raw DINOv2 token geometry, no coordinatewise standardisation.

Per frame (blocks B = stored layers, default 2,4,6,8,10,12):
  glob  [T, |B|, 2, D] fp16   raw CLS token and raw mean patch token of every block
  pu{b} [T, N, P]      fp16   for patch blocks b in PATCH_BLOCKS (default 6, 8): L2-normalised patch token,
                              centred by the reference mean unit token, projected on the top-P principal
                              directions of reference unit tokens (geometry-preserving rotation + truncation)
  pr{b} [T, N]         fp16   squared norm of the out-of-subspace residual of that centred unit token
The PCA basis is fitted once on reference-partition drives only (transforms_v2.npz).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .data import Sequence, resize, load_frame, stable_hash
from .corruptions import apply_corruption
from .extract import _to_tensor

PATCH_BLOCKS = (6, 8)
P_DIM = 64


def run_backbone_raw(bb, ims):
    out = bb.forward_layers(_to_tensor(ims))
    layers = sorted(out)
    cls = torch.stack([out[l][0] for l in layers], 1)            # B,L,D
    pat = {l: out[l][1] for l in layers}                          # B,N,D
    glob = torch.stack([cls, torch.stack([pat[l].mean(1) for l in layers], 1)], 2)
    return layers, glob.numpy(), {l: pat[l].numpy() for l in layers}


def fit_patch_pca(bb, frames, size_hw, out_path, per_frame=64, seed=0, batch=16, log=print):
    """frames: list of (Sequence, frame_index) from reference drives."""
    rng = np.random.default_rng(seed)
    bank = {b: [] for b in PATCH_BLOCKS}
    ims = []
    t0 = time.time()

    def flush():
        layers, g, pat = run_backbone_raw(bb, ims)
        for b in PATCH_BLOCKS:
            x = pat[b]
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
            for i in range(len(x)):
                bank[b].append(x[i, rng.choice(x.shape[1], per_frame, replace=False)])

    for seq, i in frames:
        ims.append(resize(load_frame(seq.frames[i], size_hw), size_hw))
        if len(ims) == batch:
            flush(); ims = []
    if ims:
        flush()
    tr = {}
    for b in PATCH_BLOCKS:
        X = np.concatenate(bank[b]).astype(np.float64)
        mu = X.mean(0)
        ev, V = np.linalg.eigh(np.cov((X - mu).T))
        tr[f"mu_{b}"] = mu.astype(np.float32)
        tr[f"W_{b}"] = V[:, ::-1][:, :P_DIM].astype(np.float32)
        tr[f"ev_{b}"] = ev[::-1].astype(np.float32)
        log(f"block {b}: top-{P_DIM} PCs explain {ev[::-1][:P_DIM].sum() / ev.sum():.3f} of unit-token variance")
    np.savez(out_path, **tr)
    log(f"patch PCA fitted on {sum(len(v) for v in bank.values())} tokens in {time.time() - t0:.0f}s")
    return tr


def project_unit(x, tr, b):
    u = x / np.linalg.norm(x, axis=-1, keepdims=True) - tr[f"mu_{b}"]
    p = u @ tr[f"W_{b}"]
    r = (u * u).sum(-1) - (p * p).sum(-1)
    return p, r


def extract_clip_v2(bb, seq: Sequence, start, end, size_hw, tr, out_path, corruption=None, severity=None, batch=16):
    G, PU, PR, idx = [], {b: [] for b in PATCH_BLOCKS}, {b: [] for b in PATCH_BLOCKS}, []
    seed = stable_hash(seq.seq_id + "|" + str(corruption))
    ims, ids = [], []
    tb = 0.0

    def flush():
        nonlocal tb
        t = time.time()
        layers, g, pat = run_backbone_raw(bb, ims)
        tb += time.time() - t
        G.append(g.astype(np.float16)); idx.extend(ids)
        for b in PATCH_BLOCKS:
            p, r = project_unit(pat[b], tr, b)
            PU[b].append(p.astype(np.float16)); PR[b].append(r.astype(np.float16))

    for i in range(start, end):
        im = resize(load_frame(seq.frames[i], size_hw), size_hw)
        if corruption is not None:
            im = apply_corruption(im, corruption, severity, seed, i)
        ims.append(im); ids.append(i)
        if len(ims) == batch:
            flush(); ims, ids = [], []
    if ims:
        flush()
    arrs = {"glob": np.concatenate(G), "frame_idx": np.array(idx)}
    for b in PATCH_BLOCKS:
        arrs[f"pu{b}"] = np.concatenate(PU[b]); arrs[f"pr{b}"] = np.concatenate(PR[b])
    np.savez(out_path, **arrs)
    return tb / max(len(idx), 1)
