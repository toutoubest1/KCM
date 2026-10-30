"""Feature extraction with reference-fitted projections.

Stage 1 (``fit_reference``): run the frozen backbone on the *reference-fit* frames,
keep their raw tokens (fp16 memmap, enables later re-fitting for sensitivity
studies) and fit, for each layer l,

  robust  : coordinatewise median m_l / MAD s_l over reference patch tokens,
            spatial-sign PCA basis W_l^R (D x K) of z = (x-m_l)/s_l
  classic : coordinatewise mean / std, ordinary PCA basis W_l^C

Stage 2 (``extract_clip``): for every other clip (a frame range of one sequence,
possibly corrupted) store per frame and layer
  * raw global tokens  g = [CLS, mean-patch]           (L, 2, D)   fp16
  * robust patch projections u^R = W^R' z^R            (L, N, K)   fp16
  * robust residual energy  ||z^R||^2 - ||u^R||^2      (L, N)      fp32
  * the same for the classical transform (u^C, r^C)
No covariance matrix is ever inverted; only D x D eigen-decompositions.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .data import Sequence, resize, load_frame, stable_hash
from .corruptions import apply_corruption

K_MAX = 32


def _to_tensor(ims: list[np.ndarray]) -> torch.Tensor:
    x = np.stack(ims).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def frames_for(seq: Sequence, start: int, end: int, size_hw, corruption=None, severity=None):
    seed = stable_hash(seq.seq_id + "|" + str(corruption))
    for i in range(start, end):
        im = resize(load_frame(seq.frames[i], size_hw), size_hw)
        if corruption is not None:
            im = apply_corruption(im, corruption, severity, seed, i)
        yield i, im


def run_backbone(bb, ims: list[np.ndarray]):
    out = bb.forward_layers(_to_tensor(ims))
    layers = sorted(out)
    cls = torch.stack([out[l][0] for l in layers], 1)            # B,L,D
    pat = torch.stack([out[l][1] for l in layers], 1)            # B,L,N,D
    glob = torch.stack([cls, pat.mean(2)], 2)                     # B,L,2,D
    return glob.numpy(), pat.numpy()


def fit_reference(bb, ref: list[tuple[Sequence, list[int]]], size_hw, out_dir: str, batch: int = 16,
                  n_patch_fit: int = 200_000, seed: int = 0, log=print):
    """ref: list of (sequence, frame indices). Saves raw tokens + transforms."""
    os.makedirs(out_dir, exist_ok=True)
    n = sum(len(ix) for _, ix in ref)
    ims, meta = [], []
    raw_p = raw_g = None
    k = 0
    t0 = time.time()
    for seq, ix in ref:
        for i in ix:
            ims.append(resize(load_frame(seq.frames[i], size_hw), size_hw))
            meta.append((seq.seq_id, i))
            if len(ims) == batch:
                g, p = run_backbone(bb, ims)
                if raw_p is None:
                    raw_p = np.lib.format.open_memmap(os.path.join(out_dir, "ref_patches.npy"), "w+", np.float16, (n,) + p.shape[1:])
                    raw_g = np.zeros((n,) + g.shape[1:], np.float32)
                raw_p[k:k + len(ims)] = p
                raw_g[k:k + len(ims)] = g
                k += len(ims)
                ims = []
    if ims:
        g, p = run_backbone(bb, ims)
        if raw_p is None:
            raw_p = np.lib.format.open_memmap(os.path.join(out_dir, "ref_patches.npy"), "w+", np.float16, (n,) + p.shape[1:])
            raw_g = np.zeros((n,) + g.shape[1:], np.float32)
        raw_p[k:k + len(ims)] = p
        raw_g[k:k + len(ims)] = g
        k += len(ims)
    raw_p.flush()
    np.save(os.path.join(out_dir, "ref_globals.npy"), raw_g.astype(np.float16))
    np.save(os.path.join(out_dir, "ref_meta.npy"), np.array(meta, dtype=object), allow_pickle=True)
    log(f"reference extraction: {n} frames in {time.time()-t0:.1f}s")
    tr = fit_patch_transforms(raw_p, n_patch_fit, seed)
    np.savez(os.path.join(out_dir, "transforms.npz"), **tr)
    return tr


def fit_patch_transforms(raw_p, n_patch_fit=200_000, seed=0):
    n, L, N, D = raw_p.shape
    rng = np.random.default_rng(seed)
    tr = {}
    per_frame = min(N, max(1, n_patch_fit // n))
    for l in range(L):
        X = np.concatenate([raw_p[f, l, rng.choice(N, per_frame, replace=False)] for f in range(n)]).astype(np.float64)
        # robust
        med = np.median(X, 0)
        mad = 1.4826 * np.median(np.abs(X - med), 0)
        mad = np.maximum(mad, 1e-3 * np.median(mad) + 1e-8)
        Z = (X - med) / mad
        S = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)   # spatial signs
        ev, V = np.linalg.eigh(S.T @ S / len(S))
        Wr = V[:, ::-1][:, :K_MAX]
        # classical
        mu = X.mean(0)
        sd = X.std(0) + 1e-8
        Zc = (X - mu) / sd
        evc, Vc = np.linalg.eigh(Zc.T @ Zc / len(Zc))
        Wc = Vc[:, ::-1][:, :K_MAX]
        tr[f"rob_loc_{l}"], tr[f"rob_scale_{l}"], tr[f"rob_W_{l}"] = med, mad, Wr
        tr[f"cls_loc_{l}"], tr[f"cls_scale_{l}"], tr[f"cls_W_{l}"] = mu, sd, Wc
        tr[f"rob_ev_{l}"], tr[f"cls_ev_{l}"] = ev[::-1][:K_MAX], evc[::-1][:K_MAX]
    return tr


def project_patches(p: np.ndarray, tr: dict, prefix: str):
    """p: [B,L,N,D] raw -> u [B,L,N,K], r [B,L,N]."""
    B, L, N, D = p.shape
    U = np.empty((B, L, N, K_MAX), np.float32)
    R = np.empty((B, L, N), np.float32)
    for l in range(L):
        z = (p[:, l].astype(np.float32) - tr[f"{prefix}_loc_{l}"]) / tr[f"{prefix}_scale_{l}"]
        u = z @ tr[f"{prefix}_W_{l}"].astype(np.float32)
        U[:, l] = u
        R[:, l] = (z ** 2).sum(-1) - (u ** 2).sum(-1)
    return U, R


def extract_clip(bb, seq: Sequence, start: int, end: int, size_hw, tr: dict, out_path: str,
                 corruption=None, severity=None, batch: int = 16, rob_only: bool = False):
    G, UR, RR, UC, RC, idx = [], [], [], [], [], []
    ims, ids = [], []
    t_bb = 0.0
    n = 0

    def flush():
        nonlocal t_bb, n
        t = time.time()
        g, p = run_backbone(bb, ims)
        t_bb += time.time() - t
        n += len(ims)
        ur, rr = project_patches(p, tr, "rob")
        G.append(g.astype(np.float16)); UR.append(ur.astype(np.float16)); RR.append(rr); idx.extend(ids)
        if not rob_only:
            uc, rc = project_patches(p, tr, "cls")
            UC.append(uc.astype(np.float16)); RC.append(rc)

    for i, im in frames_for(seq, start, end, size_hw, corruption, severity):
        ims.append(im); ids.append(i)
        if len(ims) == batch:
            flush(); ims, ids = [], []
    if ims:
        flush()
    extra = {} if rob_only else {"u_cls": np.concatenate(UC), "r_cls": np.concatenate(RC)}
    np.savez(out_path, glob=np.concatenate(G), u_rob=np.concatenate(UR), r_rob=np.concatenate(RR),
             frame_idx=np.array(idx), **extra)
    return t_bb / max(n, 1)
