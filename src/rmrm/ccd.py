"""Stage-2: channel scores on raw DINOv2 token geometry, evidence normalisation, detectors and calibration.

Channel scores (computed once per clip, cached):
  g{b}  : log distance of the frame's L2-normalised raw block-b token [CLS, mean-patch] to its k-th nearest
          neighbour in the reference bank (reference-partition drives only)          b in stored blocks
  pmap{b}: per-position log distance of the (L2-normalised, reference-centred, PCA-64 projected) patch token
          to its k-th nearest reference patch at the same grid position               b in {6, 8}
Frame-level patch channels p{b}top / p{b}mean are derived from pmap after per-position robust normalisation on
the norm partition. No coordinatewise standardisation or clipping of token coordinates is used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .extract_v2 import PATCH_BLOCKS


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def kth_nn_logdist(Q, B, k):
    """Q [T,d], B [n,d] (both unit or arbitrary) -> log squared distance to k-th NN, [T]."""
    d2 = (Q * Q).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * Q @ B.T
    return np.log(np.partition(np.maximum(d2, 1e-12), k - 1, axis=1)[:, k - 1])


class ChannelScorer:
    def __init__(self, stored, k=10, patch_ref_stride=10, glob_ref_stride=2):
        self.stored = list(stored); self.k = k
        self.prs, self.grs = patch_ref_stride, glob_ref_stride

    def fit(self, ref_clips):
        """ref_clips: list of clip dicts (glob, pu{b}, pr{b})."""
        G = np.concatenate([c["glob"][::self.grs] for c in ref_clips])           # n, L, 2, D
        self.gbank = {i: l2n(G[:, i].reshape(len(G), -1)) for i in range(len(self.stored))}
        self.pbank = {}
        for b in PATCH_BLOCKS:
            pu = np.concatenate([c[f"pu{b}"][::self.prs] for c in ref_clips]).astype(np.float32)   # n,N,P
            pr = np.concatenate([c[f"pr{b}"][::self.prs] for c in ref_clips]).astype(np.float32)   # n,N
            v = np.concatenate([pu, np.sqrt(np.maximum(pr, 0))[..., None]], -1)                    # n,N,P+1
            self.pbank[b] = np.ascontiguousarray(v.transpose(1, 0, 2))                             # N,n,P+1
        return self

    def score_clip(self, c):
        out = {}
        T = len(c["glob"])
        for i, b in enumerate(self.stored):
            q = l2n(np.asarray(c["glob"][:, i], np.float32).reshape(T, -1))
            out[f"g{b}"] = kth_nn_logdist(q, self.gbank[i], self.k).astype(np.float32)
        for b in PATCH_BLOCKS:
            v = np.concatenate([np.asarray(c[f"pu{b}"], np.float32),
                                np.sqrt(np.maximum(np.asarray(c[f"pr{b}"], np.float32), 0))[..., None]], -1)  # T,N,P+1
            Q = np.ascontiguousarray(v.transpose(1, 0, 2))                    # N,T,d
            B = self.pbank[b]; bb = (B * B).sum(-1)[:, None, :]; Bt = B.transpose(0, 2, 1)
            res = np.empty((Q.shape[0], T), np.float32)
            for t0 in range(0, T, 20):
                q = Q[:, t0:t0 + 20]
                d2 = (q * q).sum(-1)[..., None] + bb - 2 * np.matmul(q, Bt)
                res[:, t0:t0 + 20] = np.partition(np.maximum(d2, 1e-12), self.k - 1, axis=-1)[..., self.k - 1]
            out[f"pmap{b}"] = np.log(res).T.astype(np.float16)  # T,N
        return out


def stillness_maps(c):
    """Per-position temporal change of the patch token relative to the frame's median change:
    st[t,j] = log||x_t,j - x_{t-1,j}||^2 - median_j(.)  (x = projected unit token + residual magnitude).
    A camera-fixed occluder makes a region locally static while the rest of the scene moves (low st);
    ego standstill makes the whole frame static, which this relative measure does not flag."""
    out = {}
    for b in PATCH_BLOCKS:
        v = np.concatenate([np.asarray(c[f"pu{b}"], np.float32),
                            np.sqrt(np.maximum(np.asarray(c[f"pr{b}"], np.float32), 0))[..., None]], -1)
        d = np.zeros(v.shape[:2], np.float32)
        d[1:] = np.log(((v[1:] - v[:-1]) ** 2).sum(-1) + 1e-6)
        d[1:] -= np.median(d[1:], axis=1, keepdims=True)
        out[f"st{b}"] = d.astype(np.float16)
    return out


# ----------------------------------------------------------------------------------------- evidence
def robust_ls(x, axis=0):
    med = np.median(x, axis=axis)
    mad = 1.4826 * np.median(np.abs(x - np.expand_dims(med, axis)), axis=axis)
    return med, np.maximum(mad, 1e-6)


class Evidence:
    """Frame-level channel evidence z (robust standardisation of channel scores on the NORM partition)."""

    def __init__(self, q_top=0.1):
        self.q_top = q_top

    def fit(self, norm_scores):
        self.pos = {}
        for b in PATCH_BLOCKS:
            M = np.concatenate([s[f"pmap{b}"] for s in norm_scores]).astype(np.float32)
            self.pos[b] = robust_ls(M, 0)                                         # per-position centre/scale
            if all(f"st{b}" in x for x in norm_scores):
                S = np.concatenate([np.asarray(x[f"st{b}"], np.float32)[1:] for x in norm_scores])
                self.pos[f"st{b}"] = robust_ls(S, 0)
        X = np.concatenate([self.raw(s) for s in norm_scores])
        self.names = self._names
        self.loc, self.sc = robust_ls(X, 0)
        Z = (X - self.loc) / self.sc
        r = np.argsort(np.argsort(Z, 0), 0).astype(float)
        R = 2 * np.sin(np.pi * np.corrcoef(r.T) / 6)
        w, V = np.linalg.eigh(R); self.R = (V * np.maximum(w, 1e-4)) @ V.T
        self.norm_Z = Z
        return self

    def raw(self, s):
        cols, names = [], []
        gk = sorted((k for k in s if k.startswith("g") and k[1:].isdigit()), key=lambda k: int(k[1:]))
        for k in gk:
            cols.append(np.asarray(s[k], np.float32)); names.append(k)
        for b in PATCH_BLOCKS:
            med, mad = self.pos[b]
            zm = (np.asarray(s[f"pmap{b}"], np.float32) - med) / mad
            m = max(1, int(np.ceil(self.q_top * zm.shape[1])))
            cols.append(np.partition(zm, zm.shape[1] - m, axis=1)[:, -m:].mean(1)); names.append(f"p{b}top")
            cols.append(zm.mean(1)); names.append(f"p{b}mean")
        self._names = names
        return np.stack(cols, 1)

    def z(self, s):
        return (self.raw(s) - self.loc) / self.sc

    def idx(self, names):
        return [self.names.index(n) for n in names]

    def zstill(self, s, b):
        med, mad = self.pos[f"st{b}"]
        return -(np.asarray(s[f"st{b}"], np.float32) - med) / mad      # positive = locally more static than usual

    def zmap(self, s, b):
        med, mad = self.pos[b]
        return (np.asarray(s[f"pmap{b}"], np.float32) - med) / mad


# ----------------------------------------------------------------------------------------- detectors
def ewma(x, lam, start):
    out = np.full(len(x), -np.inf); e = 0.0
    for t in range(start, len(x)):
        e = (1 - lam) * e + lam * x[t]; out[t] = e
    return out


@dataclass
class DetSpec:
    """kind: single | equal | max | ccd_diff | ccd_resid | ccd_lr | ccd_gate
    cond / cont: channel names. lam_ewma: EWMA smoothing. lam_c: content coefficient for ccd_diff."""
    name: str
    kind: str
    cond: tuple = ()
    cont: tuple = ()
    lam_ewma: float = 0.1
    lam_c: float = 1.0
    extra: dict = field(default_factory=dict)


class Detector:
    def __init__(self, spec: DetSpec, ev: Evidence, start: int):
        self.s, self.ev, self.start = spec, ev, start

    def fit(self, norm_Z, lr_data=None):
        s = self.s
        ic = self.ev.idx(s.cond) if s.cond else []; iq = self.ev.idx(s.cont) if s.cont else []
        self.ic, self.iq = ic, iq
        Z = norm_Z
        if s.kind in ("equal", "ccd_diff", "ccd_resid", "ccd_gate"):
            c = Z[:, ic].mean(1)
            q = Z[:, iq].mean(1) if iq else np.zeros(len(Z))
            if s.kind == "ccd_resid":
                # robust (Theil-Sen-like via median of pairwise slopes on a subsample) regression of c on q
                rng = np.random.default_rng(0); i = rng.choice(len(c), min(4000, len(c)), replace=False)
                j = rng.permutation(i)
                dq = q[i] - q[j]; ok = np.abs(dq) > 1e-3
                self.beta = float(np.median((c[i] - c[j])[ok] / dq[ok]))
                r = c - self.beta * q
            elif s.kind == "ccd_diff":
                self.beta = s.lam_c; r = c - self.beta * q
            else:
                self.beta = 0.0; r = c
            self.mu_r, self.sd_r = robust_ls(r, 0)
            if s.kind == "ccd_gate":
                self.mu_q, self.sd_q = robust_ls(q, 0)
        if s.kind == "ccd_resid_max":
            q = Z[:, iq].mean(1); rng = np.random.default_rng(0)
            i = rng.choice(len(q), min(4000, len(q)), replace=False); j = rng.permutation(i); dq = q[i] - q[j]; ok = np.abs(dq) > 1e-3
            self.beta_l = np.array([np.median((Z[i, c] - Z[j, c])[ok] / dq[ok]) for c in ic])
            Rz = Z[:, ic] - q[:, None] * self.beta_l
            self.mu_l, self.sd_l = robust_ls(Rz, 0)
        if s.kind in ("ccd_lr", "ccd_lr_resid"):
            X, y = lr_data
            from sklearn.linear_model import LogisticRegression
            self.lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[:, ic], y)
            sc = self.lr.decision_function(Z[:, ic])
            if s.kind == "ccd_lr_resid":
                q = Z[:, iq].mean(1); rng = np.random.default_rng(0)
                i = rng.choice(len(q), min(4000, len(q)), replace=False); j = rng.permutation(i); dq = q[i] - q[j]; ok = np.abs(dq) > 1e-3
                self.beta = float(np.median((sc[i] - sc[j])[ok] / dq[ok])); sc = sc - self.beta * q
            self.mu_r, self.sd_r = robust_ls(sc, 0)
        return self

    def frame_evidence(self, Z):
        s = self.s
        if s.kind == "single":
            return Z[:, self.ic[0]]
        if s.kind == "max":
            return Z[:, self.ic]
        if s.kind in ("equal", "ccd_diff", "ccd_resid", "ccd_gate"):
            c = Z[:, self.ic].mean(1)
            q = Z[:, self.iq].mean(1) if self.iq else 0.0
            return ((c - self.beta * q) - self.mu_r) / self.sd_r
        if s.kind == "ccd_lr":
            return (self.lr.decision_function(Z[:, self.ic]) - self.mu_r) / self.sd_r
        if s.kind == "ccd_lr_resid":
            return (self.lr.decision_function(Z[:, self.ic]) - self.beta * Z[:, self.iq].mean(1) - self.mu_r) / self.sd_r
        raise ValueError(s.kind)

    def stat(self, Z, scores=None):
        lam = self.s.lam_ewma; f = np.sqrt(lam / (2 - lam))
        if self.s.kind == "still_scan":
            blocks = self.s.extra.get("blocks", [6, 8])
            zm = np.clip(np.mean([self.ev.zstill(scores, b) for b in blocks], 0), -3.0, None)
            out = np.full(len(zm), -np.inf); e = np.zeros(zm.shape[1])
            for t in range(self.start, len(zm)):
                e = (1 - lam) * e + lam * zm[t]; out[t] = e.max() / f
            return out
        if self.s.kind == "ccd_resid_max":
            Rz = (Z[:, self.ic] - Z[:, self.iq].mean(1, keepdims=True) * self.beta_l - self.mu_l) / self.sd_l
            return np.max(np.stack([ewma(Rz[:, j], lam, self.start) for j in range(Rz.shape[1])], 1), 1) / f
        if self.s.kind == "patch_scan":
            # spatial scan of per-position EWMAs: a persistent localised anomaly (e.g. a lens occluder) keeps
            # the same grid positions atypical over time, whereas moving scene content does not.
            blocks = self.s.extra.get("blocks", [6, 8])
            zm = np.mean([self.ev.zmap(scores, b) for b in blocks], 0)                       # T,N
            if self.s.extra.get("clip_neg", True):
                zm = np.maximum(zm, -3.0)
            out = np.full(len(zm), -np.inf); e = np.zeros(zm.shape[1])
            for t in range(self.start, len(zm)):
                e = (1 - lam) * e + lam * zm[t]; out[t] = e.max() / f
            return out
        e = self.frame_evidence(Z)
        if self.s.kind == "max":
            return np.max(np.stack([ewma(e[:, j], lam, self.start) for j in range(e.shape[1])], 1), 1) / f
        st = ewma(e, lam, self.start) / f
        if self.s.kind == "ccd_gate":
            # condition statistic counts only while the content statistic is NOT itself atypical
            c = Z[:, self.iq].mean(1); qs = ewma((c - self.mu_q) / self.sd_q, lam, self.start) / f
            st = np.where(qs > self.s.extra.get("gate", 2.0), -np.inf, st)
        return st


# ----------------------------------------------------------------------------------------- calibration
def drive_max(stat, start, end):
    s = stat[start:end]; s = s[np.isfinite(s)]
    return s.max() if len(s) else -np.inf


def conformal_threshold(maxima, alpha):
    """Drive-level split-conformal threshold: with n exchangeable calibration drives, a new drive's
    in-control maximum exceeds the ceil((n+1)(1-alpha))-th order statistic with probability <= alpha."""
    m = np.sort(np.asarray(maxima)); n = len(m)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(m[min(k, n) - 1]) if k <= n else float("inf")


def clopper_pearson(x, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else beta.ppf(a, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a, x + 1, n - x)
    return lo, hi
