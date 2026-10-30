"""Baseline online detectors, all operating on the SAME frozen features.

Unless stated otherwise baselines use the standard frame embedding = final-layer
CLS token (after final LayerNorm), fitted on the same reference-fit frames and
normalised on the same held-out clean frames as RMRM.  Every detector returns a
causal statistic trajectory; an alarm is raised at the first t with stat_t > h.

  hotelling_shewhart : T^2 on whitened classical PCA(k=32) coordinates, per frame
  mewma              : multivariate EWMA (Lowry et al. 1992) of whitened PCA coords
  cusum              : standard CUSUM on the standardised T^2 score
  ewma               : EWMA of the standardised T^2 score
  cosine / cosine_ewma : cosine distance to reference mean embedding
  knn / knn_ewma     : deep-kNN OOD score (Sun et al., ICML 2022), k-th NN distance of
                       L2-normalised embeddings to the reference bank
  mmd                : sliding-window unbiased MMD^2 (Gaussian kernel, median heuristic,
                       Gretton et al. 2012) between the last w frames and a reference sample
  ks_bonf            : "Failing loudly" (Rabanser et al., NeurIPS 2019): PCA + per-dim
                       two-sample KS tests on a sliding window, Bonferroni aggregated
  mewma_multilayer   : MEWMA on concatenated whitened PCA coords of all layers' globals
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def _whiten_pca(X, k):
    mu = X.mean(0)
    Xc = X - mu
    ev, V = np.linalg.eigh(np.cov(Xc.T))
    ev, V = ev[::-1][:k], V[:, ::-1][:, :k]
    return mu, V / np.sqrt(np.maximum(ev, 1e-10))


def _ewma(x, lam):
    out = np.empty_like(x, dtype=np.float64)
    s = 0.0 if x.ndim == 1 else np.zeros(x.shape[1])
    for t in range(len(x)):
        s = lam * x[t] + (1 - lam) * s
        out[t] = s
    return out


def _cusum(x, kappa):
    out = np.empty(len(x)); s = 0.0
    for t in range(len(x)):
        s = max(0.0, s + x[t] - kappa); out[t] = s
    return out


def final_cls(b, layer_pos=-1):
    return np.asarray(b["glob"][:, layer_pos, 0], np.float64)


class Baseline:
    """Names ending in `_wu` are warm-up (self-referenced) variants added in v1 for a fair comparison
    with RMRM-v1: the scalar score is recentred on the stream's first `warmup` frames (median) and
    scaled by the within-stream MAD estimated on the clean norm streams, then fed to a CUSUM;
    `mewma_wu` centres the whitened PCA coordinates at the warm-up mean."""

    def __init__(self, name, k=32, lam=0.1, kappa=0.5, window=20, n_ref_mmd=300, knn_k=10, seed=0, warmup=20):
        self.name, self.k, self.lam, self.kappa, self.w = name, k, lam, kappa, window
        self.warmup = warmup if name.endswith("_wu") else 0
        self.n_ref_mmd, self.knn_k, self.seed = n_ref_mmd, knn_k, seed

    def _emb(self, b):
        if self.name == "mewma_multilayer":
            g = np.asarray(b["glob"][:, :, 0], np.float64)  # CLS of every stored layer
            return g.reshape(len(g), -1)
        return final_cls(b)

    def fit(self, ref, norm_streams):
        X = self._emb(ref)
        if self.name == "mewma_multilayer":
            # per-layer standardisation then PCA of concatenation
            self.sd = X.std(0) + 1e-6
            X = X / self.sd
        self.mu, self.P = _whiten_pca(X, self.k)
        self.Xref = X
        self.ref_mean = X.mean(0)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.bank = Xn
        Y = (X - self.mu) @ self.P
        rng = np.random.default_rng(self.seed)
        self.Yref = Y[rng.choice(len(Y), min(self.n_ref_mmd, len(Y)), replace=False)]
        d2 = ((self.Yref[:, None] - self.Yref[None]) ** 2).sum(-1)
        self.bw2 = np.median(d2[d2 > 0])
        Kxx = np.exp(-d2 / self.bw2)
        m = len(self.Yref)
        self.kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        # standardisation of T^2 on held-out clean data (for cusum/ewma)
        t2 = np.concatenate([self._t2(b) for b in norm_streams])
        self.t2_mu, self.t2_sd = t2.mean(), t2.std() + 1e-9
        if self.warmup and self.name != "mewma_wu":
            m = self.warmup
            res = []
            for b in norm_streams:
                sc = self._raw_scalar(b)
                res.append(sc[m:] - np.median(sc[:m]))
            res = np.concatenate(res)
            self.wu_sc = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
        return self

    def _raw_scalar(self, b):
        n = self.name
        X, Y = self._proj(b)
        if n.startswith("cusum"):
            return (Y ** 2).sum(1)
        if n.startswith("knn"):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            d = np.sqrt(np.maximum(2 - 2 * Xn @ self.bank.T, 0))
            return np.partition(d, self.knn_k - 1, axis=1)[:, self.knn_k - 1]
        if n.startswith("cosine"):
            # self-referenced: cosine distance to the warm-up mean embedding
            mu = X[:self.warmup].mean(0)
            return 1 - (X @ mu) / (np.linalg.norm(X, axis=1) * np.linalg.norm(mu))
        raise ValueError(n)

    def _proj(self, b):
        X = self._emb(b)
        if self.name == "mewma_multilayer":
            X = X / self.sd
        return X, (X - self.mu) @ self.P

    def _t2(self, b):
        return (self._proj(b)[1] ** 2).sum(1)

    def score_stream(self, b):
        n = self.name
        if n.endswith("_wu"):
            m = self.warmup
            out = np.full(len(b["glob"]), -np.inf)
            if n == "mewma_wu":
                X, Y = self._proj(b)
                Yc = Y[m:] - Y[:m].mean(0)
                Z = _ewma(Yc, self.lam)
                out[m:] = (Z ** 2).sum(1) * (2 - self.lam) / self.lam
                return out, {}
            sc = self._raw_scalar(b)
            zz = (sc[m:] - np.median(sc[:m])) / self.wu_sc
            out[m:] = _cusum(zz, self.kappa)
            return out, {}
        X, Y = self._proj(b)
        if n == "hotelling_shewhart":
            return (Y ** 2).sum(1), {}
        if n in ("mewma", "mewma_multilayer"):
            Z = _ewma(Y, self.lam)
            return (Z ** 2).sum(1) * (2 - self.lam) / self.lam, {}
        if n in ("cusum", "ewma"):
            s = ((Y ** 2).sum(1) - self.t2_mu) / self.t2_sd
            return (_cusum(s, self.kappa) if n == "cusum" else _ewma(s, self.lam)), {}
        if n.startswith("cosine"):
            cs = 1 - (X @ self.ref_mean) / (np.linalg.norm(X, axis=1) * np.linalg.norm(self.ref_mean))
            return (cs if n == "cosine" else _ewma(cs, self.lam)), {}
        if n.startswith("knn"):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            d = np.sqrt(np.maximum(2 - 2 * Xn @ self.bank.T, 0))
            kd = np.partition(d, self.knn_k - 1, axis=1)[:, self.knn_k - 1]
            return (kd if n == "knn" else _ewma(kd, self.lam)), {}
        if n == "mmd":
            T = len(Y); out = np.full(T, -np.inf)
            kyx = np.exp(-((Y[:, None] - self.Yref[None]) ** 2).sum(-1) / self.bw2).mean(1)  # T
            kyy = np.exp(-((Y[:, None] - Y[None]) ** 2).sum(-1) / self.bw2)
            for t in range(4, T):
                s = max(0, t - self.w + 1); n_ = t - s + 1
                K = kyy[s:t + 1, s:t + 1]
                out[t] = (K.sum() - n_) / (n_ * (n_ - 1)) + self.kxx - 2 * kyx[s:t + 1].mean()
            return out, {}
        if n == "ks_bonf":
            T = len(Y); out = np.full(T, -np.inf)
            Yr = self.Yref
            for t in range(4, T):
                s = max(0, t - self.w + 1)
                pmin = min(ks_2samp(Y[s:t + 1, j], Yr[:, j]).pvalue for j in range(Y.shape[1]))
                out[t] = -np.log10(max(pmin * Y.shape[1], 1e-300))
            return out, {}
        raise ValueError(n)


BASELINES = ["hotelling_shewhart", "mewma", "cusum", "ewma", "cosine", "cosine_ewma",
             "knn", "knn_ewma", "mmd", "ks_bonf", "mewma_multilayer"]
BASELINES_WU = ["cusum_wu", "knn_cusum_wu", "cosine_cusum_wu", "mewma_wu"]
