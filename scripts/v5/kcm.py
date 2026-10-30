"""FROZEN known-condition monitor (KCM) — FROZEN_METHOD.md. Do not edit after the freeze commit.

phi_B(t) = concat_b in B [ l2(CLS_b(t)), l2(meanpatch_b(t)) ]           (default B = blocks 2,4,6,8 -> layer idx 0..3)
w        = logistic regression (C=0.1, L2, lbfgs, max_iter 5000) on StandardScaler(phi), four equally weighted groups:
           BDD synthetic-corrupted normT frames (+), BDD clean normT frames (-), ACDC adverse frames (+), ACDC normal (-)
d(t)     = w^T phi(t), robustly standardised (median / 1.4826 MAD) on in-control normalisation frames (BDD normS)
S(t)     = EWMA_0.1(d)(t) / sqrt(0.1/1.9), started at t = 20; alarm at first t >= 20 with S(t) > h
h        = drive-level split-conformal threshold (alpha = 0.05) of max_{t in [20,100)} S(t) over calibration drives
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LAM = 0.1; FAC = np.sqrt(LAM / (2 - LAM)); MS = 20; C_REG = 0.1


def l2n(x):
    x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def phi(glob, idx=(0, 1, 2, 3)):
    g = np.asarray(glob, np.float32)
    return np.concatenate([np.concatenate([l2n(g[:, i, 0]), l2n(g[:, i, 1])], 1) for i in idx], 1)


def robust_ls(x):
    med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med)); return med, max(mad, 1e-6)


def ewma(x, start=MS, lam=LAM):
    out = np.full(len(x), -np.inf); e = 0.0
    for t in range(start, len(x)):
        e = (1 - lam) * e + lam * x[t]; out[t] = e
    return out


class KCM:
    def __init__(self, idx=(0, 1, 2, 3), C=C_REG):
        self.idx, self.C = tuple(idx), C

    def fit(self, groups):
        """groups: list of (glob_or_phi_array, label, is_phi)"""
        Xs = [g[0] if g[2] else phi(g[0], self.idx) for g in groups]
        X = np.concatenate(Xs); y = np.concatenate([np.full(len(x), g[1]) for x, g in zip(Xs, groups)])
        w = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in Xs]) * len(X) / len(Xs)
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc.transform(X), y, sample_weight=w)
        self.n_train = [len(x) for x in Xs]
        return self

    def raw(self, glob):
        return self.lr.decision_function(self.sc.transform(phi(glob, self.idx)))

    def normalise(self, norm_globs):
        self.mu, self.sd = robust_ls(np.concatenate([self.raw(g) for g in norm_globs])); return self

    def stat(self, glob):
        return ewma((self.raw(glob) - self.mu) / self.sd) / FAC


def drive_max(s, a=MS, b=100):
    x = s[a:b]; x = x[np.isfinite(x)]; return x.max() if len(x) else -np.inf


def conformal(maxima, alpha=0.05):
    m = np.sort(np.asarray(maxima)); n = len(m); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(m[k - 1]) if k <= n else float("inf")
