"""Stage-4 two-branch detector library (METHOD_DEVELOPMENT_LOG_V4.md, Entry V4-0).

Branches (each returns a causal statistic trajectory, -inf before the monitoring start ms):
  D   directional known-condition branch: logistic direction on phi(t) = [l2(CLS_b), l2(meanpatch_b)], b=2,4,6,8
  N1  novelty: LR8 weighting of the 8 early/mid k-NN distance evidences (trained on corrupted vs clean normT frames)
  N0  novelty: untrained mean of the same 8 evidences
  ST  local stillness scan (untrained; BDD only)
  g2  single-layer novelty baseline; MEWMA multilayer baseline (stage-2 objects)
Combinations: C1 = max_i S_i / q_i (q_i: 95% quantile of in-control sequence maxima on the NORMALISATION role);
              C2 = max_i -log p_i(t) (p_i: upper-tail rank of S_i(t) among in-control statistic values of the
              NORMALISATION role). Thresholds: conformal on the CALIBRATION role. Training, normalisation and
              calibration roles are disjoint (CALIBRATION_AUDIT_V4.md).
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rmrm.ccd import l2n, robust_ls, ewma, Detector, DetSpec, drive_max, conformal_threshold  # noqa: E402

COND8 = ["g2", "g4", "g6", "g8", "p6top", "p6mean", "p8top", "p8mean"]
FAMILIES = {"F1_illumination": ["brightness", "contrast", "low_light"], "F2_blur": ["defocus_blur", "motion_blur"],
            "F3_noise": ["gaussian_noise"], "F4_weather": ["fog", "rain", "snow"], "F5_occlusion": ["occlusion"]}
REAL_CLOSE = {"F1_illumination": ["night"], "F4_weather": ["fog", "rain", "snow"]}
FAM_OF = {c: f for f, cs in FAMILIES.items() for c in cs}
LAM = 0.1; FAC = np.sqrt(LAM / (2 - LAM))
DIR_BLOCKS = [0, 1, 2, 3]


def phi(glob):
    g = np.asarray(glob, np.float32)
    return np.concatenate([np.concatenate([l2n(g[:, i, 0]), l2n(g[:, i, 1])], 1) for i in DIR_BLOCKS], 1)


class Dir:
    """D branch. groups: list of (X, label); equal total weight per group; C fixed 0.1."""
    def __init__(self, groups, C=0.1):
        X = np.concatenate([g[0] for g in groups]); y = np.concatenate([np.full(len(g[0]), g[1]) for g in groups])
        w = np.concatenate([np.full(len(g[0]), 1.0 / len(g[0])) for g in groups]) * len(X) / len(groups)
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=C, max_iter=5000).fit(self.sc.transform(X), y, sample_weight=w)
        self.sizes = [len(g[0]) for g in groups]

    def raw(self, glob):
        return self.lr.decision_function(self.sc.transform(phi(glob)))

    def standardise(self, norm_globs):
        self.mu, self.sd = robust_ls(np.concatenate([self.raw(g) for g in norm_globs]), 0); return self

    def stat(self, scores, Z, glob, ms):
        return ewma((self.raw(glob) - self.mu) / self.sd, LAM, ms) / FAC


class N1:
    def __init__(self, names, Zpos, Zneg):
        self.ic = [names.index(n) for n in COND8]
        X = np.concatenate([Zpos, Zneg])[:, self.ic]; y = np.r_[np.ones(len(Zpos)), np.zeros(len(Zneg))]
        self.lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)

    def standardise(self, Znorm):
        self.mu, self.sd = robust_ls(self.lr.decision_function(Znorm[:, self.ic]), 0); return self

    def stat(self, scores, Z, glob, ms):
        return ewma((self.lr.decision_function(Z[:, self.ic]) - self.mu) / self.sd, LAM, ms) / FAC


class N0:
    def __init__(self, names):
        self.ic = [names.index(n) for n in COND8]

    def standardise(self, Znorm):
        self.mu, self.sd = robust_ls(Znorm[:, self.ic].mean(1), 0); return self

    def stat(self, scores, Z, glob, ms):
        return ewma((Z[:, self.ic].mean(1) - self.mu) / self.sd, LAM, ms) / FAC


class Wrap:
    """wrap a stage-2 Detector (single g2, still_scan) or a Baseline (MEWMA)"""
    def __init__(self, obj, kind):
        self.obj, self.kind = obj, kind

    def stat(self, scores, Z, glob, ms):
        if self.kind == "baseline":
            return np.r_[np.full(ms, -np.inf), self.obj.score_stream({"glob": glob[ms:]})[0]]
        return self.obj.stat(Z, scores)


def g2_branch(ev, ms):
    return Wrap(Detector(DetSpec("single_g2", "single", cond=("g2",)), ev, ms).fit(ev.norm_Z), "det")


def st_branch(ev, ms):
    return Wrap(Detector(DetSpec("still_scan_6_8", "still_scan", extra={"blocks": [6, 8]}), ev, ms).fit(ev.norm_Z), "det")


class Combo:
    """rule 'C1' (max of maxima-quantile-normalised statistics) or 'C2' (max of -log rank p-values)."""
    def __init__(self, branches, rule):
        self.br, self.rule = branches, rule

    def fit(self, norm_items, ms, end):
        """norm_items: list of dicts {branch_name: trajectory} on in-control NORMALISATION sequences"""
        if self.rule == "C1":
            self.q = {b: float(np.quantile([drive_max(it[b], ms, end) for it in norm_items], 0.95)) for b in self.br}
        else:
            self.ref = {b: np.sort(np.concatenate([it[b][ms:end][np.isfinite(it[b][ms:end])] for it in norm_items])) for b in self.br}
        return self

    def parts(self, traj):
        if self.rule == "C1":
            return {b: traj[b] / self.q[b] for b in self.br}
        out = {}
        for b in self.br:
            r = self.ref[b]; x = traj[b]; n = len(r)
            ge = n - np.searchsorted(r, np.where(np.isfinite(x), x, -1e30), side="left")
            p = (1 + ge) / (n + 1); v = -np.log(p); v[~np.isfinite(x)] = -np.inf; out[b] = v
        return out

    def stat(self, traj):
        return np.max(np.stack(list(self.parts(traj).values())), 0)


def threshold(cal_traj, ms, end, alpha=0.05):
    return conformal_threshold([drive_max(t, ms, end) for t in cal_traj], alpha)
