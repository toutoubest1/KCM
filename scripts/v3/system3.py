"""Stage-3 helpers: build the stage-2 CCD-MM system (unchanged code paths from scripts/v2/run_v2.py) and
re-instantiate it on another in-control population WITHOUT changing any learned weight.

  U0 : everything fitted on BDD (reference bank, evidence normalisation, LR weights, beta, normalisers);
       only the conformal threshold is set on the target domain's calibration sequences.
  U1 : the population statistics that define 'in control' are re-estimated on the target domain's
       normal-condition data (k-NN reference bank, patch PCA kept from BDD, per-position/robust evidence
       normalisation, score standardisation mu_r/sd_r, joint normalisers, MEWMA reference) while the learned
       condition model (LR coefficients and the content slope beta) is kept EXACTLY as trained on BDD.
"""
import copy, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v2"))
from rmrm.ccd import ChannelScorer, Evidence, Detector, DetSpec, drive_max, robust_ls  # noqa: E402
from rmrm.baselines import Baseline  # noqa: E402

# the stage-2 candidate and the pre-specified comparators (configs/v2/det_iter3.json, unchanged)
COND8 = ["g2", "g4", "g6", "g8", "p6top", "p6mean", "p8top", "p8mean"]
ALL10 = ["g2", "g4", "g6", "g8", "g10", "g12", "p6top", "p6mean", "p8top", "p8mean"]
DETS = [
    {"name": "mewma_multilayer", "kind": "baseline", "baseline": "mewma_multilayer"},
    {"name": "single_g2", "kind": "single", "cond": ["g2"]},
    {"name": "ccd_lr_all", "kind": "ccd_lr", "cond": ALL10},                     # H5 (no content residualisation)
    {"name": "ccd_lr_resid", "kind": "ccd_lr_resid", "cond": COND8, "cont": ["g10", "g12"]},   # H5r
    {"name": "still_scan_6_8", "kind": "still_scan", "extra": {"blocks": [6, 8]}},
    {"name": "ccdmm_lrresid_still", "kind": "joint", "components": ["ccd_lr_resid", "still_scan_6_8"]},  # CCD-MM v2
]


class System:
    """Holds fitted objects; stat(name, scores, glob) gives the detector trajectory."""

    def __init__(self, scorer, ev, dets, baselines, ms, tau):
        self.scorer, self.ev, self.dets, self.bl, self.ms, self.tau = scorer, ev, dets, baselines, ms, tau
        self.joint_s = {}

    def stat(self, name, scores, glob):
        cu = getattr(self, "custom", {})
        if name in cu:
            return cu[name].stat(glob, self.ms)
        d = next(x for x in DETS if x["name"] == name)
        if d["kind"] == "joint":
            return np.max(np.stack([self.stat(c, scores, glob) / self.joint_s[name][i]
                                    for i, c in enumerate(d["components"])]), 0)
        if d["kind"] == "baseline":
            b = self.bl[name]
            return np.r_[np.full(self.ms, -np.inf), b.score_stream({"glob": glob[self.ms:]})[0]]
        return self.dets[name].stat(self.ev.z(scores), scores)

    def fit_joint(self, norm_items, q=0.95, window=None):
        """norm_items: list of (scores, glob) in-control sequences; normaliser = q-quantile of component maxima."""
        a, b = window or (self.ms, self.tau)
        for d in DETS:
            if d["kind"] == "joint":
                s = []
                for c in d["components"]:
                    s.append(float(np.quantile([drive_max(self.stat(c, sc, gl), a, b) for sc, gl in norm_items], q)))
                self.joint_s[d["name"]] = s

    # ---------------------------------------------------------------- frame-level decomposition (diagnostic)
    def components(self, scores):
        """Per-frame diagnostic quantities: 10 channel evidences, H5 score, condition-LR score (8-ch, before
        residualisation), content score q, and the H5r residual score - all on the scale used by the detectors."""
        Z = self.ev.z(scores)
        out = {n: Z[:, i] for i, n in enumerate(self.ev.names)}
        d5 = self.dets["ccd_lr_all"]; out["H5_lr_all"] = d5.frame_evidence(Z)
        dr = self.dets["ccd_lr_resid"]
        lr8 = dr.lr.decision_function(Z[:, dr.ic]); q = Z[:, dr.iq].mean(1)
        out["LR8_cond"] = (lr8 - dr.mu_lr8) / dr.sd_lr8
        out["content_q"] = q
        out["H5r_resid"] = dr.frame_evidence(Z)
        return out


def fit_bdd(cfg, store, part, lr_data_fn):
    ids = lambda p: sorted(k for k, v in part.items() if v == p)
    ref_clips = [store.clip(f"{k}__clean") for k in ids("ref")]
    scorer = ChannelScorer(cfg["stored_layers"], k=cfg.get("knn_k", 10)).fit(ref_clips)
    ref_glob = {"glob": np.concatenate([c["glob"][::cfg["ref_stride"]] for c in ref_clips])}
    del ref_clips; store.c.clear()
    norm_s = [store.scores(f"{k}__clean", scorer) for k in ids("norm")]
    ev = Evidence(q_top=cfg.get("q_top", 0.1)).fit(norm_s)
    lr_data = lr_data_fn(ev, scorer)
    dets, bl = {}, {}
    for d in DETS:
        if d["kind"] == "baseline":
            b = Baseline(d["baseline"], **cfg.get("baseline_params", {}))
            b.fit(ref_glob, [{"glob": store.clip(f"{k}__clean")["glob"]} for k in ids("norm")]); bl[d["name"]] = b
        elif d["kind"] != "joint":
            dets[d["name"]] = Detector(DetSpec(**d), ev, cfg["monitor_start"]).fit(ev.norm_Z, lr_data)
    _lr8_std(dets["ccd_lr_resid"], ev.norm_Z)
    sysm = System(scorer, ev, dets, bl, cfg["monitor_start"], cfg["tau"])
    norm_items = []
    for k in ids("norm"):
        s = store.scores(f"{k}__clean", scorer); g = store.clip(f"{k}__clean")["glob"]
        norm_items.append(({kk: v[:cfg["tau"]] for kk, v in s.items() if kk != "frame_idx"}, g[:cfg["tau"]]))
    sysm.fit_joint(norm_items)
    return sysm, ref_glob


def _lr8_std(det, normZ):
    lr8 = det.lr.decision_function(normZ[:, det.ic]); det.mu_lr8, det.sd_lr8 = robust_ls(lr8, 0)


def reinstantiate(bdd: System, ref_clips, norm_scores_fn, norm_globs, cfg, scorer=None):
    """U1: same code, same learned weights; in-control population statistics from the target domain.
    ref_clips: target-domain normal-condition clips for the k-NN bank and MEWMA reference.
    norm_scores_fn(scorer) -> list of score dicts of target-domain norm sequences (normal condition)."""
    scorer = scorer or ChannelScorer(cfg["stored_layers"], k=cfg.get("knn_k", 10),
                                     glob_ref_stride=1, patch_ref_stride=cfg.get("acdc_patch_ref_stride", 1)).fit(ref_clips)
    norm_s = norm_scores_fn(scorer)
    ev = Evidence(q_top=cfg.get("q_top", 0.1)).fit(norm_s)
    dets = {}
    for name, d0 in bdd.dets.items():
        d = copy.copy(d0); d.ev = ev
        Z = ev.norm_Z
        k = d.s.kind
        if k in ("ccd_lr", "ccd_lr_resid"):
            sc = d.lr.decision_function(Z[:, d.ic])
            if k == "ccd_lr_resid":
                sc = sc - d.beta * Z[:, d.iq].mean(1)
                lr8 = d.lr.decision_function(Z[:, d.ic]); d.mu_lr8, d.sd_lr8 = robust_ls(lr8, 0)
            d.mu_r, d.sd_r = robust_ls(sc, 0)
        dets[name] = d
    bl = {}
    for name, b0 in bdd.bl.items():
        b = Baseline(name, **cfg.get("baseline_params", {}))
        b.fit({"glob": np.concatenate([c["glob"] for c in ref_clips])}, [{"glob": g} for g in norm_globs]); bl[name] = b
    return System(scorer, ev, dets, bl, bdd.ms, bdd.tau)
