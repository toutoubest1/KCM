"""RADIATE figures (reads stored per-stream results; the trajectory panel re-evaluates the stored frozen DINOv2 KCM with
the stored RADIATE normalisation and threshold and asserts the alarm outcomes equal the stored per-stream results)."""
import json, os, sys, pickle, copy, glob
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT); sys.path.insert(0, "scripts/v5"); sys.path.insert(0, "src")
from kcm import drive_max, MS  # noqa
OUT = os.environ.get("KCM_RADIATE_FIG_DIR", "figures/v6"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"KCM": "#1f4e79", "MEWMA_multilayer": "#c55a11", "g2_kNN_EWMA": "#548235", "deepkNN_EWMA_final": "#7f6000"}
LAB = {"KCM": "KCM (frozen)", "MEWMA_multilayer": "Multilayer MEWMA", "g2_kNN_EWMA": "Block-2 $k$-NN EWMA", "deepkNN_EWMA_final": "Deep-$k$NN EWMA (final)"}
M = list(COL)
q = pd.read_csv("results/RADIATE_PER_SEQUENCE_RESULTS.csv"); q = q[(q.alpha == 0.05) & (q.backbone == "dinov2_s14")]
S = pd.read_csv("results/RADIATE_SUMMARY_RESULTS.csv"); S = S[S.alpha == 0.05]
def save(f, n): f.savefig(f"{OUT}/{n}.pdf", bbox_inches="tight"); f.savefig(f"{OUT}/{n}.png", dpi=160, bbox_inches="tight"); plt.close(f); print("wrote", n)

# ---- per-sequence detection and delay (DINOv2)
seqs = [s for c in ["fog", "rain", "snow", "night"] for s in sorted(q[q.condition == c].sequence.unique())]
f, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True); w = 0.2
for j, m in enumerate(M):
    z = q[q.method == m].set_index("sequence")
    ax[0].bar(np.arange(len(seqs)) + (j - 1.5) * w, [z.loc[s, "detection"] for s in seqs], w, color=COL[m], label=LAB[m])
    dl = [z.loc[s, "median_delay"] for s in seqs]
    ax[1].bar(np.arange(len(seqs)) + (j - 1.5) * w, [x if np.isfinite(x) else 62 for x in dl], w, color=COL[m])
ax[0].set_ylabel("detection within 60 frames"); ax[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
ax[1].set_ylabel("median delay (frames; 62 = none)"); ax[1].axhline(60, c="grey", ls=":", lw=0.7)
ax[1].set_xticks(range(len(seqs))); ax[1].set_xticklabels(seqs, rotation=35, ha="right")
hn = S.set_index(["backbone", "method"]).loc[("dinov2_s14", "KCM"), "hn"]
ax[0].axhline(hn + 0.5, c=COL["KCM"], ls="--", lw=0.8); ax[0].text(len(seqs) - 0.5, hn + 0.52, "KCM HN + 0.50", ha="right", fontsize=6.5, color=COL["KCM"])
save(f, "radiate_per_sequence")

# ---- calibration / false alarms (KCM, DINOv2)
cal = json.load(open("results/v6/dinov2_s14/calibration.json")); h = cal["thresholds"]["KCM"]["0.05"]
d = pd.read_csv("results/v6/dinov2_s14/per_stream.csv"); d = d[(d.alpha == 0.05)]
d["fa"] = d.false_alarm.astype(str) == "True"; d["det"] = ~d.fa & np.isfinite(d.delay.astype(float))
k = d[d.method == "KCM"]
f, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
x = np.sort(cal["calib_maxima"]["KCM"]); ax[0].step(x, 1 - np.arange(1, len(x) + 1) / len(x), where="post", c="#1f4e79", label=f"C: calibration chunks (n={len(x)})")
for s, c in [("city_7_0", "#548235"), ("motorway_1_0", "#7f6000"), ("rural_1_1", "#c00000")]:
    v = k[(k.kind == "clean") & k.pre.str.startswith(s)].pre_max.values
    ax[0].plot(v, np.full(len(v), 0.012) * (1 + 0.3 * ["city_7_0", "motorway_1_0", "rural_1_1"].index(s)), "|", ms=12, c=c, label=f"T0 {s} chunks")
ax[0].axvline(h, c="k", ls="--", lw=0.8); ax[0].text(h + 0.15, 0.08, f"h = {h:.2f}", fontsize=7)
ax[0].set_yscale("log"); ax[0].set_xlabel("chunk maximum of $S_t$ over [20,100)"); ax[0].set_ylabel("survival (C)"); ax[0].legend(frameon=False, fontsize=6.0, loc="upper right", bbox_to_anchor=(1.0, 1.0)); ax[0].set_xlim(right=9)
ax[0].set_title("(a) Calibration (C) vs clean test (T0) maxima")
hh = k[(k.kind == "hard_negative") & ~k.fa]
pairs = sorted(hh["shift"].unique())
for i, p in enumerate(pairs):
    z = hh[hh["shift"] == p]; ax[1].scatter(np.full(len(z), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(z)), z.post_max, s=12,
                                            c=np.where(z.det, "#c00000", "#1f4e79"))
ax[1].axhline(h, c="k", ls="--", lw=0.8); ax[1].set_xticks(range(len(pairs))); AB = {"city_7": "c7", "motorway_1": "m1", "rural_1": "r1"}
ax[1].set_xticklabels([AB[p[3:].split("->")[0]] + "→" + AB[p[3:].split("->")[1]] for p in pairs], fontsize=7); ax[1].set_xlabel("c7 = city_7, m1 = motorway_1, r1 = rural_1 (glare)", fontsize=6.5)
ax[1].set_ylabel("max $S_t$ after the cut"); ax[1].set_title("(b) Clear→clear hard negatives (red: alarm)")
save(f, "radiate_calibration_hn")

# ---- cross-dataset comparison (KCM / MEWMA, DINOv2)
bdd = pd.read_csv("results/CONFIRMATORY_SUMMARY_RESULTS.csv", index_col=0)
acdc = pd.read_csv("results/KNOWN_CONDITION_RESULTS_V4.csv"); acdc = acdc[acdc.dataset.astype(str).str.startswith("ACDC")].set_index("method")
rows = []
for m, am, rm in [("KCM", "M2", "KCM"), ("MEWMA_multilayer", "MEWMA", "MEWMA_multilayer"), ("g2_kNN_EWMA", "g2", "g2_kNN_EWMA")]:
    b = bdd.loc[m]; a = acdc.loc[am]; r = S[(S.backbone == "dinov2_s14") & (S.method == rm)].iloc[0]
    rows += [dict(method=m, dataset="BDD100K holdout (confirmatory)", pfa=b.pfa, hn=b.hn_pooled, fog=np.nan, rain=b.real_rain, snow=b.real_snow, night=b.real_night),
             dict(method=m, dataset="ACDC (development, cross-fitted)", pfa=a.pfa, hn=a.hn, fog=a.fog, rain=a.rain, snow=a.snow, night=a.night),
             dict(method=m, dataset="RADIATE (independent external)", pfa=r.pfa, hn=r.hn, fog=r.fog_det, rain=r.rain_det, snow=r.snow_det, night=r.night_det)]
X = pd.DataFrame(rows); X.to_csv("results/v6/cross_dataset_comparison.csv", index=False)
DS = ["BDD100K holdout (confirmatory)", "ACDC (development, cross-fitted)", "RADIATE (independent external)"]
f, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for i, ds in enumerate(DS):
    for j, m in enumerate(["KCM", "MEWMA_multilayer", "g2_kNN_EWMA"]):
        r = X[(X.dataset == ds) & (X.method == m)].iloc[0]
        vals = [r.pfa, r.hn, r.fog, r.rain, r.snow, r.night]
        ax[i].bar(np.arange(6) + (j - 1) * 0.27, [0 if np.isnan(v) else v for v in vals], 0.27, color=COL[m], label=LAB[m] if i == 0 else None)
        for t, v in enumerate(vals):
            if np.isnan(v): ax[i].text(t + (j - 1) * 0.27, 0.02, "n/a", rotation=90, fontsize=5.5, ha="center")
    ax[i].set_xticks(range(6)); ax[i].set_xticklabels(["PFA", "HN", "fog", "rain", "snow", "night"], fontsize=7)
    ax[i].axhline(0.05, c="grey", ls=":", lw=0.6); ax[i].set_title(ds, fontsize=7.5)
ax[0].legend(frameon=False, fontsize=6.3, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.1)); ax[0].set_ylabel("rate")
save(f, "radiate_cross_dataset")

if not (os.path.isdir("data/features_v6/dinov2_s14") and os.path.isdir("data/raw/radiate")):
    print("SKIP radiate_examples: needs data/features_v6/dinov2_s14 and data/raw/radiate (see README)")
else:
    # ---- example frames + trajectories (DINOv2 KCM; first stream of each type in generation order)
    mdl = copy.deepcopy(pickle.load(open("results/v5/dino/kcm_model.pkl", "rb"))["KCM"]); mdl.mu, mdl.sd = cal["kcm_norm_radiate"]
    G = lambda s: np.load(f"data/features_v6/dinov2_s14/{s}.npz")["glob"].astype(np.float32)
    ex = []
    for cond in ["fog", "rain", "snow", "night"]:
        st = k[(k.kind == "condition") & (k["shift"] == cond)].iloc[0]; ex.append((cond, st))
    st = k[(k.kind == "hard_negative") & k.post.str.startswith("rural")].iloc[0]; ex.append(("clear→clear (into rural_1_1)", st))
    f, ax = plt.subplots(2, 5, figsize=(7.6, 3.0), gridspec_kw={"height_ratios": [1, 1.1]})
    for i, (t, st) in enumerate(ex):
        ps, pi = st.pre.split("#"); qs, qj = st.post.split("@"); pi, qj = int(pi), int(qj)
        g = np.concatenate([G(ps)[pi * 100:(pi + 1) * 100], G(qs)[qj * 60:(qj + 1) * 60]]); s_ = mdl.stat(g)
        ta = np.nonzero(s_ > h)[0]; ta = ta[ta >= MS]
        det = len(ta) > 0 and 100 <= ta[0] < 160; assert det == bool(st.det) and (len(ta) > 0 and ta[0] < 100) == bool(st.fa), (st.pre, st.post)
        im = cv2.cvtColor(cv2.imread(sorted(glob.glob(f"data/raw/radiate/{qs}/*.jpg"))[qj * 60 + 30]), cv2.COLOR_BGR2RGB)
        ax[0, i].imshow(im); ax[0, i].axis("off"); ax[0, i].set_title(f"{t}\n{qs}", fontsize=6.5)
        ax[1, i].plot(np.arange(MS, len(s_)), s_[MS:], c="#1f4e79", lw=1); ax[1, i].axhline(h, c="k", ls="--", lw=0.7); ax[1, i].axvline(100, c="#c55a11", lw=0.8)
        if len(ta): ax[1, i].plot(ta[0], s_[ta[0]], "v", c="r", ms=4)
        ax[1, i].set_xlabel("frame (5 Hz)", fontsize=7)
    ax[1, 0].set_ylabel("$S_t$")
    save(f, "radiate_examples")

# ---- backbones on RADIATE
if S.backbone.nunique() > 1:
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    BK = [b for b in ["dinov2_s14", "deit_s16", "clip_b16_openai"] if b in set(S.backbone)]
    for k_, (col, t) in enumerate([("pfa", "(a) Clean PFA"), ("hn", "(b) HN alarm rate"), ("pooled_det", "(c) Pooled condition detection")]):
        for j, m in enumerate(M):
            v = [S[(S.backbone == b) & (S.method == m)][col].values[0] for b in BK]
            ax[k_].bar(np.arange(len(BK)) + (j - 1.5) * 0.2, v, 0.2, color=COL[m], label=LAB[m])
        ax[k_].set_xticks(range(len(BK))); ax[k_].set_xticklabels([b.split("_")[0] for b in BK]); ax[k_].set_title(t)
    ax[0].axhline(0.05, c="k", ls=":", lw=0.7); ax[1].axhline(0.10, c="k", ls=":", lw=0.7)
    ax[0].legend(frameon=False, fontsize=6.3)
    save(f, "radiate_backbones")
