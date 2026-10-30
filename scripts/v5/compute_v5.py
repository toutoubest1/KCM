"""Stage-5 E11 compute benchmark (2 CPU threads, batch 16, run with no other job active)."""
import json, os, pickle, sys, time, resource
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/v5"); sys.path.insert(0, "scripts/v2")
from rmrm.backbones import build_backbone  # noqa
from rmrm.baselines import Baseline  # noqa
torch.set_num_threads(2); out = {}
for name, hw in [("dinov2_s14", (182, 322)), ("clip_b16_openai", (176, 320)), ("deit_s16", (176, 320))]:
    bb = build_backbone(name, [2, 4, 6, 8, 10, 12]); x = torch.rand(16, 3, *hw); bb.forward_layers(x[:2])
    ts = []
    for _ in range(3):
        t = time.perf_counter(); bb.forward_layers(x); ts.append((time.perf_counter() - t) / 16)
    out[name] = {"encoder_ms_per_frame": 1000 * float(np.median(ts)), "params_M": sum(p.numel() for p in bb.parameters()) / 1e6}
M = pickle.load(open("results/v5/dino/kcm_model.pkl", "rb"))["KCM"]
g = np.random.randn(160, 6, 2, 384).astype(np.float32)
t = time.perf_counter()
for _ in range(20): M.stat(g)
out["KCM_update_ms_per_frame"] = 1000 * (time.perf_counter() - t) / (20 * 160)
out["KCM_model_params"] = int(M.lr.coef_.size + 2 * M.sc.mean_.size + 1)
cal = json.load(open("results/v5/dino/calibration.json"))
from run_v2 import Store  # noqa
cfg = json.load(open("configs/v2/base.json")); part = json.load(open(cfg["partitions"])); store = Store(cfg["feature_root"])
ref = {"glob": np.concatenate([store.clip(f"{k}__clean")["glob"][::2] for k in sorted(k for k, v in part.items() if v == "ref")])}
norm = [{"glob": store.clip(f"{k}__clean")["glob"]} for k in sorted(k for k, v in part.items() if v == "norm")]
for n in ["mewma_multilayer", "knn_ewma", "mmd", "cusum"]:
    t = time.perf_counter(); b = Baseline(n).fit(ref, norm); fit = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(5): b.score_stream({"glob": g})
    out[f"{n}_update_ms_per_frame"] = 1000 * (time.perf_counter() - t) / (5 * 160); out[f"{n}_fit_s"] = fit
out["kNN_bank_frames"] = int(len(ref["glob"]))
out["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(out, open("results/v5/compute.json", "w"), indent=1); print(json.dumps(out, indent=1))
