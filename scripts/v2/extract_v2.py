"""Stage-2 feature extraction (raw-token cache). Holdout clips are extracted only with --splits holdout,
which must not be run before the method is frozen.

usage: python scripts/v2/extract_v2.py --config configs/v2/base.json --parts ref norm calib dev
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rmrm.data import load_index, load_streams  # noqa: E402
from rmrm.backbones import build_backbone  # noqa: E402
from rmrm.extract_v2 import fit_patch_pca, extract_clip_v2  # noqa: E402


def main(a):
    cfg = json.load(open(a.config))
    torch.set_num_threads(cfg["threads"]); torch.manual_seed(0)
    seqs = {s.seq_id: s for s in load_index(cfg["index"])}
    part = json.load(open(cfg["partitions"]))
    fdir = cfg["feature_root"]; os.makedirs(os.path.join(fdir, "clips"), exist_ok=True)
    size = tuple(cfg["size_hw"])
    bb = build_backbone(cfg["backbone"], cfg["stored_layers"])
    log = open(os.path.join("logs", "extract_v2.log"), "a")
    def P(*x):
        m = time.strftime("%H:%M:%S ") + " ".join(map(str, x)); print(m, flush=True); log.write(m + "\n"); log.flush()
    trp = os.path.join(fdir, "transforms_v2.npz")
    if not os.path.exists(trp):
        ref = sorted(k for k, v in part.items() if v == "ref")
        frames = [(seqs[k], i) for k in ref[: a.pca_drives] for i in range(0, min(len(seqs[k].frames), 200), 10)]
        P(f"fitting patch PCA on {len(frames)} frames from {min(len(ref), a.pca_drives)} reference drives")
        fit_patch_pca(bb, frames, size, trp, log=P)
    tr = dict(np.load(trp))
    jobs = {}
    for k, v in sorted(part.items()):
        if v in a.parts:
            jobs[f"{k}__clean"] = (k, 0, min(len(seqs[k].frames), cfg["max_clean_len"]), None, None)
    streams = [s for s in load_streams(cfg["streams"]) if s.split in a.parts]
    for s in streams:
        for g in s.segments:
            if g.corruption is None:
                if f"{g.seq_id}__clean" not in jobs:
                    if part.get(g.seq_id, "").startswith("cond_"):      # condition drives: only the frames used
                        jobs[f"{g.seq_id}__clean"] = (g.seq_id, 0, cfg["H"], None, None)
                    else:
                        n = min(len(seqs[g.seq_id].frames), cfg["max_clean_len"])
                        jobs[f"{g.seq_id}__clean"] = (g.seq_id, 0, max(n, g.end), None, None)
            else:
                jobs[g.clip_id] = (g.seq_id, g.start, g.end, g.corruption, g.severity)
    if a.norm_corrupt:
        # training data for a learned condition-evidence model: corrupted frames of NORM drives only
        jobs = {}
        for i, k in enumerate(sorted(k for k, v in part.items() if v == "norm")):
            n = len(seqs[k].frames)
            for j, c in enumerate(cfg["corruptions"]):
                sev = (i + j) % 3 + 1; st = min(100, n - 30)
                jobs[f"{k}__{c}_s{sev}"] = (k, st, st + 30, c, sev)
    P(f"{len(jobs)} clips, {sum(j[2] - j[1] for j in jobs.values())} frames")
    for i, (cid, (sid, st, en, c, sv)) in enumerate(sorted(jobs.items())):
        out = os.path.join(fdir, "clips", cid.replace("/", "__") + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        tpf = extract_clip_v2(bb, seqs[sid], st, en, size, tr, out, c, sv, batch=cfg["batch"])
        if i % 20 == 0:
            P(f"[{i+1}/{len(jobs)}] {cid} {en-st}fr {time.time()-t0:.1f}s ({tpf*1000:.0f} ms/frame)")
    P("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--parts", nargs="+", default=["ref", "norm", "calib", "dev"])
    p.add_argument("--pca_drives", type=int, default=40)
    p.add_argument("--norm_corrupt", action="store_true")
    main(p.parse_args())
