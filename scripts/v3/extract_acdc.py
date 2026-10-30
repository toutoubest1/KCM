"""Extract raw-token features (stage-2 cache format, BDD patch-PCA basis unchanged) for every ACDC sequence."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from rmrm.data import load_index  # noqa: E402
from rmrm.backbones import build_backbone  # noqa: E402
from rmrm.extract_v2 import extract_clip_v2  # noqa: E402
cfg = json.load(open("configs/v3/acdc.json")); torch.set_num_threads(2); torch.manual_seed(0)
fdir = cfg["feature_root"]; tr = dict(np.load(os.path.join(fdir, "transforms_v2.npz")))
bb = build_backbone("dinov2_s14", cfg["stored_layers"])
seqs = load_index(cfg["index"]); t0 = time.time(); n = 0
for i, s in enumerate(seqs):
    out = os.path.join(fdir, "clips", f"{s.seq_id}__clean".replace("/", "__") + ".npz")
    if os.path.exists(out): continue
    tpf = extract_clip_v2(bb, s, 0, len(s.frames), tuple(cfg["size_hw"]), tr, out, batch=16); n += len(s.frames)
    print(f"[{i+1}/{len(seqs)}] {s.seq_id} {len(s.frames)}fr {tpf*1000:.0f} ms/frame total {time.time()-t0:.0f}s", flush=True)
print("done", n, time.time() - t0)
