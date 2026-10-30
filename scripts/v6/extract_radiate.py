"""RADIATE glob features (CLS and mean-patch of blocks 2..12) for one frozen encoder, all 22 pre-registered
sequences, all 5 Hz frames. DINOv2: 480x270 -> 182x322 (frozen input). CLIP/DeiT: 182x322 -> 176x320 exactly as in
stage-5 E4. Resumable. usage: extract_radiate.py dinov2_s14|clip_b16_openai|deit_s16"""
import glob, json, os, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "src")
from rmrm.backbones import build_backbone  # noqa
from rmrm.extract_v2 import run_backbone_raw  # noqa
from rmrm.data import resize, load_frame  # noqa
torch.set_num_threads(2); torch.manual_seed(0)
name = sys.argv[1]; SZ = (182, 322); IN = SZ if name == "dinov2_s14" else (176, 320)
OUT = f"data/features_v6/{name}"; os.makedirs(OUT, exist_ok=True)
bb = build_backbone(name, [2, 4, 6, 8, 10, 12])
SEQ = sorted(os.listdir("data/raw/radiate")); t0 = time.time()
for s in SEQ:
    f = f"{OUT}/{s}.npz"
    if os.path.exists(f): continue
    fs = sorted(glob.glob(f"data/raw/radiate/{s}/*.jpg")); G = []
    for j in range(0, len(fs), 16):
        ims = [resize(load_frame(p, SZ), SZ) for p in fs[j:j + 16]]
        if IN != SZ: ims = [cv2.resize(im, (IN[1], IN[0]), interpolation=cv2.INTER_AREA) for im in ims]
        _, g, _ = run_backbone_raw(bb, ims); G.append(g.astype(np.float16))
    np.savez(f + ".tmp.npz", glob=np.concatenate(G)); os.replace(f + ".tmp.npz", f)
    print(f"[{time.time()-t0:6.0f}s] {s} {len(fs)}", flush=True)
print("done", flush=True)
