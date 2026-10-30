"""BDD100K holdout label-audit sheets (stage 5, E1b): frames 5, 30 and 55 of the 60-frame post-change window of every
cond_holdout drive (sorted by tag, then drive id, as in data/annotations/bdd_index.json), 16 drives per sheet.
The ratings themselves are a manual, pre-results judgement stored in data/annotations/bdd_ratings.json.
NOTE: re-implementation of the original (unsaved) inline sheet generator; requires the raw BDD100K frames.
usage: python scripts/v5/label_audit_sheets_bdd.py [--out results/v5/label_audit]"""
import argparse, json, os, sys
import cv2, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT); sys.path.insert(0, "src")
from rmrm.data import load_index  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/v5/label_audit"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
seqs = {s.seq_id: s for s in load_index("data/v2/index.json")}
idx = json.load(open("data/annotations/bdd_index.json"))
rows = []
for r in idx:
    fs = seqs[r["drive"]].frames
    im = np.concatenate([cv2.resize(cv2.imread(fs[i]), (320, 180)) for i in (5, 30, 55)], 1)
    cv2.putText(im, f"#{r['idx']} {r['tag']}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); rows.append(im)
for k in range(0, len(rows), 16):
    cv2.imwrite(os.path.join(a.out, f"sheet_{k // 16:02d}.jpg"), np.concatenate(rows[k:k + 16], 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(rows), "drives")
