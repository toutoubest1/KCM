"""ACDC pre-processing (DATA_SPLIT_V3.md): every adverse image and its normal-condition counterpart from
rgb_anon_trainvaltest.zip, resized 1920x1080 -> 480x270 (INTER_AREA), JPEG q90, written to
<out-dir>/<condition>/<split>[_ref]/<recording>/<name>.jpg (same file stem, .png -> .jpg).
NOTE: the original packing script was run on the PI's computer and was not preserved; this file re-implements
the documented procedure (same resize/quality/naming as the files indexed in data/v3/acdc_index.json).
usage: python scripts/data_prep/acdc_pack.py --zip /path/to/rgb_anon_trainvaltest.zip [--out-dir data/raw/acdc]"""
import argparse, os, zipfile
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--zip", required=True); ap.add_argument("--out-dir", default="data/raw/acdc")
ap.add_argument("--match", default="", help="optional substring filter (testing)"); a = ap.parse_args()
z = zipfile.ZipFile(a.zip); n_done = 0
for n in z.namelist():
    if not n.lower().endswith(".png") or "/rgb_anon/" not in "/" + n or a.match not in n: continue
    rel = n.split("rgb_anon/", 1)[1]                      # <cond>/<split>[_ref]/<rec>/<file>.png
    if rel.split("/")[0] not in ("fog", "night", "rain", "snow"): continue
    out = os.path.join(a.out_dir, rel[:-4] + ".jpg")
    if os.path.exists(out): continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(out, cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90]); n_done += 1
print("written", n_done)
