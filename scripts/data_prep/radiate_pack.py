"""RADIATE pre-processing exactly as pre-registered (RADIATE_PREREGISTRATION.md sec. 2; run log R-0):
zed_left frames in timestamp order (zed_left.txt), every 3rd frame (15 Hz -> 5 Hz), resize 672x376 -> 480x270
INTER_AREA, JPEG q90. No crop, no rectification. Frames that are listed in zed_left.txt but missing from the zip
(city_2_0) are skipped, as in the original run.

usage: python scripts/data_prep/radiate_pack.py --zip-dir /path/to/radiate_zips [--out-dir data/raw/radiate] [SEQ ...]
(default SEQ list = the 22 pre-registered sequences). Writes <out-dir>/<seq>/<00000>.jpg and <out-dir>/<seq>/meta.json.
(The original run wrote the same files into one tar per sequence on the PI's computer and extracted them unchanged.)"""
import argparse, io, json, os, zipfile
import cv2, numpy as np
PREREG = ["city_2_0", "city_5_0", "junction_3_0", "city_1_1", "city_3_0", "city_6_0", "junction_1_2", "junction_2_0", "motorway_2_1",
          "city_7_0", "motorway_1_0", "rural_1_1", "fog_6_0", "fog_8_0", "fog_8_1", "fog_8_2", "rain_4_0", "rain_4_1", "snow_1_0",
          "night_1_0", "night_1_2", "night_1_4"]
ap = argparse.ArgumentParser(); ap.add_argument("--zip-dir", required=True); ap.add_argument("--out-dir", default="data/raw/radiate")
ap.add_argument("seqs", nargs="*"); a = ap.parse_args()
for seq in a.seqs or PREREG:
    od = os.path.join(a.out_dir, seq)
    if os.path.exists(os.path.join(od, "meta.json")): print("skip", seq); continue
    os.makedirs(od, exist_ok=True)
    z = zipfile.ZipFile(os.path.join(a.zip_dir, f"{seq}.zip"))
    rows = [l.split() for l in z.read("zed_left.txt").decode().strip().splitlines()]
    fr = sorted(((float(r[3]), r[1]) for r in rows))
    names = set(z.namelist()); kept = []; shapes = set()
    for ts, f in fr[::3]:
        n = f"zed_left/{f}.png"
        if n not in names: continue
        im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR); shapes.add(im.shape)
        im = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        open(os.path.join(od, f"{len(kept):05d}.jpg"), "wb").write(buf.tobytes()); kept.append([f, ts])
    json.dump({"seq": seq, "n_raw": len(fr), "n_kept": len(kept), "frames": kept, "raw_shapes": [list(s) for s in shapes],
               "meta": json.loads(z.read("meta.json"))}, open(os.path.join(od, "meta.json"), "w"))
    print(seq, len(fr), len(kept), shapes, flush=True)
