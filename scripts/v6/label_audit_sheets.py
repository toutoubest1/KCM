"""RADIATE pre-results label audit (RADIATE_PREREGISTRATION.md sec. 4): three frames per condition sequence
(positions n/6, n/2, 5n/6 of the 5 Hz sequence), rated V/A/N before any feature or score exists."""
import json, glob, os, cv2, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ROOT); C = ["fog_6_0", "fog_8_0", "fog_8_1", "fog_8_2", "rain_4_0", "rain_4_1", "snow_1_0", "night_1_0", "night_1_2", "night_1_4"]
rows = []
for s in C:
    fs = sorted(glob.glob(f"data/raw/radiate/{s}/*.jpg")); n = len(fs)
    ims = [cv2.imread(fs[int(n * q)]) for q in (1 / 6, 1 / 2, 5 / 6)]
    r = np.concatenate(ims, 1); cv2.putText(r, s, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2); rows.append(r)
for i in range(0, len(rows), 5):
    cv2.imwrite(f"results/v6/label_audit/sheet_{i // 5}.jpg", np.concatenate(rows[i:i + 5], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("ok")
