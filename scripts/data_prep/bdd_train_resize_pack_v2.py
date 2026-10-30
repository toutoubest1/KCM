# Resize+pack only needed BDD MOT videos (clean daytime + condition drives; night capped per part).
# usage: python bdd_train_resize_pack_v2.py images20-track-train-K.zip out_dir A B   (run with bdd100k_attrs_v2.csv in the cwd;
#        then extract the tars into data/raw/bdd_track_train/, giving data/raw/bdd_track_train/<video>/<frame>.jpg)
import zipfile, tarfile, io, os, sys, time, csv
from multiprocessing import Pool
import numpy as np, cv2
src, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
NIGHT_CAP = 40
os.makedirs(out, exist_ok=True)
att = {r['name']: r for r in csv.DictReader(open('bdd100k_attrs_v2.csv'))}
z = zipfile.ZipFile(src)
names = sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
vids = sorted(set(n.split('/')[-2] for n in names))
keep, nn = [], 0
for v in vids:
    r = att.get(v)
    if r is None: continue
    clean = r['timeofday'] == 'daytime' and r['weather'] in ('clear', 'partly cloudy', 'overcast')
    night = r['timeofday'] == 'night'
    cond = r['timeofday'] == 'dawn/dusk' or (r['timeofday'] == 'daytime' and r['weather'] in ('rainy', 'snowy')) or r['weather'] == 'foggy'
    if clean or cond: keep.append(v)
    elif night and nn < NIGHT_CAP: keep.append(v); nn += 1
sel = set(keep[a:b])
todo = [n for n in names if n.split('/')[-2] in sel]
ZZ = None
def init():
    global ZZ; ZZ = zipfile.ZipFile(src)
def work(n):
    im = cv2.imdecode(np.frombuffer(ZZ.read(n), np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
    return '/'.join(n.split('/')[-2:]), cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
tag = os.path.basename(src).replace(' ', '').replace('.zip', '')
fn = os.path.join(out, f'{tag}_v{a:03d}-{b:03d}.tar'); t0 = time.time()
if todo:
    with Pool(4, initializer=init) as p, tarfile.open(fn + '.partial', 'w') as tf:
        for rel, bts in p.imap(work, todo, chunksize=16):
            ti = tarfile.TarInfo(rel); ti.size = len(bts); tf.addfile(ti, io.BytesIO(bts))
    os.rename(fn + '.partial', fn)
print(len(vids), 'videos;', len(keep), 'kept;', len(sel), 'in chunk;', len(todo), 'frames', f'{time.time()-t0:.0f}s')
