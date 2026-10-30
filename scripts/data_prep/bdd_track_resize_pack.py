"""Resize BDD100K MOT frames to 480x270 JPEG (q=90) and pack videos [a,b) into one tar (resumable chunks).
usage: python bdd_track_resize_pack.py images20-track-val-1.zip out_dir 0 40"""
import zipfile, tarfile, io, os, sys, time
from multiprocessing import Pool
import numpy as np, cv2
src, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
z = zipfile.ZipFile(src)
names = sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
vids = sorted(set(n.split('/')[-2] for n in names)); sel = set(vids[a:b])
todo = [n for n in names if n.split('/')[-2] in sel]
ZZ = None
def init():
    global ZZ; ZZ = zipfile.ZipFile(src)
def work(n):
    im = cv2.imdecode(np.frombuffer(ZZ.read(n), np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
    return '/'.join(n.split('/')[-2:]), cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
t0 = time.time(); fn = os.path.join(out, f'bdd_track_480_v{a:03d}-{b:03d}.tar')
with Pool(4, initializer=init) as p, tarfile.open(fn + '.partial', 'w') as tf:
    for rel, bts in p.imap(work, todo, chunksize=16):
        ti = tarfile.TarInfo(rel); ti.size = len(bts); tf.addfile(ti, io.BytesIO(bts))
os.rename(fn + '.partial', fn)
print(len(vids), 'videos;', len(todo), 'frames', f'{time.time()-t0:.0f}s')
