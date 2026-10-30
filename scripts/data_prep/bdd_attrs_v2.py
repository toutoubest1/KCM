# Per-video key-frame attributes + object composition from the official BDD100K 100k labels.
# usage: python scripts/data_prep/bdd_attrs_v2.py /path/to/bdd100k_labels.zip   (writes bdd100k_attrs_v2.csv in the cwd; copy to data/)
import zipfile, json, csv, collections, sys
z = zipfile.ZipFile(sys.argv[1] if len(sys.argv) > 1 else 'bdd100k_labels.zip')
CATS = ['car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor', 'traffic light', 'traffic sign', 'train']
rows = []
for n in z.namelist():
    if not n.endswith('.json'): continue
    try: d = json.loads(z.read(n))
    except Exception: continue
    a = d.get('attributes', {})
    cnt = collections.Counter()
    for fr in d.get('frames', [])[:1]:
        for o in fr.get('objects', []):
            if 'box2d' in o: cnt[o.get('category')] += 1
    rows.append([n.split('/')[-1][:-5], n.split('/')[1], a.get('weather'), a.get('scene'), a.get('timeofday')] + [cnt[c] for c in CATS])
with open('bdd100k_attrs_v2.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['name', 'split', 'weather', 'scene', 'timeofday'] + ['n_' + c.replace(' ', '_') for c in CATS]); w.writerows(rows)
print(len(rows))
