"""Stage-3 ACDC index, recording-level partitions and streams (deterministic; no feature is used).

Unit = (condition, GoPro recording number): chapters GOPRxxxx, GP01xxxx, GP02xxxx ... of one recording are one
unit; frames are ordered by (chapter, frame number). The ACDC train/val/test split is ignored (recordings cross it).
Each unit has an adverse sequence and its corresponding normal-condition sequence (ACDC *_ref images, same order).
Partition rule (per condition): order recordings by stable hash; fill dev until >= dev_share of the condition's frames,
the remaining recordings form the in-control pool, split at the CHAPTER level (separate GoPro video files) into
ref / calib / norm by largest remaining deficit w.r.t. pool_shares (chapters in hash order). dev is recording-disjoint
from everything; ref / calib / norm may share a recording but never a chapter.
(Earlier drafts — recording-level greedy fills — left the reference bank empty or calib < 19 sequences because the
26 recordings are few and large; replaced before any feature was computed.)
(First version filled dev/calib/norm before ref and left the reference bank EMPTY; changed before any feature.)
Streams (tau=50 normal frames, monitored from 20, H=30):
  clean      : non-overlapping 50-frame chunks of a unit's normal sequence (calib -> calibration, dev -> PFA, norm -> normalisers)
  condition  : dev unit, blocks j = 0, 20, 40, ...: normal[j, j+50) -> adverse[j+50, j+80)  (route continues, condition changes)
  hn_scene   : dev unit u, same pre-change block, post = normal frames of another dev unit (cyclic order), 30 frames
"""
import json, os, sys, glob, re, collections
sys.path.insert(0, "src")
from rmrm.data import Sequence, save_index, stable_hash, Segment, StreamSpec  # noqa: E402

cfg = json.load(open("configs/v3/acdc.json")); R = cfg["acdc_root"]
units = collections.defaultdict(list)
for p in glob.glob(f"{R}/*/*/*/*_rgb_anon.jpg"):
    c, sp, seq, f = p.split("/")[-4:]
    if sp.endswith("_ref"): continue
    chap = 0 if seq.startswith("GOPR") else int(seq[2:4])
    fr = int(f.split("_frame_")[1][:6])
    ref = f"{R}/{c}/{sp}_ref/{seq}/{f.replace('_rgb_anon.jpg', '_rgb_ref_anon.jpg')}"
    assert os.path.exists(ref), ref
    units[(c, seq[-4:])].append((chap, fr, p, ref, sp))
def add_unit(c, uid, fr, role):
    seqs.append(Sequence(f"{uid}_adv", "acdc", [x[2] for x in fr], {"condition": c, "unit": uid, "kind": "adverse"}))
    seqs.append(Sequence(f"{uid}_ref", "acdc", [x[3] for x in fr], {"condition": "normal", "unit": uid, "kind": "normal", "pair_condition": c}))
    part[uid] = role
    meta.append({"unit": uid, "condition": c, "n_frames": len(fr), "role": role, "acdc_splits": dict(collections.Counter(x[4] for x in fr))})


seqs, part, meta = [], {}, []
for c in cfg["conditions"]:
    us = sorted([u for u in units if u[0] == c], key=lambda u: stable_hash(f"acdc|{u[0]}|{u[1]}"))
    tot = sum(len(units[u]) for u in us); acc = collections.Counter()
    pool = []
    for u in us:
        n = len(units[u]); fr = sorted(units[u])
        if acc["dev"] < cfg["dev_share"] * tot:
            acc["dev"] += n; add_unit(c, f"acdc/{c}_{u[1]}", fr, "dev")
        else:
            pool.append((u, fr))
    # non-dev recordings: CHAPTER-level units (separate GoPro video files = different road segments), each assigned
    # to the in-control role with the largest remaining deficit relative to its target share of the pool
    chaps = sorted({(u, x[0]) for u, fr in pool for x in fr}, key=lambda t: stable_hash(f"acdc|{c}|{t[0][1]}|{t[1]}"))
    ptot = sum(len(fr) for _, fr in pool); pacc = collections.Counter(); tgt = cfg["pool_shares"]
    merged = collections.defaultdict(list)
    for u, ch in chaps:
        fr = [x for uu, f in pool if uu == u for x in f if x[0] == ch]
        role = max(tgt, key=lambda r: tgt[r] * ptot - pacc[r]); pacc[role] += len(fr)
        merged[(u, role)] += fr
    for (u, role), fr in sorted(merged.items()):          # chapters of one recording with the same role, in order
        add_unit(c, f"acdc/{c}_{u[1]}_{role}", sorted(fr), role)

os.makedirs("data/v3", exist_ok=True)
save_index(seqs, cfg["index"]); json.dump(part, open(cfg["partitions"], "w"), indent=1)
json.dump(meta, open("data/v3/acdc_units.json", "w"), indent=1)
L = {s.seq_id: len(s.frames) for s in seqs}
tau, H, st = cfg["tau"], cfg["H"], cfg["stream_stride"]
streams = []
dev_units = sorted([u for u, r in part.items() if r == "dev"], key=lambda u: stable_hash(u))
for u, role in sorted(part.items()):
    n = L[f"{u}_ref"]
    for k, j in enumerate(range(0, n - tau + 1, tau)):          # clean chunks (all roles; role decides use)
        streams.append(StreamSpec(f"{u}__clean{k}", role, "clean", "none", None, [Segment(f"{u}_ref", j, j + tau)], None, u))
    if role != "dev": continue
    other = dev_units[(dev_units.index(u) + 1) % len(dev_units)]
    no = L[f"{other}_ref"]
    for k, j in enumerate(range(0, n - tau - H + 1, st)):
        cnd = u.split("/")[1].split("_")[0]
        streams.append(StreamSpec(f"{u}__cond{k}", "dev", "condition", cnd, None,
                                  [Segment(f"{u}_ref", j, j + tau), Segment(f"{u}_adv", j + tau, j + tau + H)], tau, u))
        j2 = (j * 7) % max(no - H, 1)
        streams.append(StreamSpec(f"{u}__hn{k}", "dev", "hard_negative", "hn_scene", None,
                                  [Segment(f"{u}_ref", j, j + tau), Segment(f"{other}_ref", j2, j2 + H)], tau, u))
json.dump([s.to_json() for s in streams], open(cfg["streams"], "w"), indent=1)
for m in meta: print(m)
cnt = collections.Counter((s.split, s.kind, s.shift) for s in streams)
for k, v in sorted(cnt.items()): print(k, v)
