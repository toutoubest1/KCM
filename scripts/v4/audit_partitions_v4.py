"""Programmatic audit of stage-4 data roles (CALIBRATION_AUDIT_V4.md, part 1). Fails loudly on any overlap."""
import json, os, collections, itertools
import pandas as pd
out = []
P = lambda *a: (print(*a), out.append(" ".join(map(str, a))))
part = json.load(open("data/v2/partitions.json")); ns = json.load(open("data/v4/bdd_norm_split.json"))
role = {k: v for k, v in part.items()}
for k in ns["normT"]: role[k] = "normT"
for k in ns["normS"]: role[k] = "normS"
groups = collections.defaultdict(set)
for k, v in role.items(): groups[v].add(k)
for a, b in itertools.combinations(sorted(groups), 2):
    assert not (groups[a] & groups[b]), (a, b)
P("BDD roles (drives):", {k: len(v) for k, v in sorted(groups.items())}, "-> pairwise disjoint: OK")
fd = "data/features_v2/dinov2_s14/clips"; files = os.listdir(fd)
corr = [f for f in files if not f.endswith("__clean.npz") and f.startswith("bdd__")]
drv = collections.Counter(role.get(f.split("__")[0] + "/" + f.split("__")[1], "?") for f in corr)
P("BDD corrupted feature clips by role of their drive:", dict(drv))
trainable = [f for f in corr if role.get(f.split("__")[0] + "/" + f.split("__")[1]) == "normT"]
P("-> corrupted clips usable for training (normT only):", len(trainable))
hold = groups["holdout"] | groups["cond_holdout"]
hf = [f for f in files if f.startswith("bdd__") and ("bdd/" + f.split("__")[1]) in hold]
assert not hf, hf[:3]
P("BDD holdout / cond_holdout drives with ANY extracted feature file:", len(hf), "(must be 0): OK")
T = pd.read_csv("data/v4/acdc_frames.csv", dtype={"rec": str}); F = json.load(open("data/v4/acdc_folds.json"))
evals = []
for k, v in F.items():
    ev = set(v["eval"]); evals += v["eval"]
    rec_roles = collections.defaultdict(set)
    for u, r in v["roles"].items():
        c, rec, ch = u.split("_"); rec_roles[f"{c}_{rec}"].add(r)
        assert f"{c}_{rec}" not in ev, (k, u)
    cal = {r for r, s in rec_roles.items() if "a_calib" in s}
    assert all(rec_roles[r] == {"a_calib"} for r in cal), "calibration recordings must be whole"
    fr = collections.Counter()
    for u, r in v["roles"].items():
        c, rec, ch = u.split("_"); fr[r] += int(((T.cond == c) & (T.rec == rec) & (T.chap == int(ch[2:]))).sum())
    n_eval = int(T[T.apply(lambda x: f"{x.cond}_{x.rec}" in ev, axis=1)].shape[0])
    assert sum(fr.values()) + n_eval == len(T)
    P(f"ACDC fold {k}: eval recordings {len(ev)} ({n_eval} pairs); roles (pairs) {dict(fr)}; "
      f"eval ∩ roles = ∅ (recording level); each chapter in exactly one role; calibration recordings whole and disjoint "
      f"from a_ref/a_train/a_norm recordings: {not any(rec_roles[r] - {'a_calib'} for r in cal)}")
assert sorted(evals) == sorted(set(evals)) and len(evals) == T.groupby(['cond', 'rec']).ngroups
P("ACDC: every one of", len(evals), "recordings is evaluated exactly once across the 4 folds: OK")
P("Directional branch training data: BDD normT corrupted (+) / normT clean (−) / ACDC a_train adverse (+) / a_train normal (−);"
  " normalisation: BDD normS, ACDC a_norm; thresholds: BDD calib, ACDC a_calib. ACDC never enters BDD normalisation/calibration.")
open("results/v4_partition_audit.txt", "w").write("\n".join(out) + "\n")
