"""FINAL_DATA_SPLIT_AUDIT (stage 5) — programmatic. Run before any confirmatory score is computed."""
import json, os, collections, itertools
import pandas as pd
L = []; P = lambda *a: (print(*a), L.append(" ".join(map(str, a))))
part = json.load(open("data/v2/partitions.json")); ns = json.load(open("data/v4/bdd_norm_split.json")); old = json.load(open("data/roles_pilot.json"))
role = dict(part)
for k in ns["normT"]: role[k] = "normT"
for k in ns["normS"]: role[k] = "normS"
G = collections.defaultdict(set)
for k, v in role.items(): G[v].add(k)
for a, b in itertools.combinations(sorted(G), 2): assert not (G[a] & G[b]), (a, b)
P("BDD v2 drive roles:", {k: len(v) for k, v in sorted(G.items())}, "pairwise disjoint: OK")
oldtest = {k for k, v in old.items() if v.startswith("test")}
assert not (oldtest & set(role)); P("closed v1 test drives (", len(oldtest), ") appear in no v2 role: OK")
olddev = {k for k, v in old.items() if not v.startswith("test")}
P("stage-1 (v1) development drives in v2 roles:", dict(collections.Counter(role[k] for k in olddev if k in role)), "-> none in holdout/cond_holdout:",
  not any(role.get(k) in ("holdout", "cond_holdout") for k in olddev))
fd = "data/features_v2/dinov2_s14/clips"; files = os.listdir(fd)
hold = G["holdout"] | G["cond_holdout"]
hf = [f for f in files if "bdd/" + f.split("__")[1] in hold] if files else []
assert not hf; P("feature files of holdout/cond_holdout drives before stage 5:", len(hf), "(must be 0): OK")
dt = pd.read_csv("data/v2/drive_table.csv").set_index("vid")
ch = [k.split("/")[1] for k in G["cond_holdout"]]
P("cond_holdout drives by BDD tag:", dict(dt.loc[ch].condition.value_counts()), "; all from BDD MOT train parts:", set(dt.loc[ch].src))
s = json.load(open("data/v2/streams.json")); h = [x for x in s if x["split"] == "holdout"]
segs = {g["seq_id"] for x in h for g in x["segments"]}
assert segs <= hold, segs - hold
P("holdout streams:", dict(collections.Counter((x["kind"]) for x in h)), "; every segment drive is holdout/cond_holdout: OK")
use = {"KCM direction training (+/-)": "BDD normT (26 drives: 260 corrupted clips + clean frames) + ALL 26 ACDC recordings (adverse + normal)",
       "KCM normalisation": "BDD normS (26 drives)", "KCM threshold": "BDD calib (66 drives)",
       "baselines reference / normalisation": "BDD ref (128, stride 2) / norm = normT+normS (52)", "baselines threshold": "BDD calib (66)",
       "k-NN / localisation reference bank": "BDD ref (128)", "per-position localisation normalisation": "BDD normS",
       "dev-only pre-freeze choices": "single-layer ablation block b* (BDD dev)",
       "CONFIRMATORY evaluation": "BDD holdout (62) + cond_holdout (129) drives only",
       "development-reused (not confirmatory)": "ACDC 26 recordings (4-fold cross-fitting as in stage 4); BDD dev (58 + 190 cond_dev)"}
for k, v in use.items(): P(f"  {k}: {v}")
open("results/v5/final_split_audit.txt", "w").write("\n".join(L) + "\n")
