"""Run from the repo root after main_queries.py: python figs/make_fig_data.py
Data for the writeup figures: (1) GP error as the most important questions are added,
(2) pairwise error landscape over the top features. Seeds 0-2, same GP as main_queries.py."""
import sys, json, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from functools import partial
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
from extractors.matscibert import sentence_embedding
from extractors.saved_material_embeddings import SavedMaterialEmbeddings
from extractors.formula_features import formula_features, MeasurementConditions
from gp.gp import e

SEEDS = (0, 1, 2)
D = [json.loads(l) for l in Path("data/PL_descriptions_with_peak.jsonl").read_text().splitlines()]
ns = {}
src = Path("main_grad_agent.py").read_text()
exec(src[src.index("Q_syn = ["):src.index("qa_syn = QAEmbedder")], ns)
qa_syn, qa_exp = QAEmbedder("synthesis", False), QAEmbedder("experimental", False)
qa_syn.embed(D, ns["Q_syn"]); qa_exp.embed(D, ns["Q_exp"])
saved = SavedMaterialEmbeddings(D, model="matscibert", field="combined")
cond = MeasurementConditions(D)

def as_list(v):
    return v if isinstance(v, list) else [v]

ext = [partial(qa_syn.get_f, qid=k) for k in range(len(ns["Q_syn"]))]
ext += [partial(qa_exp.get_f, qid=k) for k in range(len(ns["Q_exp"]))]
ext += [lambda i, j, d: sentence_embedding(i, D, flag="maxpool").tolist(),
        lambda i, j, d: sentence_embedding(i, D, flag="average").tolist(),
        saved.get_f, formula_features, cond.get_f]
names = [*ns["Q_names"], "se_max", "se_avg", "saved material", "formula", "PL conditions"]
B = {n: np.array([as_list(f(i, 0, d)) for i, d in enumerate(D)], dtype=float) for n, f in zip(names, ext)}
y = np.array([[get_pl(i, 0, d)] for i, d in enumerate(D)], dtype=float)

def score(feats):
    F = np.hstack([y] + [B[f] for f in feats])
    return [e(F, seed=s) for s in SEEDS]

yy = y[~np.isnan(y[:, 0]), 0]
e_none = [float(np.mean([np.mean(np.abs(yy[te] - yy[tr].mean())) for tr, te in KFold(5, shuffle=True, random_state=s).split(yy)])) for s in SEEDS]

# order by standalone gain on all data (seed-averaged, from main_queries.py)
mq = json.load(open(".cache/main_queries.json"))
gain = dict(zip(mq["names"], mq["results"]["all"]["grad_plus"]))
questions = sorted([n for n in names if n.startswith(("syn:", "exp:"))], key=lambda n: gain[n])

out = dict(seeds=list(SEEDS), e_none=e_none, order=questions, forward={}, pairs={})
for base_name, base in (("questions only", []), ("formula + conditions + questions", ["formula", "PL conditions"])):
    curve = [dict(added=None, scores=score(base) if base else e_none)]
    for k in range(1, len(questions) + 1):
        curve.append(dict(added=questions[k - 1], scores=score(base + questions[:k])))
        print(base_name, k, np.mean(curve[-1]["scores"]), flush=True)
    out["forward"][base_name] = curve

top = questions[:6] + ["formula", "PL conditions"]
out["top"] = top
for a in top:
    out["pairs"][f"{a}|{a}"] = score([a])
for a, b in combinations(top, 2):
    out["pairs"][f"{a}|{b}"] = score([a, b])
    print("pair", a, b, np.mean(out["pairs"][f"{a}|{b}"]), flush=True)

Path("figs/fig_data.json").write_text(json.dumps(out, indent=1))
print("done")
