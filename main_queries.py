# NOTE: IMPORT EXTRACTORS HERE #
#------------------------------#
from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
from extractors.matscibert import sentence_embedding
from extractors.saved_material_embeddings import SavedMaterialEmbeddings
from extractors.formula_features import formula_features, MeasurementConditions
#------------------------------#

from functools import partial
import numpy as np
import json
from pprint import pprint
from pathlib import Path

D = [json.loads(line) for line in Path.read_text(Path("data/PL_descriptions_with_peak.jsonl")).splitlines()]



# NOTE: DO PRECOMPUTATIONS HERE #
#-------------------------------#
# Loads the shared NPZ once; all 211 combined vectors are valid.
# Combined includes experimental prose: check PL-target leakage before interpreting scores.
saved_material = SavedMaterialEmbeddings(D, model="matscibert", field="combined")
conditions = MeasurementConditions(D) # PL temperature and excitation from PL_subsets[0]

# Kept the two strongest originals (solute, instrument); dropped temperature control (weak alone).
# New questions target what sets the PL peak in hybrid perovskites: halide, metal,
# dimensionality, organic cation, and how the sample was excited and measured.
# Answers are asked to be short and categorical so their embeddings cluster cleanly.
Q_syn = [
    "Name one solute:",
    "Which halogen elements (Cl, Br, I) appear in the precursors? Answer with the element symbols only.",
    "Which metal element (for example Pb, Sn, Ge, Bi, Cu) is in the inorganic precursor? Answer with the element symbol only.",
    "What is the organic cation or amine used? Give only its name or abbreviation.",
    "Is the product described as a 2D layered, quasi-2D, 1D, 0D, or 3D perovskite? If a layer number n is given, include it.",
    "What is the final product form: single crystal, thin film, powder, or nanocrystal?",
]
Q_exp = [
    "What type of instrument (without brand name) was used to measure peak photoluminescence?",
    "What was the excitation wavelength or excitation light source?",
    "At what temperature was the photoluminescence measured? Answer with a number and unit.",
    "What form was the measured sample in: single crystal, thin film, powder, or solution?",
]
Q_names = [
    "syn: solute", "syn: halide", "syn: metal", "syn: organic cation", "syn: dimensionality", "syn: product form",
    "exp: instrument", "exp: excitation", "exp: temperature", "exp: sample form",
]

qa_syn = QAEmbedder("synthesis", False)
qa_exp = QAEmbedder("experimental", False)

qa_syn.embed(D, Q_syn)
qa_exp.embed(D, Q_exp)
#-------------------------------#


# NOTE: REGISTER EXTRACTORS HERE #
#--------------------------------#
se_max = lambda i,j,d: sentence_embedding(i, D, flag="maxpool").tolist()
se_avg = lambda i,j,d: sentence_embedding(i, D, flag="average").tolist()

E = [
    get_pl,
    *[partial(qa_syn.get_f, qid=k) for k in range(len(Q_syn))],
    *[partial(qa_exp.get_f, qid=k) for k in range(len(Q_exp))],
    se_max,
    se_avg,
    saved_material.get_f,
    formula_features,
    conditions.get_f,
] # documents x features
names = [*Q_names, "se_max", "se_avg", "saved material", "formula", "PL conditions"] # one per feature in E[1:]
#--------------------------------#

def normalize(v):
    v = v if isinstance(v, list) else [v]
    assert all(isinstance(vi,(float,int)) for vi in v), "vector must be numeric"
    return v


# NOTE: DEFINE QUERIES HERE #
#---------------------------#
# Each query keeps only the records whose fields match its regex, then the GP and the
# ablation are run on that subset alone. Patterns carry their own case flags: (?i:...) for words,
# case-sensitive for element symbols (Pb, Sn). "all" is the unfiltered reference.
QUERIES = {
    "all": (r"", ("compound_name",)),
    "lead": (r"(?i:\blead\b)|Pb", ("compound_name", "formula")),
    "iodide": (r"(?i)iodide", ("compound_name",)),
    "lead iodide": (r"(?i)\blead\b.*iodide", ("compound_name",)),
    "bromide": (r"(?i)bromide", ("compound_name",)),
    "spin coat": (r"(?i)spin[- ]?coat", ("synthesis",)),
    "slow cooling": (r"(?i)slow(ly)? cool", ("synthesis",)),
    "laser excitation": (r"(?i)laser", ("experimental",)),
}
MIN_POINTS = 40  # below this a 5-fold GP score is mostly noise
SEEDS = (0, 1, 2)  # CV splits to average over; a single split varied e_all by up to ~15 nm
#---------------------------#

import re
from gp.gp import e # single ARD kernel on raw features; alternatives: gp.gp_blocks (additive per-block), gp.gp_pca (global PCA)
from sklearn.model_selection import KFold


def query(D, pattern, keys):
    """indices of the records where any of the given fields matches the regex"""
    rx = re.compile(pattern)
    return np.array([i for i, d in enumerate(D) if any(rx.search(str(d.get(k) or "")) for k in keys)], dtype=int)


# every extractor is evaluated once on the full D (they index into it); queries select rows
B = [np.array([normalize(ex(i, j, d)) for i, d in enumerate(D)], dtype=float) for j, ex in enumerate(E)]
blocks_all = [b.shape[1] for b in B[1:]] # B[0] is the target y


def _F(rows, feats):
    """F for the given rows: target y then the blocks of the chosen feature indices (1-based into E)"""
    return np.hstack([B[0][rows]] + [B[k][rows] for k in feats]), [blocks_all[k - 1] for k in feats]


def e_mean(rows, seed):
    """CV error of predicting the train-fold mean, on the same folds as gp.e"""
    y = B[0][rows, 0]
    y = y[~np.isnan(y)]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean([np.mean(np.abs(y[test] - np.mean(y[train]))) for train, test in cv.split(y)]))


def score(rows, feats):
    """GP CV error per seed (array over SEEDS)"""
    F, blocks = _F(rows, feats)
    return np.array([e(F, blocks, seed=s) for s in SEEDS])


results = {}
features = list(range(1, len(E)))

for qname, (pattern, keys) in QUERIES.items():
    rows = query(D, pattern, keys)
    n = int((~np.isnan(B[0][rows, 0])).sum())
    if n < MIN_POINTS:
        print(f"skipping {qname!r}: {n} records with a PL value (< {MIN_POINTS})")
        continue

    # differences are taken per seed (same folds on both sides), then averaged over seeds
    e_none = np.array([e_mean(rows, s) for s in SEEDS])
    e_all = score(rows, features)
    grad_plus = [float(np.mean(score(rows, [k]) - e_none)) for k in features]
    grad_minus = [float(np.mean(e_all - score(rows, [f for f in features if f != k]))) for k in features]
    results[qname] = dict(n=n, e_none=float(e_none.mean()), e_all=float(e_all.mean()), e_all_seeds=e_all.tolist(),
                          grad_plus=grad_plus, grad_minus=grad_minus)

    print(f"\n=== {qname} (n={n})  e_none={e_none.mean():.2f}  e_all={e_all.mean():.2f} "
          f"(seeds {', '.join(f'{v:.1f}' for v in e_all)})", flush=True)
    print(f"{'feature':<22}{'grad_plus':>12}{'grad_minus':>12}")
    for name, gp_, gm in zip(names, grad_plus, grad_minus):
        print(f"{name:<22}{gp_:>12.2f}{gm:>12.2f}")


# features x queries summaries
for key in ("grad_plus", "grad_minus"):
    print(f"\n{key} by query")
    print(f"{'feature':<22}" + "".join(f"{q[:12]:>13}" for q in results))
    for k, name in enumerate(names):
        print(f"{name:<22}" + "".join(f"{r[key][k]:>13.2f}" for r in results.values()))
print(f"{'e_none / e_all':<22}" + "".join(f"{r['e_none']:>6.1f}/{r['e_all']:<6.1f}" for r in results.values()))

out = Path(".cache/main_queries.json")
out.write_text(json.dumps(dict(names=names, queries={q: dict(pattern=p, keys=list(k)) for q, (p, k) in QUERIES.items()}, results=results), indent=2))
print(f"\nwrote {out}")
