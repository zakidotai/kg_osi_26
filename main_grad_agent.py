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


def _F(_E):
    """returns F (documents x features) and the width of each feature block after the target"""
    _F = []
    sizes = [None] * len(_E)

    for i,d in enumerate(D):
        Fi = []
        for j,e in enumerate(_E):
            s = normalize(e(i,j,d)) # Extractor call
            assert sizes[j] in (None, len(s)), f"extractor {j} returned {len(s)} values, expected {sizes[j]}"
            sizes[j] = len(s)
            Fi = [*Fi, *s]

        _F.append(Fi)

    return _F, sizes[1:] # sizes[0] is the target y

F, blocks = _F(E)

from gp.gp import e # single ARD kernel on raw features; alternatives: gp.gp_blocks (additive per-block), gp.gp_pca (global PCA)
from sklearn.model_selection import KFold

y = np.array(F)[:,0].flatten()
y = y[~np.isnan(y)]

cv = KFold(n_splits=5, shuffle=True, random_state=0)
e_none = float(np.mean([np.mean(np.abs(y[test] - np.mean(y[train]))) for train, test in cv.split(y)]))
e_all = e(F, blocks)

grad_plus = []
grad_minus = []

for i in range(1, len(E)):
    E_plus = [E[0], E[i]]
    F, blocks = _F(E_plus)
    grad_plus.append(e(F, blocks) - e_none)

    E_minus = [e for j,e in enumerate(E) if j!=i]
    F, blocks = _F(E_minus)
    grad_minus.append(e_all - e(F, blocks))


print(e_none, e_all)
print(f"{'feature':<22}{'grad_plus':>12}{'grad_minus':>12}")
for name, gp_, gm in zip(names, grad_plus, grad_minus):
    print(f"{name:<22}{gp_:>12.2f}{gm:>12.2f}")


