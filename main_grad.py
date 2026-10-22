# NOTE: IMPORT EXTRACTORS HERE #
#------------------------------#
from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
from extractors.matscibert import sentence_embedding
from extractors.saved_material_embeddings import SavedMaterialEmbeddings
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

Q_syn = [
    "Name one solute:",
    "How was the temperature controlled?",
]
Q_exp = ["What type of instrument (without brand name) was used to measure peak photoluminescence?"]

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
    partial(qa_syn.get_f, qid=0),
    partial(qa_syn.get_f, qid=1),
    partial(qa_exp.get_f, qid=0),
    se_max,
    se_avg,
    saved_material.get_f,
] # documents x features
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

from gp.gp import e
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
print(grad_plus)
print(grad_minus)


