# NOTE: IMPORT EXTRACTORS HERE #
#------------------------------#
from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
from extractors.matscibert import sentence_embedding
from extractors.saved_material_embeddings import SavedMaterialEmbeddings
#------------------------------#

from functools import partial
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


F = []

for i,d in enumerate(D):
    Fi = []
    for j,e in enumerate(E):
        s = e(i,j,d) # Extractor call
        Fi = [*Fi, *normalize(s)]
        
    F.append(Fi)

print(len(F), len(F[0]))

from gp.gp import e

pprint(e(F))
