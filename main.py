# NOTE: IMPORT EXTRACTORS HERE #
#------------------------------#
from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
#------------------------------#

from functools import partial
import json
from pprint import pprint
from pathlib import Path

D = [json.loads(line) for line in Path.read_text(Path("data/PL_descriptions_with_peak.jsonl")).splitlines()]



# NOTE: DO PRECOMPUTATIONS HERE #
#-------------------------------#
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
E = [
    get_pl,
    partial(qa_syn.get_f, qid=0),
    partial(qa_syn.get_f, qid=1),
    partial(qa_exp.get_f, qid=0),
]
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
