# NOTE: IMPORT EXTRACTORS HERE #
#------------------------------#
from extractors.get_pl import get_pl
from extractors.qa_embedder import QAEmbedder
#------------------------------#

from functools import partial
import json
from pprint import pprint
from pathlib import Path

D = [json.loads(line) for line in Path.read_text(Path("data/PL_descriptions.jsonl")).splitlines()]



# NOTE: DO PRECOMPUTATIONS HERE #
#-------------------------------#
Q_syn = [
    "Name one solute:",
    "How was the temperature controlled?",
]
Q_exp = ["What type of instrument (without brand name) was used to measure peak photoluminescence?"]

qa_syn = QAEmbedder("synthesis")
qa_exp = QAEmbedder("experimental")

qa_syn.embed(D, Q_syn)
qa_exp.embed(D, Q_exp)
#-------------------------------#


# NOTE: REGISTER EXTRACTORS HERE #
#--------------------------------#
E = [
    get_pl,
    qa_syn.get_f,
    qa_syn.get_f,
    qa_exp.get_f,
]
#--------------------------------#

F = []

for i,d in enumerate(D):
    Fi = []
    for j,e in enumerate(E):
        Fi.append(e(i,j,d))
    F.append(Fi)



# from gp.gp import e

# pprint(e(F))
