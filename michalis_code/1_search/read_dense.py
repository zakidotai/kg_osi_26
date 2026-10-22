import numpy as np
d = np.load("syn_text_dense.npz")
# d = np.load("syn_facets_sim.npz")
print(d["ids"])
print(d["similarity"].round(2))