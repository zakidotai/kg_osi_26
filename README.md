# kg_osi_26

Hackathon project to extract material formulas, synthesis, and experimental details from papers, using PL descriptions as ground truth.

| File | Purpose |
| --- | --- |
| `data/PL_descriptions.jsonl` | Original ground truth; Git-ignored. |
| `scripts/group_pl_ground_truth.py` | Groups ground-truth records by DOI. |
| `data/local/pl_ground_truth_10_1016/papers.jsonl` | Grouped ground truth for sharing through Git. |

To regenerate the outputs, place the original export at the input path above and run from the project root:

```bash
python3 scripts/group_pl_ground_truth.py
```

The default selects `10.1016/`: 7 records across 5 papers, stored as one `{ "doi": ..., "records": [...] }` object per line. Records with the same formula stay separate because formula alone does not uniquely identify a material.
