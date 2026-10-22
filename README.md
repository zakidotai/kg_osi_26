# kg_osi_26

Hackathon project to extract material formulas, synthesis, and experimental details from papers, using PL descriptions as ground truth.

| File | Purpose |
| --- | --- |
| `data/PL_descriptions.jsonl` | Full source ground truth with compound names, included in Git. |
| `scripts/group_pl_ground_truth.py` | Groups records by DOI and checks formula/name uniqueness. |
| `data/local/pl_ground_truth_10_1016/papers.jsonl` | Grouped ground truth, including compound names, for sharing through Git. |

To regenerate the outputs from the included source, run from the project root:

```bash
python3 scripts/group_pl_ground_truth.py
```

The default selects `10.1016/`: 7 records across 5 papers, stored as one `{ "doi": ..., "records": [...] }` object per line. Records with the same formula stay separate because formula alone does not uniquely identify a material.

Use DOI + `formula` + `compound_name` to identify materials in this export, ignoring name capitalization and repeated whitespace while preserving R/S distinctions. Keep `dataset_id` and `system_id` to retain multiple records per material; extraction synonyms need explicit matching. The optional generated audit remains Git-ignored.
