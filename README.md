# kg_osi_26

Hackathon project to extract material formulas, synthesis, and experimental details from papers, using PL descriptions as ground truth.

| File | Purpose |
| --- | --- |
| `data/PL_descriptions.jsonl` | Full source ground truth with compound names, included in Git. |
| `scripts/group_pl_ground_truth.py` | Groups records by DOI and checks formula/name uniqueness. |
| `data/local/pl_ground_truth_10_1016/papers.jsonl` | Grouped ground truth, including compound names, for sharing through Git. |
| `data/embeddings/` | Precomputed mat2vec and MatSciBERT features for all 211 records, with IDs, validity masks and provenance. |

To regenerate the outputs from the included source, run from the project root:

```bash
python3 scripts/group_pl_ground_truth.py
```

The default selects `10.1016/`: 7 records across 5 papers, stored as one `{ "doi": ..., "records": [...] }` object per line. Records with the same formula stay separate because formula alone does not uniquely identify a material.

Use DOI + `formula` + `compound_name` to identify materials in this export, ignoring name capitalization and repeated whitespace while preserving R/S distinctions. Keep `dataset_id` and `system_id` to retain multiple records per material; extraction synonyms need explicit matching. The optional generated audit remains Git-ignored.

To use the shared embeddings, install NumPy (`pip install numpy`) and run from the project root. No model downloads are needed:

```python
import numpy as np

with np.load("data/embeddings/matscibert.npz", allow_pickle=False) as data:
    field = list(data["fields"]).index("synthesis")
    keep = data["valid"][:, field]
    X = data["embeddings"][keep, field]  # 202 records with synthesis, 768 features
    dataset_ids = data["ids"][keep]
    system_ids = data["system_ids"][keep]
```

Both files contain five fields: formula, compound name, synthesis, experimental and combined. Match rows to source data by dataset/system IDs; always apply `valid` because invalid vectors are zero placeholders. See [embedding setup and usage](docs/embeddings.md) to select other features or regenerate outputs.

**Coverage:** 217 of 2,110 vector slots are invalid, mostly due to mat2vec formula vocabulary gaps. All 211 records are retained, and both models have valid combined vectors for every record. See the [coverage table and missing-value guidance](docs/embeddings.md#coverage-and-missing-values). Experimental/combined text may reveal PL targets and needs review before prediction.
