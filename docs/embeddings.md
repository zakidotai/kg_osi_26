# PL record embeddings

`pl_embeddings` turns formula, compound name, synthesis and experimental prose
into separate mat2vec (200 values) and MatSciBERT (768 values) features. It does
not train a predictor or change the source data or database-backed embedder.

## Use the shared outputs without model setup

The full 211-record snapshot is committed in `data/embeddings/`: `mat2vec.npz`,
`matscibert.npz` and `metadata.json` (about 3 MB total). Reading it only requires
NumPy: `pip install numpy`. You do not need PyTorch, either checkpoint, or the
embedding environment unless you want to generate new vectors.

The quick example in the root README selects synthesis vectors and their IDs.
Choose any field listed in `fields`, and choose either model file. Match `ids`
and `system_ids` to the source records' `dataset_id` and `system_id`, converting
IDs to strings. Apply the same validity mask to features and IDs before joining
to targets. The arrays contain features, not PL target values or predictions.

The published snapshot stays separate from Git-ignored experimental runs under
`data/local/`. To deliberately regenerate the shared snapshot after model setup,
run `python -m pl_embeddings --output-dir data/embeddings` and review all three
changed files together. `metadata.json` records the source hash, checkpoint
identities, preprocessing and coverage.

## Example: one material

For **CsPbBr3 / Cesium lead bromide**, dataset `1649`, system `326`, the saved
vectors begin as follows (first three coordinates, rounded; all ten statuses
are `ok`):

| Field | mat2vec preview (200 coordinates total) | MatSciBERT preview (768 coordinates total) |
| --- | --- | --- |
| formula | `[0.091681, 0.075739, 0.025084, ...]` | `[0.025533, -0.007679, -0.000334, ...]` |
| compound_name | `[0.032546, 0.186298, -0.091086, ...]` | `[0.021041, 0.000468, -0.017967, ...]` |
| synthesis | `[0.039607, 0.105612, -0.009525, ...]` | `[0.047233, -0.019035, 0.018059, ...]` |
| experimental | `[0.083413, 0.093261, 0.000088, ...]` | `[0.010275, -0.010143, 0.020875, ...]` |
| combined | `[0.061605, 0.113159, -0.011082, ...]` | `[0.047970, -0.017597, 0.016475, ...]` |

These numbers are learned features, not predicted PL peaks or physical
measurements. To read this record as the same ordered list of ten named outputs
returned by `MaterialEmbedder`, without loading the models:

```python
import numpy as np

example = []
for model in ("mat2vec", "matscibert"):
    with np.load(f"data/embeddings/{model}.npz", allow_pickle=False) as data:
        matches = np.flatnonzero((data["ids"] == "1649") & (data["system_ids"] == "326"))
        assert len(matches) == 1
        row = matches[0]
        for col, field in enumerate(data["fields"]):
            example.append({
                "model": model,
                "field": str(field),
                "vector": data["embeddings"][row, col].tolist() if data["valid"][row, col] else None,
                "status": str(data["status"][row, col]),
            })
```

This gives the full vectors; the table above is only a readable preview.

## Setup

Run from the repository root with Python 3.11 in an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embeddings.txt
python -m pip install --no-deps ChemDataExtractor==1.3.0
python -m pl_embeddings.download
```

If using `uv`, the equivalent is `uv venv --python 3.11 .venv`, followed by
`uv pip install --python .venv/bin/python -r requirements-embeddings.txt` and
`uv pip install --python .venv/bin/python --no-deps ChemDataExtractor==1.3.0`.

The one-time model download is approximately 1.4 GB. We use the authors'
[mat2vec weights and processor](https://github.com/materialsintelligence/mat2vec),
[MatSciBERT normalization](https://github.com/M3RG-IITD/MatSciBERT) and
[m3rg-iitd/matscibert checkpoint](https://huggingface.co/m3rg-iitd/matscibert).
Pinned revisions and downloaded file hashes are recorded in the local manifest.
Inference uses local files and never downloads models implicitly.

mat2vec's original preprocessing needs older dependencies. The adapter keeps the
official processor and chemistry tokenizer, temporarily restores Python's moved
`collections.abc` imports during loading, and uses DAWG2's compatible `dawg`
module. Its tokenizer and word-cluster data also stay in the local cache.
Install ChemDataExtractor separately as shown to avoid its obsolete DAWG
build dependency. Missing checkpoints or incompatible packages produce setup
errors rather than substitute embeddings.

## Python interface

```python
from pl_embeddings import MaterialEmbedder, load_records

embedder = MaterialEmbedder(batch_size=16, device="cpu")
outputs = embedder.embed_record(
    formula="CsPbBr3",
    compound_name="Cesium lead bromide",
    synthesis="The precursors were heated and stirred.",
    experimental="Photoluminescence spectra were measured at room temperature.",
)
# outputs[0]: {"model": "mat2vec", "field": "formula",
#              "vector": [...200 floats...], "status": "ok", ...}

records = load_records("data/PL_descriptions.jsonl")
results = embedder.embed_records(records)
# Each result retains dataset_id, system_id, normalized doi, and embeddings.
```

Each output list has 10 entries: `formula`, `compound_name`, `synthesis`,
`experimental`, `combined` for mat2vec, then the same order for MatSciBERT.
The two vector spaces remain separate. Models load once per instance and are
reused. Batch inference retains source order and duplicate observations.

Combined text joins labeled, nonempty fields with newlines, in that order:

```text
Formula: CsPbBr3
Compound name: Cesium lead bromide
Synthesis: The precursors were heated and stirred.
Experimental: Photoluminescence spectra were measured at room temperature.
```

Whitespace is trimmed for embedding only; source records are never edited.
Fields accept strings or `None`. Blank fields return `vector=None` and
`status="missing"`; other input types raise an error.

- **mat2vec:** official materials normalization and phrase detection, followed
  by the mean of known token vectors and L2 normalization. Coverage counts token
  occurrences, including repetitions. Unknown tokens are skipped; an entirely
  unknown input returns `out_of_vocabulary`. Entries include token counts,
  coverage and unknown tokens.
- **MatSciBERT:** official text normalization and checkpoint WordPiece tokenizer,
  final hidden layer, with padding and special tokens excluded. Inputs are split
  into nonoverlapping windows of at most 510 content tokens. All token sums and
  counts are combined before averaging and L2 normalization. Entries include
  token and window counts. No usable tokens returns `empty_after_preprocessing`.

Both models are frozen; BERT runs in evaluation mode with dropout disabled.
Model preprocessing is separate from DOI/formula/name identification rules.

## Generate arrays

```bash
# Seven records whose DOI starts with 10.1016/
python -m pl_embeddings --doi-prefix 10.1016/ \
  --output-dir data/local/pl_embeddings_10_1016

# All 211 records from the tracked export (the default)
python -m pl_embeddings

# The existing grouped export is also supported
python -m pl_embeddings \
  --input data/local/pl_ground_truth_10_1016/papers.jsonl \
  --output-dir data/local/pl_embeddings_grouped
```

Optional flags: `--input`, `--doi-prefix`, `--output-dir`, `--model-dir`,
`--batch-size` (16 by default), `--device` (`cpu` by default).
Inputs can be JSON arrays or JSONL, either flat records or `{doi, records}` groups.
Every batch record must have `dataset_id`, `system_id`, and a DOI in `doi` or
`doi_isbn` (or inherited from its group).

Each output directory contains:

| File | Contents |
| --- | --- |
| `mat2vec.npz` | `embeddings` shaped `(records, 5, 200)` |
| `matscibert.npz` | `embeddings` shaped `(records, 5, 768)` |
| `metadata.json` | Input hash, aligned IDs, checkpoint identities/hashes, package versions, preprocessing, pooling and vocabulary coverage |

Both NPZ files also contain `valid`, `status`, `ids` (dataset IDs as strings),
`system_ids`, `dois`, `fields`, and `vocabulary_coverage` (NaN when not applicable).
Invalid embeddings are zero placeholders paired with `valid=False`. Always use
the mask; zeros are not real embeddings:

```python
import numpy as np

with np.load("data/embeddings/matscibert.npz", allow_pickle=False) as data:
    column = list(data["fields"]).index("synthesis")
    keep = data["valid"][:, column]
    X = data["embeddings"][keep, column]
    dataset_ids = data["ids"][keep]
    system_ids = data["system_ids"][keep]
```

Checkpoints live in `data/local/embedding_models/`. New runs default to
`data/local/pl_embeddings*/`. These directories and `.venv` are Git-ignored;
the reviewed full snapshot in `data/embeddings/` is tracked.

## Connection to the shared pipeline

Keep model inference in `pl_embeddings/`. The shared `main.py` scaffold collects
features through `extractors/`; a future adapter there can load the saved NPZ
files once, align rows by dataset/system IDs, and return each selected model/field
vector together with its validity. Never assume row order alone establishes a match.
The adapter is not implemented yet; the QA extractor interface is still unfinished.

The QA scaffold embeds answers to questions with Qwen; this package embeds the
four source fields directly. They provide different features. Keep their Python
environments separate: the shared `requirements.txt` requests `transformers>=5.20`,
whereas this package was tested with `4.46.3`. Reading the saved arrays only needs
NumPy and does not require loading either of our models in the QA environment.

The peak export added in commit `e60442e` has the same 211 record IDs, order and
embedding inputs as `data/PL_descriptions.jsonl`, so the generated vectors still
align. Choosing and validating peak targets remains a separate step.

## Validation and later prediction

Run offline contract tests with `python -m pytest -q tests/test_pl_embeddings.py`.
They cover field ordering, missing/OOV values, forbidden input leakage, duplicate
records, padding/special-token exclusion, full long-text coverage, token-weighted
pooling, unit norms, single/batch consistency and saved masks/IDs.

After downloading the official checkpoints, run real-model validation with
`PL_EMBEDDING_REAL_TESTS=1 python -m pytest -q tests/test_embedding_checkpoints.py`.
This checks the seven-record subset, single/batch agreement, frozen weights,
official preprocessing, and a long input spanning at least three windows.

Validated on Python 3.11.4 / macOS arm64: 43 offline tests and 3 real-checkpoint
tests passed. Both the seven-record and full 211-record exports were generated.

### Coverage and missing values

The shared 211-record snapshot contains **2,110 vector slots: 1,893 valid and
217 invalid**. These are counts of vectors, not dropped records. All 211 records
are retained, each with five slots per model.

| Field | mat2vec invalid slots | MatSciBERT invalid slots |
| --- | ---: | ---: |
| formula | 172 | 0 |
| compound_name | 1 | 0 |
| synthesis | 9 | 9 |
| experimental | 13 | 13 |
| combined | 0 | 0 |
| **Total invalid** | **195 / 1,055** | **22 / 1,055** |

The 173 mat2vec formula/name failures have `status="out_of_vocabulary"`:
none of their processed tokens occur in the checkpoint vocabulary. mat2vec
recognizes 39/211 formulas alone. The missing synthesis and experimental slots
have `status="missing"` in both models because the source text is empty.

Both models have valid combined vectors for all 211 records. For any selected
field, filter with `valid` and apply the same filter to IDs and targets. Invalid
slots contain zeros only to preserve array shape; never feed those zeros to a
model as genuine embeddings. Use `status` to distinguish missing source text
from vocabulary gaps. Counts can be checked with `(~data["valid"]).sum(axis=0)`
in the saved `fields` order.

A valid vector does not guarantee that material identity is preserved. In the
seven-record subset, mat2vec skips the distinguishing R/S name fragments for
datasets 1857 and 1853, producing identical compound-name and combined vectors.
Keep the source formula/name and IDs for identity; treat embeddings as features
to evaluate, not unique material identifiers. MatSciBERT's complete formula
coverage alone does not establish chemical accuracy or predictive performance.

Only the four requested fields enter the models: IDs, DOI, `PL_subsets`, and
structured peak values are excluded. Experimental prose may itself state a peak;
review that leakage before GP evaluation. mat2vec also normalizes standalone
numbers, so these vectors are not a replacement for explicit numeric process
features. GP training and adoption/correction of the newer peak export are
separate steps.
