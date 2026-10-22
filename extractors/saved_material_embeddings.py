"""Expose saved PL vectors through main.py's extractor(i, j, record) interface."""

import json
from pathlib import Path

import numpy as np

from pl_embeddings.assets import ROOT, sha256_file
from pl_embeddings.core import DIMENSIONS, FIELDS, load_records, record_texts


def _record_key(record):
    return str(record["dataset_id"]), str(record["system_id"])


class SavedMaterialEmbeddings:
    """Load once, align to records, and reject unavailable or stale features.

    This adapter does not load a model, drop records, or impute missing vectors.
    Only NumPy and the standard library are needed at runtime.
    """

    def __init__(
        self, records, model="matscibert", field="combined", *,
        output_dir=ROOT / "data/embeddings",
        source_path=ROOT / "data/PL_descriptions.jsonl",
    ):
        if model not in DIMENSIONS or field not in FIELDS:
            raise ValueError(f"Choose model from {tuple(DIMENSIONS)} and field from {FIELDS}")
        self.model, self.field = model, field
        output_dir = Path(output_dir)
        source_path = Path(source_path)
        metadata = json.loads((output_dir / "metadata.json").read_text())
        if sha256_file(source_path) != metadata.get("input_sha256"):
            raise ValueError("Saved embedding source hash differs; regenerate the outputs before using them")
        source = {_record_key(row): row for row in load_records(source_path)}
        records = list(records)
        self._keys = [_record_key(row) for row in records]

        with np.load(output_dir / f"{model}.npz", allow_pickle=False) as data:
            fields = data["fields"].tolist()
            if fields != list(FIELDS):
                raise ValueError("Saved embedding fields do not match the expected order")
            column = fields.index(field)
            keys = list(zip(data["ids"].tolist(), data["system_ids"].tolist()))
            if len(set(keys)) != len(keys):
                raise ValueError("Saved dataset/system IDs are ambiguous; cannot align features")
            lookup = {key: index for index, key in enumerate(keys)}
            if data["embeddings"].shape != (len(keys), len(FIELDS), DIMENSIONS[model]):
                raise ValueError("Unexpected saved embedding shape")

            indices = []
            for row, key in zip(records, self._keys):
                if key not in lookup or key not in source:
                    raise ValueError(f"No saved embedding for dataset/system {key}; regenerate outputs")
                # The source hash identifies the export used for inference.
                # Compare only the selected input, allowing new PL target metadata.
                if record_texts(row)[column] != record_texts(source[key])[column]:
                    raise ValueError(f"{field} text changed for dataset/system {key}; regenerate outputs")
                index = lookup[key]
                if not data["valid"][index, column]:
                    status = str(data["status"][index, column])
                    raise ValueError(
                        f"{model}/{field} unavailable for dataset/system {key}: {status}. "
                        "Select an available feature or explicitly handle missing values; no records were dropped."
                    )
                indices.append(index)
            self.vectors = data["embeddings"][indices, column].copy()

        if not np.isfinite(self.vectors).all():
            raise ValueError("Saved embeddings contain non-finite values")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Saved embeddings must have unit length")
        self.vectors.flags.writeable = False

    def get_f(self, i, j, d):
        """Return a numeric list for main.py; j is its global extractor index."""
        if i < 0 or i >= len(self._keys) or _record_key(d) != self._keys[i]:
            raise ValueError("Record order changed after loading saved embeddings; initialize the adapter again")
        return self.vectors[i].tolist()
