"""Record interface and array exports, independent of database-backed indexing."""

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import re

import numpy as np

from .assets import DEFAULT_MODEL_DIR, sha256_file

INPUT_FIELDS = ("formula", "compound_name", "synthesis", "experimental")
FIELDS = (*INPUT_FIELDS, "combined")
MODELS = ("mat2vec", "matscibert")
DIMENSIONS = {"mat2vec": 200, "matscibert": 768}
LABELS = ("Formula", "Compound name", "Synthesis", "Experimental")


def record_texts(record):
    """Allowlist inputs so IDs, PL_subsets, and other metadata never enter text."""
    texts = []
    for field in INPUT_FIELDS:
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{field} must be a string or None, got {type(value).__name__}")
        texts.append((value or "").strip())
    combined = "\n".join(f"{label}: {value}" for label, value in zip(LABELS, texts) if value)
    return [*texts, combined]


def normalize_doi(value):
    if not isinstance(value, str):
        raise ValueError("Each batch record needs a DOI in doi or doi_isbn")
    matches = re.findall(r"10\.\d{4,9}/[^\s,]+", value, re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(f"Expected one DOI, found {len(matches)} in {value!r}")
    return matches[0].lower()


def load_records(path, doi_prefix=None):
    """Read JSON or JSONL, accepting flat records or {doi, records: [...]} groups."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at line {number}: {error}") from error
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("Input must contain records or DOI groups")
    records = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Every input entry must be an object")
        if "records" in value:
            group_doi = normalize_doi(value.get("doi"))
            if not isinstance(value["records"], list):
                raise ValueError("A DOI group's records must be a list")
            for row in value["records"]:
                if not isinstance(row, dict):
                    raise ValueError("Every grouped record must be an object")
                row_doi = normalize_doi(row.get("doi") or row.get("doi_isbn") or group_doi)
                if row_doi != group_doi:
                    raise ValueError(f"Record DOI {row_doi} differs from group DOI {group_doi}")
                records.append({**row, "doi": group_doi})
        else:
            records.append(dict(value))
    if doi_prefix is not None:
        prefix = doi_prefix.strip().lower()
        if not prefix:
            raise ValueError("DOI prefix cannot be blank")
        records = [r for r in records if normalize_doi(r.get("doi") or r.get("doi_isbn")).startswith(prefix)]
    return records


class MaterialEmbedder:
    """Ten named, frozen embeddings per record; model loading is lazy and reused.

    Missing/OOV vectors are None. Valid vectors are lists of floats, making the
    return value JSON-serializable without losing field or model labels.
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, batch_size=16, device="cpu"):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model_dir = Path(model_dir).resolve()
        self.batch_size = batch_size
        self.device = device
        self._backends = {}

    def _backend(self, name):
        if name not in self._backends:
            from .backends import Mat2VecBackend, MatSciBERTBackend
            backend = Mat2VecBackend if name == "mat2vec" else MatSciBERTBackend
            try:
                self._backends[name] = backend(self.model_dir, self.batch_size, self.device)
            except Exception as error:
                raise RuntimeError(
                    f"Cannot load {name} from {self.model_dir}: {error}. "
                    "Install the embedding requirements and run "
                    "python -m pl_embeddings.download --model-dir " + str(self.model_dir)
                ) from error
        return self._backends[name]

    def _embed_text_rows(self, text_rows):
        # Compute repeated texts once, but never collapse records or field slots.
        texts = list(dict.fromkeys(text for row in text_rows for text in row if text))
        caches = {}
        for model in MODELS:
            outputs = self._backend(model).embed(texts) if texts else []
            if len(outputs) != len(texts):
                raise RuntimeError(f"{model} returned the wrong number of embeddings")
            caches[model] = dict(zip(texts, outputs))
        results = []
        for texts_for_row in text_rows:
            entries = []
            for model in MODELS:
                for field, text in zip(FIELDS, texts_for_row):
                    result = caches[model].get(text, {"vector": None, "status": "missing"})
                    entry = {"model": model, "field": field, **result}
                    if "oov_tokens" in entry:
                        entry["oov_tokens"] = list(entry["oov_tokens"])
                    if entry["vector"] is not None:
                        vector = np.asarray(entry["vector"], dtype=np.float32)
                        if vector.shape != (DIMENSIONS[model],) or not np.isfinite(vector).all():
                            raise RuntimeError(f"Invalid {model} vector for {field}")
                        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
                            raise RuntimeError(f"{model} vector for {field} is not normalized")
                        entry["vector"] = vector.tolist()
                    entries.append(entry)
            results.append(entries)
        return results

    def embed_record(self, formula=None, compound_name=None, synthesis=None, experimental=None):
        """Embed one material's four fields, returning ten named output entries."""
        row = dict(zip(INPUT_FIELDS, (formula, compound_name, synthesis, experimental)))
        return self._embed_text_rows([record_texts(row)])[0]

    def embed_records(self, records):
        """Embed a sequence of flat records, retaining IDs, DOI, order and duplicates."""
        rows = list(records)
        metadata = []
        texts = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"Record {index} must be an object")
            for field in ("dataset_id", "system_id"):
                if isinstance(row.get(field), bool) or not isinstance(row.get(field), (str, int)):
                    raise ValueError(f"Record {index} needs a string or integer {field}")
            metadata.append({
                "dataset_id": row["dataset_id"], "system_id": row["system_id"],
                "doi": normalize_doi(row.get("doi") or row.get("doi_isbn")),
            })
            texts.append(record_texts(row))
        return [
            {**identity, "embeddings": entries}
            for identity, entries in zip(metadata, self._embed_text_rows(texts))
        ]

    def metadata(self):
        manifest = self.model_dir / "manifest.json"
        versions = {}
        for package in ("numpy", "torch", "transformers", "tokenizers", "gensim", "ChemDataExtractor", "DAWG2", "pymatgen", "monty", "nltk"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {
            "python": platform.python_version(), "packages": versions,
            "device": self.device, "batch_size": self.batch_size,
            "models": {name: backend.metadata for name, backend in self._backends.items()},
            "assets": json.loads(manifest.read_text()) if manifest.exists() else None,
            "input_fields": list(INPUT_FIELDS), "fields": list(FIELDS),
            "combined_template": "Nonempty labeled fields joined with newline, in input_fields order",
            "normalization": "L2 after mean pooling", "frozen": True,
        }


def save_embeddings(results, output_dir, *, metadata, input_path=None):
    """Save one numeric NPZ per model, with masks and a JSON provenance manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(results)
    ids = np.asarray([str(row["dataset_id"]) for row in results], dtype=str)
    systems = np.asarray([str(row["system_id"]) for row in results], dtype=str)
    dois = np.asarray([row["doi"] for row in results], dtype=str)
    reports = {}
    for model in MODELS:
        embeddings = np.zeros((count, len(FIELDS), DIMENSIONS[model]), dtype=np.float32)
        valid = np.zeros((count, len(FIELDS)), dtype=bool)
        statuses = np.full((count, len(FIELDS)), "", dtype="U32")
        coverage = np.full((count, len(FIELDS)), np.nan, dtype=np.float32)
        for row_index, row in enumerate(results):
            entries = [entry for entry in row["embeddings"] if entry["model"] == model]
            if [entry["field"] for entry in entries] != list(FIELDS):
                raise ValueError(f"Invalid {model} field order for row {row_index}")
            for field_index, entry in enumerate(entries):
                statuses[row_index, field_index] = entry["status"]
                if entry["vector"] is not None:
                    embeddings[row_index, field_index] = entry["vector"]
                    valid[row_index, field_index] = True
                if "coverage" in entry:
                    coverage[row_index, field_index] = entry["coverage"]
        np.savez_compressed(
            output_dir / f"{model}.npz", embeddings=embeddings, valid=valid,
            ids=ids, system_ids=systems, dois=dois, fields=np.asarray(FIELDS),
            status=statuses, vocabulary_coverage=coverage,
        )
        reports[model] = {
            "shape": list(embeddings.shape),
            "status_counts": dict(Counter(statuses.ravel().tolist())),
            "vocabulary_coverage": [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in coverage
            ] if model == "mat2vec" else None,
        }
    report = {
        **metadata, "record_count": count,
        "records": [{k: r[k] for k in ("dataset_id", "system_id", "doi")} for r in results],
        "input_sha256": sha256_file(input_path) if input_path is not None else None,
        "exports": reports,
        "invalid_vectors": "Zero placeholders only; valid=False. Filter or explicitly handle before GP fitting.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
