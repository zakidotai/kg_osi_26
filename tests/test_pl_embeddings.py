"""Offline contract tests. Synthetic weights are used only inside these tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertTokenizer

from pl_embeddings import FIELDS, MaterialEmbedder, load_records, save_embeddings
from pl_embeddings.backends import Mat2VecBackend, MatSciBERTBackend, _normalized
from pl_embeddings.core import DIMENSIONS, MODELS, normalize_doi, record_texts


class StubBackend:
    def __init__(self, dimension):
        self.dimension = dimension
        self.calls = []
        self.metadata = {"checkpoint": "test-only"}

    def embed(self, texts):
        self.calls.append(list(texts))
        results = []
        for text in texts:
            vector = np.zeros(self.dimension)
            vector[0] = 1
            vector[1] = len(text)
            results.append({"vector": _normalized(vector), "status": "ok"})
        return results


@pytest.fixture
def embedder():
    result = MaterialEmbedder()
    result._backends = {name: StubBackend(DIMENSIONS[name]) for name in MODELS}
    return result


@pytest.fixture
def record():
    return {
        "dataset_id": 42, "system_id": "material-a",
        "doi_isbn": "https://doi.org/10.1016/EXAMPLE",
        "formula": " CsPbBr3 ", "compound_name": "Cesium lead bromide",
        "synthesis": "  Heated at 100 C.  ", "experimental": None,
        "PL_subsets": [{"peak": 520}],
    }


def test_order_combined_and_source_preservation(embedder, record):
    original = deepcopy(record)
    row = embedder.embed_records([record])[0]
    assert record == original
    assert row["dataset_id"] == 42
    assert row["system_id"] == "material-a"
    assert row["doi"] == "10.1016/example"
    assert [(e["model"], e["field"]) for e in row["embeddings"]] == [
        (model, field) for model in MODELS for field in FIELDS
    ]
    expected = "Formula: CsPbBr3\nCompound name: Cesium lead bromide\nSynthesis: Heated at 100 C."
    assert record_texts(record)[-1] == expected
    for backend in embedder._backends.values():
        assert backend.calls == [["CsPbBr3", "Cesium lead bromide", "Heated at 100 C.", expected]]
    for entry in row["embeddings"]:
        if entry["field"] == "experimental":
            assert entry["status"] == "missing" and entry["vector"] is None
        else:
            assert len(entry["vector"]) == DIMENSIONS[entry["model"]]
            assert np.isfinite(entry["vector"]).all()
            assert np.linalg.norm(entry["vector"]) == pytest.approx(1)


def test_batch_single_order_duplicates_and_no_peak_leakage(embedder, record):
    changed = deepcopy(record)
    changed.update(dataset_id=99, system_id="b", doi_isbn="10.5555/other")
    changed["PL_subsets"] = [{"peak": 999, "peak_text": "do not embed me"}]
    changed["structured_peak"] = 300
    rows = embedder.embed_records([record, changed, record])
    single = embedder.embed_record(**{k: record[k] for k in FIELDS[:-1]})
    assert [r["dataset_id"] for r in rows] == [42, 99, 42]
    assert all(row["embeddings"] == single for row in rows)
    assert len(embedder._backends["mat2vec"].calls[0]) == 4
    rows[0]["embeddings"][0]["vector"][0] = 100
    assert rows[2]["embeddings"] == single


def test_all_empty_does_not_load_models(tmp_path):
    embedder = MaterialEmbedder(model_dir=tmp_path)
    entries = embedder.embed_record(formula=" \n\t", synthesis=None)
    assert len(entries) == 10
    assert all(e["vector"] is None and e["status"] == "missing" for e in entries)
    assert embedder._backends == {}
    assert embedder.embed_records([]) == []


@pytest.mark.parametrize("value", [1, False, [], {}, 1.1])
@pytest.mark.parametrize("field", FIELDS[:-1])
def test_reject_non_string_fields(embedder, field, value):
    with pytest.raises(TypeError, match=field):
        embedder.embed_record(**{field: value})


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_bad_batch_size(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        MaterialEmbedder(batch_size=batch_size)


def test_missing_checkpoint_actionable_error(tmp_path):
    with pytest.raises(RuntimeError, match="python -m pl_embeddings.download"):
        MaterialEmbedder(model_dir=tmp_path).embed_record(formula="CsPbBr3")


@pytest.mark.parametrize("bad", [np.zeros(200), np.full(200, np.nan), np.ones(199)])
def test_invalid_model_vectors_rejected(embedder, bad):
    embedder._backends["mat2vec"].embed = lambda texts: [{"vector": bad, "status": "ok"} for _ in texts]
    with pytest.raises(RuntimeError):
        embedder.embed_record(formula="CsPbBr3")


def test_mat2vec_occurrence_mean_and_oov_coverage():
    from gensim.models import KeyedVectors
    backend = Mat2VecBackend.__new__(Mat2VecBackend)
    backend.vectors = KeyedVectors(200)
    a, b = np.zeros(200), np.zeros(200)
    a[0], b[1] = 2, 1
    backend.vectors.add_vectors(["a", "b"], [a, b])
    calls = []

    def process(text, **kwargs):
        calls.append(kwargs)
        return text.split(), []

    backend.processor = SimpleNamespace(process=process)
    known, unknown, empty = backend.embed(["a b b unknown", "unknown unknown", ""])
    np.testing.assert_allclose(known["vector"], _normalized(a + 2 * b))
    assert known["token_count"] == 4
    assert known["in_vocabulary_count"] == 3
    assert known["coverage"] == .75
    assert known["oov_tokens"] == ["unknown"]
    assert unknown["vector"] is None and unknown["status"] == "out_of_vocabulary"
    assert unknown["coverage"] == 0
    assert empty["vector"] is None and empty["token_count"] == 0
    assert calls == [{"exclude_punct": True, "make_phrases": True}] * 3


class TokenStateModel(torch.nn.Module):
    """Known hidden states reveal incorrect special-token/padding/chunk pooling."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=768, max_position_embeddings=512)
        self.calls = []

    def forward(self, input_ids, attention_mask, **kwargs):
        assert not self.training
        assert not self.weight.requires_grad
        assert not torch.is_grad_enabled()
        self.calls.append(input_ids.clone())
        hidden = torch.zeros((*input_ids.shape, 768), device=input_ids.device)
        hidden[..., 0] = (input_ids == 5).float() * 2
        hidden[..., 1] = (input_ids == 6).float()
        # PAD, UNK, CLS, SEP and MASK must not influence pooled content.
        hidden[..., 2] = (input_ids < 5).float() * 10000
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def bert_backend(tmp_path, monkeypatch):
    import transformers
    root = tmp_path / "matscibert"
    root.mkdir()
    (root / "model.safetensors").touch()
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n")
    # Test-only normalizer; production loads the authors' pinned implementation.
    (root / "normalize_text.py").write_text("def normalize(text):\n    return text.replace('Á', 'a')\n")
    tokenizer = BertTokenizer(vocab_file=str(root / "vocab.txt"), do_lower_case=False)
    model = TokenStateModel()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **kw: model)
    return MatSciBERTBackend(tmp_path, batch_size=16, device="cpu")


def test_bert_padding_specials_normalization_and_empty(bert_backend):
    entries = bert_backend.embed(["Á b", "a", "[CLS] a [SEP] [PAD]", "", "unrecognized"])
    expected = np.zeros(768)
    expected[:2] = [2, 1]
    np.testing.assert_allclose(entries[0]["vector"], _normalized(expected))
    np.testing.assert_allclose(entries[1]["vector"], entries[2]["vector"])
    assert [e["token_count"] for e in entries] == [2, 1, 1, 0, 0]
    assert entries[3]["status"] == "empty_after_preprocessing"
    assert entries[4]["vector"] is None
    assert all(e["vector"][2] == 0 for e in entries[:3])


def test_bert_long_text_covers_tail_and_weights_by_tokens(bert_backend):
    # Exactly two complete windows and a one-token tail. A mean of window means
    # would over-weight b by a factor of 510; truncation would remove it entirely.
    text = " ".join(["a"] * 1020 + ["b"])
    result = bert_backend.embed([text])[0]
    assert result["token_count"] == 1021
    assert result["chunk_count"] == 3
    expected = np.zeros(768)
    expected[:2] = [2040, 1]
    np.testing.assert_allclose(result["vector"], _normalized(expected))
    assert max(batch.shape[1] for batch in bert_backend.model.calls) == 512


def test_bert_single_batch_and_batch_size_invariance(bert_backend):
    texts = ["a b", "b", " ".join(["a"] * 511)]
    together = bert_backend.embed(texts)
    for text, result in zip(texts, together):
        alone = bert_backend.embed([text])[0]
        np.testing.assert_allclose(result["vector"], alone["vector"], atol=1e-7)
        assert result["token_count"] == alone["token_count"]
    bert_backend.batch_size = 1
    for a, b in zip(together, bert_backend.embed(texts)):
        np.testing.assert_allclose(a["vector"], b["vector"], atol=1e-7)


def test_lazy_backends_loaded_once(monkeypatch, tmp_path):
    import pl_embeddings.backends as backends
    loads = []

    def create(name):
        def factory(*args):
            loads.append(name)
            return StubBackend(DIMENSIONS[name])
        return factory

    monkeypatch.setattr(backends, "Mat2VecBackend", create("mat2vec"))
    monkeypatch.setattr(backends, "MatSciBERTBackend", create("matscibert"))
    embedder = MaterialEmbedder(model_dir=tmp_path)
    embedder.embed_record(formula="a")
    embedder.embed_record(formula="b")
    assert loads == list(MODELS)


def test_flat_grouped_loading_and_filters(tmp_path, record):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps([record, record]))
    grouped = tmp_path / "papers.jsonl"
    grouped.write_text(json.dumps({"doi": "10.1016/EXAMPLE", "records": [record, record]}) + "\n")
    assert load_records(flat) == [record, record]
    rows = load_records(grouped, doi_prefix="10.1016/")
    assert len(rows) == 2 and all(r["doi"] == "10.1016/example" for r in rows)
    assert load_records(flat, doi_prefix="10.5555/") == []
    grouped.write_text(json.dumps({"doi": "10.5555/other", "records": [record]}))
    with pytest.raises(ValueError, match="differs from group"):
        load_records(grouped)


def test_current_data_counts_and_grouped_equivalence():
    from scripts.group_pl_ground_truth import normalize_doi as grouping_doi
    root = Path(__file__).resolve().parents[1]
    flat = load_records(root / "data/PL_descriptions.jsonl")
    subset = load_records(root / "data/PL_descriptions.jsonl", doi_prefix="10.1016/")
    grouped = load_records(root / "data/local/pl_ground_truth_10_1016/papers.jsonl")
    assert len(flat) == 211 and len(subset) == len(grouped) == 7
    assert all(normalize_doi(r["doi_isbn"]) == grouping_doi(r["doi_isbn"]) for r in flat)
    key = lambda r: (r["dataset_id"], r["system_id"])
    assert {key(r): record_texts(r) for r in subset} == {key(r): record_texts(r) for r in grouped}


@pytest.mark.parametrize("value", [None, "isbn 123", "10.1234/a 10.1234/b"])
def test_invalid_doi(value):
    with pytest.raises(ValueError):
        normalize_doi(value)


def test_array_export_masks_alignment_and_provenance(tmp_path, embedder, record):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(record))
    second = {**record, "dataset_id": "second", "formula": None}
    results = embedder.embed_records([record, second])
    output = tmp_path / "out"
    save_embeddings(results, output, metadata=embedder.metadata(), input_path=input_path)
    for model in MODELS:
        with np.load(output / f"{model}.npz", allow_pickle=False) as arrays:
            assert arrays["embeddings"].shape == (2, 5, DIMENSIONS[model])
            assert arrays["valid"].shape == (2, 5)
            assert arrays["ids"].tolist() == ["42", "second"]
            assert arrays["fields"].tolist() == list(FIELDS)
            assert not arrays["valid"][1, 0]
            assert not arrays["valid"][:, 3].any()
            assert not arrays["embeddings"][~arrays["valid"]].any()
            np.testing.assert_allclose(np.linalg.norm(arrays["embeddings"][arrays["valid"]], axis=-1), 1)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert metadata["record_count"] == 2
    assert metadata["records"][0]["dataset_id"] == 42
    assert metadata["exports"]["mat2vec"]["status_counts"]["missing"] == 3


def test_empty_array_export(tmp_path):
    save_embeddings([], tmp_path, metadata={})
    with np.load(tmp_path / "mat2vec.npz", allow_pickle=False) as arrays:
        assert arrays["embeddings"].shape == (0, 5, 200)
