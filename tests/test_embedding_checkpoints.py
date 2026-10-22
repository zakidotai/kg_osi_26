"""Optional real-checkpoint checks; never download assets during a test run.

Run after setup with PL_EMBEDDING_REAL_TESTS=1 python -m pytest -q
tests/test_embedding_checkpoints.py. Default tests stay offline and lightweight.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pl_embeddings import MaterialEmbedder, load_records
from pl_embeddings.core import record_texts

pytestmark = pytest.mark.skipif(
    os.environ.get("PL_EMBEDDING_REAL_TESTS") != "1",
    reason="Opt in with PL_EMBEDDING_REAL_TESTS=1 after downloading official checkpoints",
)


@pytest.fixture(scope="module")
def embedder():
    return MaterialEmbedder()


def test_real_seven_records_batch_single_and_dimensions(embedder):
    root = Path(__file__).resolve().parents[1]
    records = load_records(root / "data/PL_descriptions.jsonl", doi_prefix="10.1016/")
    batch = embedder.embed_records(records)
    assert len(batch) == 7
    for row in batch:
        assert len(row["embeddings"]) == 10
        for entry in row["embeddings"]:
            vector = entry["vector"]
            if vector is None:
                assert entry["status"] in {"missing", "out_of_vocabulary", "empty_after_preprocessing"}
            else:
                assert len(vector) == {"mat2vec": 200, "matscibert": 768}[entry["model"]]
                assert np.isfinite(vector).all()
                assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
    one = embedder.embed_record(**{key: records[0].get(key) for key in ("formula", "compound_name", "synthesis", "experimental")})
    for alone, together in zip(one, batch[0]["embeddings"]):
        assert alone["status"] == together["status"]
        if alone["vector"] is not None:
            np.testing.assert_allclose(alone["vector"], together["vector"], atol=2e-6, rtol=2e-5)
    assert not embedder._backends["mat2vec"].vectors.vectors.flags.writeable
    bert = embedder._backends["matscibert"].model
    assert not bert.training and all(not p.requires_grad for p in bert.parameters())
    assert bert.pooler is None


def test_real_long_text_has_no_truncated_tail(embedder):
    backend = embedder._backend("matscibert")
    text = "material " * 1021
    ids = backend.tokenizer(backend.normalize(text), add_special_tokens=False)["input_ids"]
    expected_count = sum(i not in backend.tokenizer.all_special_ids for i in ids)
    result = backend.embed([text])[0]
    assert result["token_count"] == expected_count
    assert result["chunk_count"] == (len(ids) + 509) // 510
    assert result["chunk_count"] >= 3
    assert np.linalg.norm(result["vector"]) == pytest.approx(1, abs=1e-5)


def test_real_preprocessing_and_input_allowlist(embedder):
    processor = embedder._backend("mat2vec").processor
    tokens, _ = processor.process("LiCoO2 is a battery cathode material.", exclude_punct=True)
    assert tokens == ["CoLiO2", "is", "a", "battery", "cathode", "material"]
    assert embedder._backend("matscibert").normalize("Matérial") == "Material"
    a = {"formula": "CsPbBr3", "PL_subsets": [{"peak": 520}]}
    b = {**a, "PL_subsets": [{"peak": 600}], "dataset_id": "different"}
    assert record_texts(a) == record_texts(b)
