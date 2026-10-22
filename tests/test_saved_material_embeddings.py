"""Check saved-vector integration without running QA, model downloads, or GP training."""

import ast
from copy import deepcopy
import json
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest

from extractors.get_pl import get_pl
from extractors.saved_material_embeddings import SavedMaterialEmbeddings
from pl_embeddings import load_records
from pl_embeddings.assets import ROOT


@pytest.fixture
def records():
    return load_records(ROOT / "data/PL_descriptions_with_peak.jsonl")


def test_all_combined_vectors_match_current_data_and_pipeline_shape(records):
    extractor = SavedMaterialEmbeddings(records)
    with np.load(ROOT / "data/embeddings/matscibert.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(extractor.vectors, data["embeddings"][:, 4])
    # Same numeric concatenation as main.py, using the actual PL target extractor.
    features = [[get_pl(i, 0, row), *extractor.get_f(i, 1, row)] for i, row in enumerate(records)]
    assert np.asarray(features).shape == (211, 769)
    assert all(isinstance(value, float) for row in features for value in row)
    assert np.isfinite(np.asarray(features)[:, 1:]).all()
    assert not extractor.vectors.flags.writeable


def test_reordered_records_and_repeated_observations_align_by_ids(records):
    selected = [records[-1], records[0], records[-1]]
    original = SavedMaterialEmbeddings(records)
    extractor = SavedMaterialEmbeddings(selected)
    np.testing.assert_array_equal(extractor.vectors, original.vectors[[-1, 0, -1]])
    assert extractor.get_f(0, 100, selected[0]) == original.get_f(len(records)-1, 0, records[-1])
    with pytest.raises(ValueError, match="Record order changed"):
        extractor.get_f(0, 0, records[0])


def test_missing_formula_is_rejected_without_dropping_records(records):
    with pytest.raises(ValueError, match="out_of_vocabulary.*no records were dropped"):
        SavedMaterialEmbeddings(records, model="mat2vec", field="formula")
    assert SavedMaterialEmbeddings(records, model="mat2vec", field="combined").vectors.shape == (211, 200)


def test_missing_synthesis_is_rejected(records):
    with pytest.raises(ValueError, match="missing.*no records were dropped"):
        SavedMaterialEmbeddings(records, field="synthesis")


def test_changed_text_and_unknown_ids_rejected_but_peak_updates_allowed(records):
    row = deepcopy(records[0])
    row["PL_subsets"] = [{"PL_peak_position": "999 nm"}]
    assert SavedMaterialEmbeddings([row]).vectors.shape == (1, 768)
    row["synthesis"] = "New synthesis text"
    with pytest.raises(ValueError, match="text changed"):
        SavedMaterialEmbeddings([row])
    # Editing a different field does not invalidate the formula-only embedding.
    assert SavedMaterialEmbeddings([row], field="formula").vectors.shape == (1, 768)
    row["dataset_id"] = "unknown"
    with pytest.raises(ValueError, match="No saved embedding"):
        SavedMaterialEmbeddings([row])


def test_source_hash_change_rejected(records, tmp_path):
    changed_source = tmp_path / "source.json"
    changed_source.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="source hash differs"):
        SavedMaterialEmbeddings(records, source_path=changed_source)


@pytest.mark.parametrize("kwargs", [{"model": "other"}, {"field": "pl"}])
def test_unknown_model_or_field_rejected(records, kwargs):
    with pytest.raises(ValueError, match="Choose model"):
        SavedMaterialEmbeddings(records, **kwargs)


def test_main_registers_adapter_and_preserves_target_first():
    # Inspect only: importing main.py starts external models and GP training.
    tree = ast.parse((ROOT / "main.py").read_text())
    registry = next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "E" for target in node.targets)
    )
    assert ast.unparse(registry.elts[0]) == "get_pl"
    assert ast.unparse(registry.elts[-1]) == "saved_material.get_f"


def test_main_feature_loop_with_real_saved_vectors_and_stub_other_models(monkeypatch, records):
    qa = ModuleType("extractors.qa_embedder")

    class StubQAEmbedder:
        def __init__(self, key, rerun):
            pass

        def embed(self, records, questions):
            pass

        def get_f(self, i, j, d, qid):
            return [float(i), float(qid)]

    qa.QAEmbedder = StubQAEmbedder
    bert = ModuleType("extractors.matscibert")
    bert.sentence_embedding = lambda i, records, flag: np.full(3, float(i))
    gp = ModuleType("gp.gp")
    captured = {}

    def evaluate(features):
        captured["features"] = np.asarray(features)
        return 0.0

    gp.e = evaluate
    monkeypatch.setitem(sys.modules, "extractors.qa_embedder", qa)
    monkeypatch.setitem(sys.modules, "extractors.matscibert", bert)
    monkeypatch.setitem(sys.modules, "gp.gp", gp)
    monkeypatch.chdir(ROOT)
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    features = captured["features"]
    assert features.shape == (211, 1 + 3 * 2 + 2 * 3 + 768)
    np.testing.assert_allclose(features[:, 0], [get_pl(i, 0, row) for i, row in enumerate(records)], equal_nan=True)
    np.testing.assert_array_equal(features[:, -768:], SavedMaterialEmbeddings(records).vectors)
