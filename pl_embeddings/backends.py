"""Inference adapters for the authors' checkpoints and preprocessing code."""

import collections
import collections.abc
from contextlib import contextmanager
from copy import copy
from functools import partial
import importlib.util
from pathlib import Path

import numpy as np

from .assets import MAT2VEC_REVISION, MATSCIBERT_ID, MATSCIBERT_REVISION


def _load_module(name, path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _legacy_collections():
    """CDE 1.3 imports ABCs moved to collections.abc; restore names after import."""
    added = []
    for name in ("MutableSequence", "MutableMapping", "Mapping", "Sequence", "Iterable", "MutableSet", "Set"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            delattr(collections, name)


def _normalized(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("The model returned a non-finite or zero-norm vector")
    return (vector / norm).astype(np.float32)


class Mat2VecBackend:
    def __init__(self, model_dir, batch_size, device):
        from gensim.models import Word2Vec

        root = Path(model_dir) / "mat2vec"
        with _legacy_collections():
            module = _load_module("_official_mat2vec_process", root / "process.py")
            from chemdataextractor.doc import Paragraph
            from chemdataextractor.nlp.tokenize import ChemSentenceTokenizer
            from chemdataextractor.nlp.lexicon import ChemLexicon
        # Same official Paragraph/tokenizers, with the chemistry Punkt model
        # addressed locally. Do not write to the user's CDE configuration.
        punkt_path = root / "punkt_chem-1.0.pickle"
        if not punkt_path.is_file():
            raise FileNotFoundError(punkt_path)
        clusters_path = root / "clusters_chem1500-1.0.pickle"
        if not clusters_path.is_file():
            raise FileNotFoundError(clusters_path)
        # ChemLexicon is a singleton by default. Give this processor its own
        # cache and model path without changing another CDE user's configuration.
        lexicon = copy(ChemLexicon())
        lexicon.lexemes, lexicon.clusters, lexicon._loaded_clusters = {}, {}, False
        lexicon.clusters_path = str(clusters_path.resolve())
        module.Paragraph = partial(
            Paragraph, sentence_tokenizer=ChemSentenceTokenizer(model=str(punkt_path.resolve())),
            lexicon=lexicon,
        )
        self.processor = module.MaterialsTextProcessor(phraser_path=str(root / "phraser.pkl"))
        self.vectors = Word2Vec.load(str(root / "pretrained_embeddings"), mmap="r").wv
        if self.vectors.vector_size != 200:
            raise ValueError(f"Expected 200-dimensional mat2vec, got {self.vectors.vector_size}")
        self.vectors.vectors.flags.writeable = False
        self.processor.process("material", exclude_punct=True, make_phrases=True)
        self.metadata = {
            "checkpoint": "materialsintelligence/mat2vec/pretrained_embeddings",
            "code_revision": MAT2VEC_REVISION, "dimension": 200,
            "preprocessing": {
                "processor": "official MaterialsTextProcessor + ChemDataExtractor 1.3.0",
                "exclude_punct": True, "make_phrases": True, "convert_num": True,
                "normalize_materials": True, "remove_accents": True, "split_oxidation": True,
                "compatibility": "Temporary collections.abc import aliases; DAWG2 supplies dawg; local Punkt/cluster paths",
            },
            "pooling": "Arithmetic mean of in-vocabulary token occurrences, then L2 normalization",
            "unknown_tokens": "Skipped, with occurrence-based coverage; all-OOV returns None",
        }

    def embed(self, texts):
        results = []
        for text in texts:
            tokens, _ = self.processor.process(text, exclude_punct=True, make_phrases=True)
            known = [token for token in tokens if token in self.vectors]
            coverage = len(known) / len(tokens) if tokens else 0.0
            result = {
                "status": "ok" if known else "out_of_vocabulary",
                "token_count": len(tokens), "in_vocabulary_count": len(known),
                "coverage": coverage,
                "oov_tokens": sorted(set(token for token in tokens if token not in self.vectors)),
                "vector": None,
            }
            if known:
                result["vector"] = _normalized(np.asarray(self.vectors[known], dtype=np.float64).mean(axis=0))
            results.append(result)
        return results


class MatSciBERTBackend:
    def __init__(self, model_dir, batch_size, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        root = Path(model_dir) / "matscibert"
        if not (root / "model.safetensors").is_file():
            raise FileNotFoundError(root / "model.safetensors")
        self.normalize = _load_module("_official_matscibert_normalize", root / "normalize_text.py").normalize
        # The slow WordPiece tokenizer supports special-token masks when adding
        # CLS/SEP to already tokenized windows; the fast variant does not.
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            str(root), local_files_only=True, use_safetensors=True, add_pooling_layer=False,
        )
        if self.model.config.hidden_size != 768:
            raise ValueError(f"Expected 768-dimensional MatSciBERT, got {self.model.config.hidden_size}")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.content_window = min(510, self.model.config.max_position_embeddings - self.tokenizer.num_special_tokens_to_add())
        self.metadata = {
            "checkpoint": MATSCIBERT_ID, "revision": MATSCIBERT_REVISION, "dimension": 768,
            "preprocessing": "Official normalize_text.py and vocab_mappings.txt, then checkpoint tokenizer",
            "pooling": "Final-layer content-token mean, excluding padding/special tokens; then L2 normalization",
            "content_window": self.content_window, "overlap": 0,
            "chunk_aggregation": "Sum all content token states and divide by total content token count",
        }

    def embed(self, texts):
        import torch

        normalized = [self.normalize(text) for text in texts]
        token_lists = self.tokenizer(normalized, add_special_tokens=False, truncation=False)["input_ids"]
        windows = []
        counts = np.zeros(len(texts), dtype=np.int64)
        sums = np.zeros((len(texts), 768), dtype=np.float64)
        chunk_counts = np.zeros(len(texts), dtype=np.int64)
        for owner, tokens in enumerate(token_lists):
            for start in range(0, len(tokens), self.content_window):
                window = self.tokenizer.prepare_for_model(
                    tokens[start:start + self.content_window], add_special_tokens=True,
                    return_special_tokens_mask=True, return_attention_mask=True, truncation=False,
                )
                windows.append((owner, window))
                chunk_counts[owner] += 1
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                chunk = windows[start:start + self.batch_size]
                batch = self.tokenizer.pad([entry for _, entry in chunk], padding=True, return_tensors="pt")
                specials = batch.pop("special_tokens_mask").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                content_mask = batch["attention_mask"].bool() & ~specials.bool()
                for token_id in self.tokenizer.all_special_ids:
                    content_mask &= batch["input_ids"] != token_id
                chunk_sums = (hidden * content_mask.unsqueeze(-1)).sum(dim=1, dtype=torch.float64).cpu().numpy()
                chunk_sizes = content_mask.sum(dim=1).cpu().numpy()
                for row, (owner, _) in enumerate(chunk):
                    sums[owner] += chunk_sums[row]
                    counts[owner] += chunk_sizes[row]
        return [
            {
                "vector": _normalized(sums[i] / counts[i]) if counts[i] else None,
                "status": "ok" if counts[i] else "empty_after_preprocessing",
                "token_count": int(counts[i]), "chunk_count": int(chunk_counts[i]),
            }
            for i in range(len(texts))
        ]
