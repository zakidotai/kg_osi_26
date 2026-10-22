"""Frozen materials-language embeddings for PL records."""

from .core import FIELDS, MaterialEmbedder, load_records, save_embeddings

__all__ = ["FIELDS", "MaterialEmbedder", "load_records", "save_embeddings"]
