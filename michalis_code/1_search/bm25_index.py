"""BM25 sparse vectors for HybriD3 text fields, stored in Postgres/pgvector.

Uses the Qdrant/bm25 model via fastembed. That model only produces the
term-frequency half of BM25 and expects the vector DB to apply IDF; pgvector
has no IDF modifier, so this script computes IDF over the corpus and bakes it
into the stored document vectors. A query is then a plain 0/1 vector over its
tokens, and query . document = BM25 score (pgvector inner product).

Usage:
    pip install fastembed "psycopg[binary]" pgvector pandas
    export DATABASE_URL=postgresql://user:password@host:5432/dbname

    python bm25_index.py build records.csv --id-col record_id --fields syn_text exp_text
    python bm25_index.py search "PbCl2 dissolved in DMSO, single crystals" --field syn_text -k 5

Re-run `build` whenever records are added: IDF depends on the whole corpus.
"""

import argparse
import math
import os
from collections import Counter

import pandas as pd
import psycopg
from fastembed import SparseTextEmbedding
from fastembed.sparse.bm25 import remove_non_alphanumeric
from pgvector import SparseVector
from pgvector.psycopg import register_vector

# pgvector's sparsevec allows at most 1e9 dimensions, but fastembed hashes
# tokens into ids up to 2^31, so ids are folded into this range.
DIM = 1_000_000_000

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS bm25_index (
    record_id TEXT NOT NULL,
    field     TEXT NOT NULL,              -- e.g. 'syn_text' or 'exp_text'
    vec       sparsevec({DIM}) NOT NULL,  -- BM25 weights (TF x IDF) per token
    PRIMARY KEY (record_id, field)
);
"""


def load_model(avg_len: float = 256.0) -> SparseTextEmbedding:
    # BM25_MODEL_PATH: optional local copy of the model files, for offline use.
    extra = {}
    if os.environ.get("BM25_MODEL_PATH"):
        extra["specific_model_path"] = os.environ["BM25_MODEL_PATH"]
    return SparseTextEmbedding("Qdrant/bm25", avg_len=avg_len, **extra)


def fold(indices, values) -> dict[int, float]:
    """Map fastembed token ids into pgvector's index range (0-based in Python)."""
    out: dict[int, float] = {}
    for i, v in zip(indices, values):
        j = int(i) % DIM
        out[j] = out.get(j, 0.0) + float(v)
    return out


def token_count(model: SparseTextEmbedding, text: str) -> int:
    """Document length as fastembed's BM25 sees it (after stopwords + stemming)."""
    bm25 = model.model
    return len(bm25._stem(bm25.tokenizer.tokenize(remove_non_alphanumeric(text))))


def bm25_vectors(texts: list[str]) -> tuple[list[dict[int, float]], float]:
    # fastembed defaults to avg_len=256 tokens; our paragraphs are much shorter,
    # so use the corpus' real average for the length normalisation.
    probe = load_model()
    avg_len = sum(token_count(probe, t) for t in texts) / len(texts) or 1.0
    model = load_model(avg_len)

    tf = [fold(e.indices, e.values) for e in model.embed(texts)]

    n = len(tf)
    doc_freq = Counter(j for doc in tf for j in doc)
    idf = {j: math.log((n - c + 0.5) / (c + 0.5) + 1) for j, c in doc_freq.items()}

    return [{j: w * idf[j] for j, w in doc.items()} for doc in tf], avg_len


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute(SCHEMA)
    conn.commit()
    register_vector(conn)
    return conn


def build(csv_path: str, id_col: str, fields: list[str]) -> None:
    df = pd.read_csv(csv_path, dtype=str)
    with connect() as conn:
        for field in fields:
            rows = df[[id_col, field]].dropna()
            rows = rows[rows[field].str.strip() != ""]
            if rows.empty:
                print(f"{field}: no text, skipped")
                continue

            vecs, avg_len = bm25_vectors(rows[field].tolist())
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("DELETE FROM bm25_index WHERE field = %s", (field,))
                cur.executemany(
                    "INSERT INTO bm25_index (record_id, field, vec) VALUES (%s, %s, %s)",
                    [(rid, field, SparseVector(v, DIM)) for rid, v in zip(rows[id_col], vecs)],
                )
            print(f"{field}: indexed {len(rows)} records (avg length {avg_len:.1f} tokens)")


def query_vector(query: str) -> SparseVector:
    """1 for each (stemmed, non-stopword) query token, folded like the documents."""
    embedding = next(iter(load_model().query_embed(query)))
    return SparseVector(fold(embedding.indices, embedding.values), DIM)


def search(query: str, field: str, k: int = 5) -> list[tuple[str, float]]:
    q = query_vector(query)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT record_id, -(vec <#> %(q)s) AS score
            FROM bm25_index
            WHERE field = %(field)s
            ORDER BY vec <#> %(q)s
            LIMIT %(k)s
            """,
            {"q": q, "field": field, "k": k},
        ).fetchall()
    return [(rid, score) for rid, score in rows if score > 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="embed text fields from a CSV and store them")
    b.add_argument("csv")
    b.add_argument("--id-col", default="record_id")
    b.add_argument("--fields", nargs="+", default=["syn_text", "exp_text"])

    s = sub.add_parser("search", help="BM25 search over one text field")
    s.add_argument("query")
    s.add_argument("--field", default="syn_text")
    s.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "build":
        build(args.csv, args.id_col, args.fields)
    else:
        for rid, score in search(args.query, args.field, args.k):
            print(f"{score:8.3f}  {rid}")
