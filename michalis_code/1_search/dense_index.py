"""Dense sentence embeddings for HybriD3 text fields, stored in Postgres/pgvector.

These capture meaning rather than exact words ("dissolved in DMSO" is close to
"DMSO solution"). They serve two purposes:
  - semantic search (and, combined with BM25, hybrid search: see hybrid_search.py)
  - the GP kernel: cosine similarity between record embeddings (see `export`)

Embeddings are unit length, so cosine similarity = dot product.

Usage:
    pip install fastembed "psycopg[binary]" pgvector pandas numpy
    export DATABASE_URL=postgresql://user:password@host:5432/dbname

    python dense_index.py build records.csv --id-col record_id --fields syn_text exp_text
    python dense_index.py search "crystals grown slowly from a DMSO solution" --field syn_text
    python dense_index.py export --field syn_text --out syn_text_dense.npz

To try another fastembed model, put --model before the command, e.g.
    python dense_index.py --model BAAI/bge-base-en-v1.5 build records.csv
Each model's vectors are stored separately, so models can be compared side by side.
"""

import argparse
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, small and fast on CPU

# BGE models retrieve better when a short search query carries this prefix.
# Only used for search queries, never for documents or the GP kernel.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS dense_index (
    record_id TEXT NOT NULL,
    field     TEXT NOT NULL,     -- e.g. 'syn_text' or 'exp_text'
    model     TEXT NOT NULL,     -- embedding model used
    vec       vector NOT NULL,   -- unit-length embedding
    PRIMARY KEY (record_id, field, model)
);
"""


@lru_cache
def load_model(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name)


def embed(texts: list[str], model_name: str) -> np.ndarray:
    vecs = np.array(list(load_model(model_name).embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def query_vector(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    if model_name.startswith("BAAI/bge"):
        query = BGE_QUERY_PREFIX + query
    return embed([query], model_name)[0]


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute(SCHEMA)
    conn.commit()
    register_vector(conn)
    return conn


def build(csv_path: str, id_col: str, fields: list[str], model_name: str) -> None:
    df = pd.read_csv(csv_path, dtype=str)
    with connect() as conn:
        for field in fields:
            rows = df[[id_col, field]].dropna()
            rows = rows[rows[field].str.strip() != ""]
            if rows.empty:
                print(f"{field}: no text, skipped")
                continue

            vecs = embed(rows[field].tolist(), model_name)
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM dense_index WHERE field = %s AND model = %s", (field, model_name)
                )
                cur.executemany(
                    "INSERT INTO dense_index (record_id, field, model, vec) VALUES (%s, %s, %s, %s)",
                    [(rid, field, model_name, v) for rid, v in zip(rows[id_col], vecs)],
                )
            print(f"{field}: indexed {len(rows)} records with {model_name} ({vecs.shape[1]} dims)")


def search(query: str, field: str, k: int = 5, model_name: str = DEFAULT_MODEL):
    q = query_vector(query, model_name)
    with connect() as conn:
        return conn.execute(
            """
            SELECT record_id, 1 - (vec <=> %(q)s) AS cosine
            FROM dense_index
            WHERE field = %(field)s AND model = %(model)s
            ORDER BY vec <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": q, "field": field, "model": model_name, "k": k},
        ).fetchall()


def load_embeddings(field: str, model_name: str = DEFAULT_MODEL) -> tuple[list[str], np.ndarray]:
    """All stored embeddings for one field, as (record ids, N x D matrix)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT record_id, vec FROM dense_index WHERE field = %s AND model = %s ORDER BY record_id",
            (field, model_name),
        ).fetchall()
    vecs = [r[1].to_numpy() if hasattr(r[1], "to_numpy") else r[1] for r in rows]
    return [r[0] for r in rows], np.array(vecs, dtype=np.float32)


def export(field: str, out: str, model_name: str = DEFAULT_MODEL) -> None:
    """Save ids, embeddings and the N x N cosine-similarity matrix for the GP."""
    ids, X = load_embeddings(field, model_name)
    np.savez(out, ids=np.array(ids), embeddings=X, similarity=X @ X.T)
    print(f"saved {len(ids)} records ({field}, {model_name}) to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="embed text fields from a CSV and store them")
    b.add_argument("csv")
    b.add_argument("--id-col", default="record_id")
    b.add_argument("--fields", nargs="+", default=["syn_text", "exp_text"])

    s = sub.add_parser("search", help="semantic search over one text field")
    s.add_argument("query")
    s.add_argument("--field", default="syn_text")
    s.add_argument("-k", type=int, default=5)

    e = sub.add_parser("export", help="save embeddings + similarity matrix to .npz")
    e.add_argument("--field", default="syn_text")
    e.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "build":
        build(args.csv, args.id_col, args.fields, args.model)
    elif args.command == "search":
        for rid, cos in search(args.query, args.field, args.k, args.model):
            print(f"{cos:6.3f}  {rid}")
    else:
        export(args.field, args.out, args.model)
