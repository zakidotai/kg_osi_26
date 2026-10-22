"""Hybrid search: BM25 and dense results combined with reciprocal rank fusion (RRF).

Each method ranks the records on its own; a record's fused score is
    sum over methods of  weight / (RRF_C + rank)
RRF uses ranks, not raw scores, so BM25 scores (unbounded) and cosine
similarities (-1..1) never have to be put on a common scale. A record found
by both methods beats one found by only one of them.

Requires both indexes to be built first (bm25_index.py build, dense_index.py build).

Usage:
    export DATABASE_URL=postgresql://user:password@host:5432/dbname
    python hybrid_search.py "slow crystallisation from DMSO at 50 C" --field syn_text -k 5
"""

import argparse
import os

import psycopg
from pgvector.psycopg import register_vector

import bm25_index
import dense_index

RRF_C = 60        # standard RRF constant; larger = flatter weighting of top ranks
CANDIDATES = 50   # top results taken from each method before fusing

SQL = """
WITH bm25 AS (
    SELECT record_id, ROW_NUMBER() OVER (ORDER BY vec <#> %(sparse)s) AS rank
    FROM bm25_index
    WHERE field = %(field)s
      AND (vec <#> %(sparse)s) < 0          -- shares at least one word with the query
    ORDER BY vec <#> %(sparse)s
    LIMIT %(n)s
),
dense AS (
    SELECT record_id, ROW_NUMBER() OVER (ORDER BY vec <=> %(dense)s) AS rank
    FROM dense_index
    WHERE field = %(field)s AND model = %(model)s
    ORDER BY vec <=> %(dense)s
    LIMIT %(n)s
)
SELECT record_id,
       COALESCE(%(w_bm25)s  / (%(c)s + bm25.rank),  0)
     + COALESCE(%(w_dense)s / (%(c)s + dense.rank), 0) AS score,
       bm25.rank  AS bm25_rank,
       dense.rank AS dense_rank
FROM bm25 FULL OUTER JOIN dense USING (record_id)
ORDER BY score DESC
LIMIT %(k)s
"""


def hybrid_search(
    query: str,
    field: str = "syn_text",
    k: int = 5,
    model_name: str = dense_index.DEFAULT_MODEL,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
):
    """Returns (record_id, fused score, BM25 rank or None, dense rank or None)."""
    params = {
        "sparse": bm25_index.query_vector(query),
        "dense": dense_index.query_vector(query, model_name),
        "field": field,
        "model": model_name,
        "n": CANDIDATES,
        "c": RRF_C,
        "w_bm25": w_bm25,
        "w_dense": w_dense,
        "k": k,
    }
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        rows = conn.execute(SQL, params).fetchall()
    return [row for row in rows if row[1] > 0]  # drop records only a 0-weighted method found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--field", default="syn_text")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--model", default=dense_index.DEFAULT_MODEL)
    parser.add_argument("--w-bm25", type=float, default=1.0, help="weight of the BM25 ranking")
    parser.add_argument("--w-dense", type=float, default=1.0, help="weight of the dense ranking")
    args = parser.parse_args()

    print(f"{'score':>7}  {'bm25':>4}  {'dense':>5}  record")
    for rid, score, r_bm25, r_dense in hybrid_search(
        args.query, args.field, args.k, args.model, args.w_bm25, args.w_dense
    ):
        print(f"{score:7.4f}  {r_bm25 or '-':>4}  {r_dense or '-':>5}  {rid}")
