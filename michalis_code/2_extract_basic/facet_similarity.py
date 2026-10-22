"""Per-facet similarity matrices for the GP kernel, built from extracted synthesis facets.

For each facet q, similarity[i, j, q] says how alike records i and j are on that facet:
    precursors, solvents, additives    Jaccard overlap = shared items / all distinct items
                                       e.g. {DMF, DMSO} vs {DMF, DMSO, chlorobenzene} = 2/3
    method, product_form, atmosphere   1 if the same category, else 0

"unknown"/"other" categories and empty precursor/solvent lists (= nothing was extracted)
match only themselves: no evidence of similarity, but the diagonal stays 1.
An empty additives list means "no additives", so two such records do match.
Every slice is a valid (positive semi-definite) kernel, so the GP can combine them
with learned weights. Temperatures and durations are left out for now.

Output .npz (same idea as dense_index.py export):
    ids         (N,)       record ids
    facets      (Q,)       facet names, in slice order
    similarity  (N, N, Q)  one N x N matrix per facet

Usage:
    python facet_similarity.py --jsonl facets.jsonl --out syn_facets_sim.npz
    python facet_similarity.py --out syn_facets_sim.npz            # reads synthesis_facets via DATABASE_URL
    python facet_similarity.py --jsonl facets.jsonl --align syn_text_dense.npz --out syn_facets_sim.npz
        (--align: same record order as the embeddings export, so the matrices line up)
"""

import argparse
import json
import os

import numpy as np

SET_FACETS = ["precursors", "solvents", "additives"]
CATEGORY_FACETS = ["method", "product_form", "atmosphere"]
EMPTY_MEANS_NONE = {"additives"}      # empty list is a real value ("none"), not missing
UNKNOWN = {None, "", "unknown", "other"}


def load_jsonl(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["record_id"]: r for r in rows}


def load_db() -> dict[str, dict]:
    import psycopg
    from psycopg.rows import dict_row

    cols = ", ".join(SET_FACETS + CATEGORY_FACETS)
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        rows = conn.execute(f"SELECT record_id, {cols} FROM synthesis_facets").fetchall()
    return {r["record_id"]: r for r in rows}


def jaccard_matrix(sets: list[set]) -> np.ndarray:
    vocab = {v: k for k, v in enumerate(sorted(set().union(*sets)))}
    X = np.zeros((len(sets), len(vocab)), dtype=np.float32)
    for i, s in enumerate(sets):
        X[i, [vocab[v] for v in s]] = 1.0
    shared = X @ X.T
    sizes = X.sum(axis=1)
    return shared / (sizes[:, None] + sizes[None, :] - shared)


def match_matrix(values: list) -> np.ndarray:
    v = np.array([x if x not in UNKNOWN else f"__unknown_{i}" for i, x in enumerate(values)], dtype=object)
    return (v[:, None] == v[None, :]).astype(np.float32)


def facet_similarity(records: dict[str, dict], ids: list[str]) -> np.ndarray:
    slices = []
    for facet in SET_FACETS:
        sets = []
        for i, rid in enumerate(ids):
            s = set(records[rid].get(facet) or [])
            if not s:  # placeholder element: shared "none", or unique per record if missing
                s = {"__none__"} if facet in EMPTY_MEANS_NONE else {f"__unknown_{i}"}
            sets.append(s)
        slices.append(jaccard_matrix(sets))
    for facet in CATEGORY_FACETS:
        slices.append(match_matrix([records[rid].get(facet) for rid in ids]))
    return np.stack(slices, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jsonl", help="facets file from extract_facets.py (default: read the DB)")
    parser.add_argument("--align", help=".npz whose 'ids' order to follow, e.g. the dense export")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.jsonl) if args.jsonl else load_db()
    if args.align:
        ids = [str(i) for i in np.load(args.align)["ids"]]
        missing = [i for i in ids if i not in records]
        if missing:
            raise SystemExit(f"{len(missing)} records in {args.align} have no facets, e.g. {missing[:5]}")
    else:
        ids = sorted(records)

    S = facet_similarity(records, ids)
    facets = SET_FACETS + CATEGORY_FACETS
    np.savez(args.out, ids=np.array(ids), facets=np.array(facets), similarity=S)
    print(f"saved {len(ids)} records x {len(facets)} facets to {args.out}  (similarity shape {S.shape})\n")

    off = ~np.eye(len(ids), dtype=bool)
    for q, name in enumerate(facets):
        print(f"{name:13s} mean pairwise similarity {S[:, :, q][off].mean():.2f}"
              f"   min eigenvalue {np.linalg.eigvalsh(S[:, :, q]).min():+.3f}")
    if len(ids) <= 10:
        np.set_printoptions(precision=2, suppress=True)
        print("\nids:", ids)
        for q, name in enumerate(facets):
            print(f"\n{name}\n{S[:, :, q]}")


if __name__ == "__main__":
    main()
