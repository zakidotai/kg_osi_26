"""Generate frozen mat2vec and MatSciBERT feature arrays from PL ground truth."""

import argparse
from pathlib import Path

from .assets import DEFAULT_MODEL_DIR, ROOT
from .core import MaterialEmbedder, load_records, save_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/PL_descriptions.jsonl")
    parser.add_argument("--doi-prefix", help="Optional DOI prefix, e.g. 10.1016/")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/local/pl_embeddings")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        records = load_records(args.input, doi_prefix=args.doi_prefix)
        if not records:
            raise ValueError("No records selected; check input and DOI prefix")
        print(f"Embedding {len(records)} records with both models", flush=True)
        embedder = MaterialEmbedder(args.model_dir, args.batch_size, args.device)
        results = embedder.embed_records(records)
        save_embeddings(
            results, args.output_dir,
            metadata={**embedder.metadata(), "doi_prefix": args.doi_prefix}, input_path=args.input,
        )
        print(f"Saved {len(results)} records, 5 fields per model, to {args.output_dir}", flush=True)
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        parser.exit(1, f"Embedding failed: {error}\n")


if __name__ == "__main__":
    main()
