"""MatSciBERT NER, relation classification, and sentence embeddings for PL jsonl records.

Text comes from the same records as main.py (`synthesis` and `experimental`).
Relation labeling calls the matra tool `classify_matscibert_relations`.
Embeddings follow the MatSciBERT README: normalize, tokenize, then the last hidden state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRA_ROOT = Path(os.environ.get("MATRA_ROOT", "/home/mzaki4/scr16_mshiel10/mzaki4/mzaki4/github/matra"))
MATSCIBERT_DIR = Path(
    os.environ.get(
        "MATRA_MATSCIBERT",
        str(MATRA_ROOT / "models" / "matscibert"),
    )
)
EXTRACT_PYTHON = os.environ.get(
    "MATRA_EXTRACT_PYTHON",
    "/scratch4/mshiel10/mzaki4/envs/matskraft/bin/python",
)
TEXT_KEYS = ("synthesis", "experimental")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

_TOKENIZER = None
_ENCODER = None


def load_documents(path: Optional[os.PathLike] = None) -> List[dict]:
    """Load the PL jsonl used by main.py. One JSON object per line."""
    if path is not None:
        chosen = Path(path)
    else:
        candidates = [
            REPO_ROOT / "data" / "PL_descriptions.jsonl",
            REPO_ROOT / "Data_Extraction_from_HybriD3" / "PL_descriptions_with_peak.json",
        ]
        chosen = next((p for p in candidates if p.is_file()), candidates[0])
    if not chosen.is_file():
        raise FileNotFoundError(f"JSONL not found: {chosen}")
    rows = []
    for line in chosen.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def field_text(record: dict, key: str = "synthesis") -> str:
    """Pull one text field and drop newline delimiters."""
    raw = record.get(key) or ""
    if not isinstance(raw, str):
        raw = str(raw)
    flat = raw.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", flat).strip()


@lru_cache(maxsize=1)
def _normalize():
    path = MATRA_ROOT / "matra" / "ner" / "normalize_text.py"
    spec = importlib.util.spec_from_file_location("kg_osi_matscibert_normalize", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MatSciBERT normalizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize


def sentences(record: dict, key: str = "synthesis") -> List[str]:
    """Sentence list for one record. Index i is the sentence passed to the relation tool."""
    text = field_text(record, key)
    if not text:
        return []
    normed = _normalize()(text).strip()
    if not normed:
        return []
    parts = [part.strip() for part in _SENT_SPLIT.split(normed) if part.strip()]
    return parts or [normed]


def matscibert_relations(
    doc_index: int,
    extractor_index: int,
    record: dict,
    key: str = "synthesis",
    use_pos: bool = False,
    max_pairs: int = 64,
    min_confidence: float = 0.0,
) -> dict:
    """Run MatSciBERT NER and relation classification on one jsonl text field.

    `doc_index` and `extractor_index` match the `(i, j, d)` call used by main.E.
    `key` is `synthesis` or `experimental`. Set `use_pos` to assign E1/E2 from
    part-of-speech chunks when the text has no explicit markers.
    """
    text = field_text(record, key)
    sents = sentences(record, key)
    if not text:
        return {
            "doc_index": doc_index,
            "extractor_index": extractor_index,
            "dataset_id": record.get("dataset_id"),
            "key": key,
            "text": "",
            "sentences": [],
            "result": {"success": True, "n_entities": 0, "n_relations": 0, "paragraphs": []},
        }
    result = _run_relate(sents, use_pos=use_pos, max_pairs=max_pairs, min_confidence=min_confidence)
    return {
        "doc_index": doc_index,
        "extractor_index": extractor_index,
        "dataset_id": record.get("dataset_id"),
        "doi_isbn": record.get("doi_isbn"),
        "key": key,
        "text": text,
        "sentences": sents,
        "result": result,
    }


def extract_corpus(
    records: Sequence[dict],
    keys: Iterable[str] = TEXT_KEYS,
    use_pos: bool = False,
    output_path: Optional[os.PathLike] = None,
    limit: Optional[int] = None,
    max_pairs: int = 64,
) -> List[dict]:
    """Classify every selected text field and optionally write JSONL."""
    chosen = list(records)
    if limit is not None:
        chosen = chosen[: int(limit)]
    rows = []
    for doc_index, record in enumerate(chosen):
        for slot, key in enumerate(keys):
            rows.append(
                matscibert_relations(
                    doc_index,
                    slot,
                    record,
                    key=key,
                    use_pos=use_pos,
                    max_pairs=max_pairs,
                )
            )
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def sentence_embedding(
    doc_index: int,
    sent_index: int,
    records: Sequence[dict],
    key: str = "synthesis",
):
    """MatSciBERT last hidden state for one sentence.

    Returns a float tensor of shape (sequence_length, hidden_size), the same
    tensor the MatSciBERT README reads from `model(**tokenized_sents)[0]` for
    a single sentence.
    """
    import torch

    sents = sentences(records[doc_index], key)
    if sent_index < 0 or sent_index >= len(sents):
        raise IndexError(
            f"Sentence {sent_index} is outside 0..{len(sents) - 1} for document {doc_index} key {key!r}"
        )
    tokenizer, model = _encoder()
    normed = [_normalize()(sents[sent_index])]
    tokenized = tokenizer(
        normed,
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
    )
    batch = {
        name: torch.tensor(values).long()
        for name, values in tokenized.items()
        if name in {"input_ids", "attention_mask", "token_type_ids"}
    }
    device = next(model.parameters()).device
    batch = {name: value.to(device) for name, value in batch.items()}
    with torch.no_grad():
        last_hidden_state = model(**batch)[0]
    return last_hidden_state[0].cpu()


def _run_relate(sents: Sequence[str], use_pos: bool, max_pairs: int, min_confidence: float) -> dict:
    if not Path(EXTRACT_PYTHON).is_file():
        raise FileNotFoundError(
            f"MatSciBERT extract interpreter not found: {EXTRACT_PYTHON}. Set MATRA_EXTRACT_PYTHON."
        )
    payload = {
        "texts": list(sents),
        "use_pos": bool(use_pos),
        "max_pairs": int(max_pairs),
        "min_confidence": float(min_confidence),
        "batch_size": 8,
        "include_scores": False,
    }
    script = (
        "import json,sys\n"
        "from matra.tools.classify_relations import classify_matscibert_relations\n"
        "payload=json.load(sys.stdin)\n"
        "print(json.dumps(classify_matscibert_relations(**payload)))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MATRA_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [EXTRACT_PYTHON, "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(MATRA_ROOT),
    )
    if proc.returncode != 0:
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
        raise RuntimeError(f"MatSciBERT relation tool failed:\n{tail}")
    text = (proc.stdout or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise RuntimeError(f"MatSciBERT relation tool returned no JSON:\n{text[-4000:]}")
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        return parsed


def _encoder():
    global _TOKENIZER, _ENCODER
    if _ENCODER is not None:
        return _TOKENIZER, _ENCODER
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not (MATSCIBERT_DIR / "config.json").is_file():
        raise FileNotFoundError(f"MatSciBERT weights not found in {MATSCIBERT_DIR}")
    _TOKENIZER = AutoTokenizer.from_pretrained(str(MATSCIBERT_DIR))
    _ENCODER = AutoModel.from_pretrained(str(MATSCIBERT_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ENCODER = _ENCODER.to(device).eval()
    return _TOKENIZER, _ENCODER


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Write relation output for the first N records. Default is one record."""
    import argparse

    parser = argparse.ArgumentParser(description="MatSciBERT NER and relations for PL jsonl text")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--key", default="synthesis", choices=TEXT_KEYS)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--use-pos", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "matscibert_relations.jsonl"),
    )
    parser.add_argument("--embed-doc", type=int, default=None)
    parser.add_argument("--embed-sent", type=int, default=0)
    args = parser.parse_args(argv)

    records = load_documents(args.jsonl)
    if args.embed_doc is not None:
        hidden = sentence_embedding(args.embed_doc, args.embed_sent, records, key=args.key)
        print(json.dumps({"doc_index": args.embed_doc, "sent_index": args.embed_sent, "shape": list(hidden.shape)}))
        return
    rows = extract_corpus(
        records,
        keys=(args.key,),
        use_pos=args.use_pos,
        output_path=args.output,
        limit=args.limit,
        max_pairs=args.max_pairs,
    )
    summary = {
        "n_records": len(rows),
        "output": args.output,
        "n_entities": sum(row["result"].get("n_entities") or 0 for row in rows),
        "n_relations": sum(row["result"].get("n_relations") or 0 for row in rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
