"""MatSciBERT NER, relation classification, and sentence embeddings for PL jsonl records.

The NER and relation models are called from functions in this file. They do not
import or shell out to the matra package. Task checkpoints come from the Zenodo
archive at https://zenodo.org/records/6413296 (downloaded by ensure_matscibert).
The base encoder is the Hugging Face id used by that archive's README,
m3rg-iitd/matscibert. Embeddings follow that README: normalize, tokenize, then
take the last hidden state.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer

try:
    from extractors.normalize_text import normalize
except ImportError:
    from normalize_text import normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD = "https://zenodo.org/records/6413296"
ZENODO_ZIP_URL = "https://zenodo.org/records/6413296/files/MatSciBERT.zip?download=1"
ZENODO_CACHE = REPO_ROOT / "data" / "matscibert_zenodo"
# The Zenodo scripts load this Hugging Face id for the base encoder. The zip holds the task checkpoints.
BASE_ENCODER = "m3rg-iitd/matscibert"
TEXT_KEYS = ("synthesis", "experimental")
MAX_SEQ_LEN = 512
POOL_FLAGS = ("maxpool", "average")
EMBED_CACHE_PT = Path(".cache/matscibert.pt")
EMBED_CACHE_NPZ = Path(".cache/matscibert.npz")
NER_LABELS = [
    "B-APL", "B-CMT", "B-DSC", "B-MAT", "B-PRO", "B-SMT", "B-SPL",
    "I-APL", "I-CMT", "I-DSC", "I-MAT", "I-PRO", "I-SMT", "I-SPL",
    "O",
]
RELATION_LABELS = [
    "Amount_Of", "Apparatus_Attr_Of", "Apparatus_Of", "Atmospheric_Material",
    "Brand_Of", "Condition_Of", "Coref_Of", "Descriptor_Of", "Next_Operation",
    "Number_Of", "Participant_Material", "Property_Of", "Recipe_Precursor",
    "Recipe_Target", "Solvent_Material", "Type_Of",
]
MARKER_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_WORD = re.compile(r"\S+")
_POS_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-/.:][A-Za-z0-9]+)*|[^\s]")
_POS_GRAMMAR = r"""
QP: {<CD>+<NN.*>+}
QP: {<CD>+}
NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
VP: {<MD>?<RB.*>*<VB.*>+<RP|RB.*>*}
"""
_POS_LINKS = {("NP", "NP"), ("NP", "VP"), ("VP", "NP"), ("VP", "VP"), ("QP", "NP"), ("NP", "QP"), ("QP", "VP"), ("VP", "QP")}

_ENCODER_TOK = None
_ENCODER = None
_NER = None
_RC = None


class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device, batch_first: bool = False):
        super().__init__(num_tags=num_tags, batch_first=batch_first)
        self.device = device
        start_transitions = self.start_transitions.clone().detach()
        transitions = self.transitions.clone().detach()
        num_uniq = (num_tags - 1) // 2
        for i in range(num_uniq, 2 * num_uniq):
            start_transitions[i] = -10000
            for j in range(num_tags):
                if j == i or j + num_uniq == i:
                    continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)

    def forward(self, logits, labels, masks):
        new_logits, new_labels, new_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            new_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8, device=self.device))
        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_mask = pad_sequence(new_mask, batch_first=True, padding_value=0)
        loss = -super().forward(padded_logits, padded_labels, mask=padded_mask, reduction="mean")
        if self.training:
            return (loss,)
        decoded = self.decode(padded_logits, mask=padded_mask)
        out_logits = torch.zeros_like(logits)
        for i, tags in enumerate(decoded):
            k = 0
            for j in range(len(labels[i])):
                if int(labels[i][j]) == -100:
                    continue
                out_logits[i][j][tags[k]] = 1.0
                k += 1
        return (loss, out_logits)


class BERT_CRF(nn.Module):
    def __init__(self, encoder_name: str, device, config):
        super().__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(
            encoder_name, config=config, use_auth_token=None
        )
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        logits = self.encoder(**inputs)[1]
        labels = inputs["labels"]
        return self.crf(logits, labels, labels != -100)


class BERT_RC(nn.Module):
    def __init__(self, config, tokenizer_size: int):
        super().__init__()
        self.encoder = AutoModel.from_config(copy.copy(config))
        self.encoder.resize_token_embeddings(tokenizer_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(2 * config.hidden_size, len(RELATION_LABELS))

    def forward(self, input_ids, attention_mask, entity_markers):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        rows = torch.arange(hidden.size(0), device=hidden.device)
        joined = torch.cat(
            [hidden[rows, entity_markers[:, 0]], hidden[rows, entity_markers[:, 1]]],
            dim=1,
        )
        return self.linear(self.dropout(joined))


def _is_matscibert_root(path: Path) -> bool:
    return (path / "ner" / "models" / "matscholar" / "pytorch_model.bin").is_file() and (
        path / "relation_classification" / "model" / "pytorch_model.bin"
    ).is_file()


def _find_extracted_root(directory: Path) -> Optional[Path]:
    if _is_matscibert_root(directory):
        return directory
    if not directory.is_dir():
        return None
    for child in directory.iterdir():
        if child.is_dir() and _is_matscibert_root(child):
            return child
    return None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(dest)


def ensure_matscibert() -> Path:
    """Return the extracted MatSciBERT tree from Zenodo record 6413296.

    Uses MATSCIBERT_ROOT when that directory is already extracted. Otherwise
    downloads MatSciBERT.zip into data/matscibert_zenodo and unzips it.
    """
    env = os.environ.get("MATSCIBERT_ROOT", "").strip()
    if env:
        root = _find_extracted_root(Path(env))
        if root is None:
            raise FileNotFoundError(f"MATSCIBERT_ROOT is not a MatSciBERT checkout: {env}")
        return root
    cached = _find_extracted_root(ZENODO_CACHE)
    if cached is not None:
        return cached
    ZENODO_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = ZENODO_CACHE / "MatSciBERT.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        _download(ZENODO_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(ZENODO_CACHE)
    root = _find_extracted_root(ZENODO_CACHE)
    if root is None:
        raise FileNotFoundError(f"MatSciBERT.zip from {ZENODO_RECORD} did not contain the NER and relation checkpoints")
    return root


def model_endpoints(root: Optional[Path] = None) -> dict:
    """Paths of the task checkpoints inside the Zenodo MatSciBERT tree."""
    tree = root or ensure_matscibert()
    return {
        "root": tree,
        "encoder": encoder_name(),
        "ner_weights": tree / "ner" / "models" / "matscholar" / "pytorch_model.bin",
        "relation_weights": tree / "relation_classification" / "model" / "pytorch_model.bin",
        "classification_weights": tree / "cls" / "model" / "pytorch_model.bin",
        "sofc_weights": tree / "ner" / "models" / "sofc" / "pytorch_model.bin",
        "sofc_slot_weights": tree / "ner" / "models" / "sofc_slot" / "pytorch_model.bin",
    }


def encoder_name() -> str:
    """Base MatSciBERT id. The Zenodo README loads this from Hugging Face."""
    override = os.environ.get("MATSCIBERT_DIR", "").strip()
    return override or BASE_ENCODER


def load_documents(path: Optional[os.PathLike] = None) -> List[dict]:
    """Load the PL jsonl used by main.py. One JSON object per line."""
    if path is not None:
        chosen = Path(path)
    else:
        candidates = [
            REPO_ROOT / "data" / "PL_descriptions.jsonl",
            REPO_ROOT / "Data_Extraction_from_HybriD3" / "PL_descriptions_with_peak.json",
        ]
        chosen = next((item for item in candidates if item.is_file()), candidates[0])
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
    return re.sub(r"\s+", " ", raw.replace("\r\n", "\n").replace("\r", "\n")).strip()


def sentences(record: dict, key: str = "synthesis") -> List[str]:
    """Sentence list for one record. Index i is the sentence passed to relation classification."""
    text = field_text(record, key)
    if not text:
        return []
    normed = normalize(text).strip()
    parts = [part.strip() for part in _SENT_SPLIT.split(normed) if part.strip()]
    return parts or [normed]


def ner_entities(text: str, batch_size: int = 8) -> List[dict]:
    """MatSciBERT Matscholar NER. Returns one dict per sentence with entities and character offsets."""
    sents = _sentences_from_text(text)
    if not sents:
        return []
    tokenizer, model, device = _ner_runtime()
    word_rows = [_word_spans(sent) for sent in sents]
    tag_rows = _ner_tags(model, tokenizer, [[w for w, _, _ in spans] for spans in word_rows], device, batch_size)
    out = []
    for sent, spans, tags in zip(sents, word_rows, tag_rows):
        entities = _bio_entities(sent, spans, tags)
        out.append({"text": sent, "entities": entities})
    return out


def classify_relations(
    text: str,
    use_pos: bool = False,
    max_pairs: int = 64,
    min_confidence: float = 0.0,
    batch_size: int = 8,
) -> dict:
    """Find entities, then label ordered pairs with the MatSciBERT relation classifier.

    Default pairs are MatSciBERT NER spans in the same sentence. `use_pos=True`
    instead builds noun, verb, and quantity chunks and scores neighboring chunks.
    """
    sents = _sentences_from_text(text)
    ner_rows = ner_entities(text, batch_size=batch_size) if sents else []
    paragraphs = []
    pending = []
    for sent_i, row in enumerate(ner_rows):
        sentence = row["text"]
        ner = row["entities"]
        if use_pos:
            entities, pairs, truncated = _pos_pairs(sentence, ner, max_pairs)
        else:
            entities = ner
            pairs = [
                (i, j, None)
                for i in range(len(entities))
                for j in range(len(entities))
                if i != j and not _overlaps(entities[i], entities[j])
            ]
            truncated = bool(max_pairs and len(pairs) > max_pairs)
            if truncated:
                pairs = pairs[:max_pairs]
        paragraphs.append(
            {
                "text": sentence,
                "entities": [_public_entity(ent) for ent in entities],
                "relations": [],
                "truncated_pairs": truncated,
                "pairing": "pos" if use_pos else "ner",
            }
        )
        for i, j, link in pairs:
            pending.append((sent_i, sentence, entities[i], entities[j], link))
    predictions = _classify_pairs(
        [(sentence, ent1, ent2) for _, sentence, ent1, ent2, _ in pending],
        batch_size,
    )
    n_rel = 0
    for (sent_i, _, ent1, ent2, link), pred in zip(pending, predictions):
        if pred["confidence"] < min_confidence:
            continue
        row = {
            "entity1": _public_entity(ent1),
            "entity2": _public_entity(ent2),
            "relation": pred["relation"],
            "confidence": round(pred["confidence"], 4),
        }
        if link:
            row["link"] = link
        paragraphs[sent_i]["relations"].append(row)
        n_rel += 1
    for para in paragraphs:
        para["relations"].sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "success": True,
        "n_entities": sum(len(para["entities"]) for para in paragraphs),
        "n_relations": n_rel,
        "use_pos": use_pos,
        "labels": RELATION_LABELS,
        "paragraphs": paragraphs,
    }


def matscibert_relations(
    doc_index: int,
    extractor_index: int,
    record: dict,
    key: str = "synthesis",
    use_pos: bool = False,
    max_pairs: int = 64,
    min_confidence: float = 0.0,
) -> dict:
    """Run NER and relation classification on one jsonl text field.

    `doc_index` and `extractor_index` match the `(i, j, d)` call used by main.E.
    """
    text = field_text(record, key)
    sents = sentences(record, key)
    result = classify_relations(
        text,
        use_pos=use_pos,
        max_pairs=max_pairs,
        min_confidence=min_confidence,
    ) if text else {"success": True, "n_entities": 0, "n_relations": 0, "paragraphs": []}
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
    chosen = list(records[: int(limit)] if limit is not None else records)
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
    records: Optional[Sequence[dict]] = None,
    key: str = "synthesis",
    flag: str = "average",
):
    """MatSciBERT embedding of one document, pooled over the tokens of all its sentences.

    flag is "maxpool" or "average". Returns shape (hidden_size,). Pooled vectors for the
    whole corpus are computed on the first call and cached in .cache/matscibert.pt
    (.cache/matscibert.npz when torch is missing).
    """
    if flag not in POOL_FLAGS:
        raise ValueError(f"flag must be one of {POOL_FLAGS}, got {flag!r}")
    if records is None:
        records = load_documents()
    cache = _load_embedding_cache()
    name = f"{key}-{flag}"
    if name not in cache or len(cache[name]) != len(records):
        pooled = _pooled_corpus(records, key)
        cache.update({f"{key}-{pool}": value for pool, value in pooled.items()})
        _save_embedding_cache(cache)
    return cache[name][doc_index]


def _pooled_corpus(records: Sequence[dict], key: str) -> Dict[str, torch.Tensor]:
    """Max and mean pooled last hidden states per document, each of shape (n_docs, hidden_size)."""
    tokenizer, model = _embedding_runtime()
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    out = {pool: [] for pool in POOL_FLAGS}
    for record in records:
        states = []
        for sent in sentences(record, key):
            tokenized = tokenizer(
                [normalize(sent)],
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            batch = {
                name: value.to(device)
                for name, value in tokenized.items()
                if name in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.no_grad():
                states.append(model(**batch)[0][0].cpu())
        tokens = torch.cat(states, dim=0) if states else torch.zeros(1, hidden_size)
        out["maxpool"].append(tokens.max(dim=0).values)
        out["average"].append(tokens.mean(dim=0))
    return {pool: torch.stack(values) for pool, values in out.items()}


def _load_embedding_cache() -> dict:
    try:
        import torch as _torch
    except ImportError:
        import numpy as np

        if not EMBED_CACHE_NPZ.is_file():
            return {}
        with np.load(EMBED_CACHE_NPZ) as data:
            return {name: data[name] for name in data.files}
    return _torch.load(EMBED_CACHE_PT) if EMBED_CACHE_PT.is_file() else {}


def _save_embedding_cache(cache: dict) -> None:
    EMBED_CACHE_PT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch as _torch
    except ImportError:
        import numpy as np

        np.savez(EMBED_CACHE_NPZ, **{name: np.asarray(value) for name, value in cache.items()})
        return
    _torch.save(cache, EMBED_CACHE_PT)


def _sentences_from_text(text: str) -> List[str]:
    normed = normalize(text).strip()
    if not normed:
        return []
    parts = [part.strip() for part in _SENT_SPLIT.split(normed) if part.strip()]
    return parts or [normed]


def _word_spans(sentence: str) -> List[Tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in _WORD.finditer(sentence)]


def _overlaps(a: dict, b: dict) -> bool:
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def _public_entity(ent: dict) -> dict:
    row = {"text": ent["text"], "type": ent.get("type"), "start": ent["start"], "end": ent["end"]}
    if ent.get("pos"):
        row["pos"] = ent["pos"]
    if ent.get("source"):
        row["source"] = ent["source"]
    return row


def _bio_entities(sentence: str, words: Sequence[Tuple[str, int, int]], tags: Sequence[str]) -> List[dict]:
    entities = []
    i = 0
    n = min(len(words), len(tags))
    while i < n:
        tag = tags[i]
        if not tag or tag[0] != "B":
            i += 1
            continue
        etype = tag[2:]
        j = i + 1
        while j < n and tags[j][0] == "I" and tags[j][2:] == etype:
            j += 1
        start, end = words[i][1], words[j - 1][2]
        surface = sentence[start:end].strip(".,;:")
        if surface:
            entities.append({"text": surface, "type": etype, "start": start, "end": end})
        i = j
    return entities


def _n_wordpieces(tokenizer, words: Sequence[str]) -> int:
    return len(tokenizer(list(words), is_split_into_words=True, add_special_tokens=True)["input_ids"])


def _windows(tokenizer, words: Sequence[str]) -> List[Tuple[int, int]]:
    n = len(words)
    slices = []
    start = 0
    while start < n:
        lo, hi, best = 1, n - start, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _n_wordpieces(tokenizer, words[start : start + mid]) <= MAX_SEQ_LEN:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if _n_wordpieces(tokenizer, words[start : start + best]) > MAX_SEQ_LEN:
            start += 1
            continue
        slices.append((start, start + best))
        start += best
    return slices


def _encode_tags(tags, encodings, tag2id):
    import numpy as np

    encoded = []
    for doc_labels, doc_offset in zip(tags, encodings["offset_mapping"]):
        doc_enc = np.ones(len(doc_offset), dtype=int) * -100
        arr = np.array(doc_offset)
        first = (arr[:, 0] == 0) & (arr[:, 1] != 0)
        doc_enc[first] = [tag2id[tag] for tag in doc_labels][: int(first.sum())]
        encoded.append(doc_enc.tolist())
    return encoded


def _ner_tags(model, tokenizer, sentences_words, device, batch_size: int) -> List[List[str]]:
    tag2id = {tag: i for i, tag in enumerate(NER_LABELS)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    jobs = []
    for sent_i, words in enumerate(sentences_words):
        for a, b in _windows(tokenizer, words):
            jobs.append((sent_i, a, words[a:b]))
    tags = [["O"] * len(words) for words in sentences_words]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start : start + batch_size]
            batch_words = [job[2] for job in batch_jobs]
            encodings = tokenizer(
                batch_words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=False,
            )
            labels = _encode_tags([["O"] * len(words) for words in batch_words], encodings, tag2id)
            batch = {
                "input_ids": torch.tensor(encodings["input_ids"], device=device),
                "attention_mask": torch.tensor(encodings["attention_mask"], device=device),
                "labels": torch.tensor(labels, device=device),
            }
            if "token_type_ids" in encodings:
                batch["token_type_ids"] = torch.tensor(encodings["token_type_ids"], device=device)
            pred = model(**batch)[1].argmax(dim=-1).cpu().numpy()
            lab = batch["labels"].cpu().numpy()
            for (sent_i, a, words), prediction, label in zip(batch_jobs, pred, lab):
                word_tags = [id2tag[int(p)] for p, mark in zip(prediction, label) if int(mark) != -100]
                n = min(len(word_tags), len(words))
                tags[sent_i][a : a + n] = word_tags[:n]
    return tags


def _tokenize_pair(tokenizer, text: str, e1: dict, e2: dict) -> dict:
    s1, e1e = e1["start"], e1["end"]
    s2, e2e = e2["start"], e2["end"]

    def toks(chunk: str) -> List[str]:
        return tokenizer.tokenize(normalize(chunk))

    if s1 < s2:
        tokens = (
            toks(text[:s1]) + ["[E1]"] + toks(text[s1:e1e]) + ["[/E1]"]
            + toks(text[e1e:s2]) + ["[E2]"] + toks(text[s2:e2e]) + ["[/E2]"]
            + toks(text[e2e:])
        )
    else:
        tokens = (
            toks(text[:s2]) + ["[E2]"] + toks(text[s2:e2e]) + ["[/E2]"]
            + toks(text[e2e:s1]) + ["[E1]"] + toks(text[s1:e1e]) + ["[/E1]"]
            + toks(text[e1e:])
        )
    s1_idx, e1_idx = tokens.index("[E1]"), tokens.index("[/E1]")
    s2_idx, e2_idx = tokens.index("[E2]"), tokens.index("[/E2]")
    if len(tokens) <= MAX_SEQ_LEN - 2:
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
    elif s1_idx < s2_idx:
        rem = (MAX_SEQ_LEN - 2) - (e2_idx - s1_idx + 1)
        left = max(0, s1_idx - max(rem, 0) // 2)
        right = min(len(tokens) - 1, e2_idx + max(rem, 0) // 2)
        tokens = ["[CLS]"] + tokens[left : right + 1] + ["[SEP]"]
    else:
        rem = (MAX_SEQ_LEN - 2) - (e1_idx - s2_idx + 1)
        left = max(0, s2_idx - max(rem, 0) // 2)
        right = min(len(tokens) - 1, e1_idx + max(rem, 0) // 2)
        tokens = ["[CLS]"] + tokens[left : right + 1] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention = [1] * len(input_ids)
    pad = MAX_SEQ_LEN - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad
    attention += [0] * pad
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "entity_markers": [tokens.index("[E1]"), tokens.index("[E2]")],
    }


def _classify_pairs(pairs: Sequence[Tuple[str, dict, dict]], batch_size: int) -> List[dict]:
    if not pairs:
        return []
    tokenizer, model, device = _rc_runtime()
    encoded = [_tokenize_pair(tokenizer, text, ent1, ent2) for text, ent1, ent2 in pairs]
    id2label = {i: label for i, label in enumerate(RELATION_LABELS)}
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size]
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            attention = torch.tensor([row["attention_mask"] for row in batch], dtype=torch.long, device=device)
            markers = torch.tensor([row["entity_markers"] for row in batch], dtype=torch.long, device=device)
            probs = torch.softmax(model(input_ids, attention, markers), dim=-1).cpu()
            for row in probs:
                pred = int(torch.argmax(row).item())
                out.append({"relation": id2label[pred], "confidence": float(row[pred].item())})
    return out


def _pos_pairs(sentence: str, ner: Sequence[dict], max_pairs: int):
    import nltk
    from nltk import RegexpParser
    from nltk.tree import Tree

    tokens = [(match.group(), match.start(), match.end()) for match in _POS_TOKEN.finditer(sentence)]
    words = [word for word, _, _ in tokens]
    if not words:
        return [], [], False
    tagged = nltk.pos_tag(words)
    tree = RegexpParser(_POS_GRAMMAR).parse(tagged)
    chunks = []
    index = 0
    for node in tree:
        if isinstance(node, Tree):
            width = len(node)
            start, end = tokens[index][1], tokens[index + width - 1][2]
            surface = sentence[start:end].strip(".,;:()[]")
            if surface:
                chunks.append({"text": surface, "type": node.label(), "pos": node.label(), "start": start, "end": end, "source": "pos"})
            index += width
        else:
            index += 1
    pairs = []
    for i in range(len(chunks) - 1):
        left, right = chunks[i], chunks[i + 1]
        if _overlaps(left, right) or (left["pos"], right["pos"]) not in _POS_LINKS:
            continue
        pairs.append((i, i + 1, f"{left['pos']}-{right['pos']}"))
        pairs.append((i + 1, i, f"{right['pos']}-{left['pos']}"))
    truncated = bool(max_pairs and len(pairs) > max_pairs)
    if truncated:
        pairs = pairs[:max_pairs]
    return chunks, pairs, truncated


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ner_runtime():
    global _NER
    if _NER is None:
        weights = model_endpoints()["ner_weights"]
        if not weights.is_file():
            raise FileNotFoundError(f"NER weights not found: {weights}")
        device = _device()
        name = encoder_name()
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, model_max_length=MAX_SEQ_LEN)
        config = AutoConfig.from_pretrained(name, num_labels=len(NER_LABELS))
        model = BERT_CRF(name, device, config)
        state = torch.load(str(weights), map_location="cpu")
        model.load_state_dict(state)
        _NER = (tokenizer, model.to(device).eval(), device)
    return _NER


def _rc_runtime():
    global _RC
    if _RC is None:
        weights = model_endpoints()["relation_weights"]
        if not weights.is_file():
            raise FileNotFoundError(f"Relation weights not found: {weights}")
        device = _device()
        name = encoder_name()
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, model_max_length=MAX_SEQ_LEN)
        tokenizer.add_tokens(MARKER_TOKENS)
        config = AutoConfig.from_pretrained(name)
        model = BERT_RC(config, len(tokenizer))
        state = torch.load(str(weights), map_location="cpu")
        state.pop("encoder.embeddings.position_ids", None)
        model.load_state_dict(state, strict=False)
        _RC = (tokenizer, model.to(device).eval(), device)
    return _RC


def _embedding_runtime():
    global _ENCODER_TOK, _ENCODER
    if _ENCODER is None:
        name = encoder_name()
        _ENCODER_TOK = AutoTokenizer.from_pretrained(name)
        _ENCODER = AutoModel.from_pretrained(name).to(_device()).eval()
    return _ENCODER_TOK, _ENCODER


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MatSciBERT NER and relations for PL jsonl text")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--key", default="synthesis", choices=TEXT_KEYS)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--use-pos", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "matscibert_relations.jsonl"))
    parser.add_argument("--embed-doc", type=int, default=None)
    parser.add_argument("--embed-flag", default="average", choices=POOL_FLAGS)
    parser.add_argument("--endpoints", action="store_true", help="Download the Zenodo archive if needed and print checkpoint paths")
    args = parser.parse_args(argv)
    if args.endpoints:
        ends = model_endpoints()
        print(json.dumps({key: str(value) for key, value in ends.items()}, indent=2))
        return
    records = load_documents(args.jsonl)
    if args.embed_doc is not None:
        hidden = sentence_embedding(args.embed_doc, records, key=args.key, flag=args.embed_flag)
        print(json.dumps({"doc_index": args.embed_doc, "flag": args.embed_flag, "shape": list(hidden.shape)}))
        return
    rows = extract_corpus(
        records,
        keys=(args.key,),
        use_pos=args.use_pos,
        output_path=args.output,
        limit=args.limit,
        max_pairs=args.max_pairs,
    )
    print(json.dumps({
        "n_records": len(rows),
        "output": args.output,
        "n_entities": sum(row["result"].get("n_entities") or 0 for row in rows),
        "n_relations": sum(row["result"].get("n_relations") or 0 for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
