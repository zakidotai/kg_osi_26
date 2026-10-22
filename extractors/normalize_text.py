"""MatSciBERT character normalization. Mappings live next to this file."""

import os
import pathlib

from tokenizers.normalizers import BertNormalizer

_MAP_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "vocab_mappings.txt")
with open(_MAP_PATH, encoding="utf-8") as handle:
    _MAPPINGS = {line[0]: line[2:] for line in handle.read().strip().split("\n") if line}

_BERT_NORM = BertNormalizer(
    lowercase=False,
    strip_accents=True,
    clean_text=True,
    handle_chinese_chars=True,
)


def normalize(text: str) -> str:
    lines = [_BERT_NORM.normalize_str(part) for part in str(text).split("\n")]
    out = []
    for line in lines:
        out.append("".join(_MAPPINGS.get(char, " ") for char in line))
    return "\n".join(out)
