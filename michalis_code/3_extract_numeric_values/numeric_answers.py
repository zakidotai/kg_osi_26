"""Numeric similarity for temperature and time answers in the KG-GP-Embed /compare result.

Problem: /compare embeds each answer and takes the cosine similarity, so "50 °C" and
"180 °C" look like near-identical sentences. This module reads the numbers out of the
answers and, for temperature and time questions, replaces the cosine with a numeric
similarity whenever both the reference answer and the other answer state a value:

    temperature   s = 2 ** (-|T1 - T2| / 50 °C)       50 vs 100 °C -> 0.50,  50 vs 180 °C -> 0.16
    duration      s = 2 ** (-|log10(t1 / t2)|)        a factor of 10 in time halves the similarity

Both are in (0, 1], 1 = same value, as the GP API expects. Duration is compared on a
log scale because 10 min vs 20 min matters about as much as 6 h vs 12 h.

Reading values out of an answer:
    several values ("room temperature, then 50 °C")  -> the largest counts
    ranges ("6-8 h", "6 ~8 h", "6 to 8 h")             -> the midpoint
    "room temperature" / "RT" / "ambient" -> 25 °C, "overnight" -> 12 h, kelvin -> °C
A question counts as a temperature or time question from its wording ("temperature",
"how hot" / "how long", "duration", "time"); pass `kinds` to set it yourself.
All other questions, and pairs where either answer has no value ("Not stated."),
keep their cosine similarity.

Usage, client side on the API output:
    from gradio_client import Client
    from numeric_answers import apply_numeric

    result = Client("svenpvoigt/KG-GP-Embed").predict(
        questions=[...], reference_text="...", other_texts=[...], api_name="/compare")
    result = apply_numeric(result)   # same dict, numeric questions re-scored
                                     # + kinds, reference_values, other_values for checking

Or inside the Space: end compare() with  `return apply_numeric({...})`.
Also usable on raw descriptions: quantities(text) lists every temperature and duration.
"""

import math
import re

TEMP_HALF_C = 50.0          # temperature gap that halves the similarity
TIME_HALF_DECADES = 1.0     # factor of 10**this in duration halves the similarity
ROOM_TEMP_C = 25.0
OVERNIGHT_H = 12.0

_NUM = r"(\d+(?:\.\d+)?)"
_RANGE = _NUM + r"(?:\s*(?:-|–|—|~|∼|to)\s*~?\s*" + _NUM + r")?\s*"
_TEMP = re.compile(
    r"(?<![\d.])" + _RANGE
    + r"(?:(?P<c>[°º]\s*C|℃|o\s*C|deg(?:rees?)?\s*C(?:elsius)?|degrees?\s*Celsius|C)|(?P<k>K))(?![A-Za-z])"
)
_TIME_UNITS = {
    "weeks": 168, "week": 168, "days": 24, "day": 24, "d": 24,
    "hours": 1, "hour": 1, "hrs": 1, "hr": 1, "h": 1,
    "minutes": 1 / 60, "minute": 1 / 60, "mins": 1 / 60, "min": 1 / 60,
    "seconds": 1 / 3600, "second": 1 / 3600, "secs": 1 / 3600, "sec": 1 / 3600, "s": 1 / 3600,
}
_TIME = re.compile(
    r"(?<![\d.])" + _RANGE + r"(" + "|".join(sorted(_TIME_UNITS, key=len, reverse=True)) + r")\b"
)
_ROOM_TEMP = re.compile(r"room[\s-]*temperature|\bRT\b|ambient\s+temperature", re.I)
_OVERNIGHT = re.compile(r"\bovernight\b", re.I)

_QUESTION_KIND = [  # checked in order: "how long ... at that temperature?" is about time
    ("duration", re.compile(r"how\s+long|duration|\btime\b|hours|minutes", re.I)),
    ("temperature", re.compile(r"temperature|how\s+hot|°C|degrees", re.I)),
]


def _value(m: re.Match) -> float:
    lo, hi = float(m.group(1)), m.group(2)
    return (lo + float(hi)) / 2 if hi else lo


def quantities(text: str) -> dict[str, list[float]]:
    """Every temperature (°C) and duration (hours) mentioned in a text."""
    temps = []
    for m in _TEMP.finditer(text):
        temps.append(_value(m) - 273.15 if m.group("k") else _value(m))
    temps += [ROOM_TEMP_C] * len(_ROOM_TEMP.findall(text))
    hours = [_value(m) * _TIME_UNITS[m.group(3)] for m in _TIME.finditer(text)]
    hours += [OVERNIGHT_H] * len(_OVERNIGHT.findall(text))
    return {"temperature": temps, "duration": hours}


def answer_value(answer: str, kind: str) -> float | None:
    """The value an answer gives for one kind: the largest one mentioned, or None."""
    values = quantities(answer)[kind]
    return max(values) if values else None


def question_kind(question: str) -> str | None:
    for kind, pattern in _QUESTION_KIND:
        if pattern.search(question):
            return kind
    return None


def numeric_similarity(a: float, b: float, kind: str) -> float:
    if kind == "temperature":
        return 2 ** (-abs(a - b) / TEMP_HALF_C)
    a, b = max(a, 1 / 3600), max(b, 1 / 3600)   # guard against zero durations
    return 2 ** (-abs(math.log10(a / b)) / TIME_HALF_DECADES)


def apply_numeric(result: dict, kinds: list[str | None] | None = None) -> dict:
    """Re-score the temperature/time questions of a /compare result numerically."""
    questions = result["questions"]
    kinds = list(kinds) if kinds is not None else [question_kind(q) for q in questions]
    ref_values = [answer_value(a, k) if k else None for a, k in zip(result["reference_answers"], kinds)]

    sims, other_values = [], []
    for answers, row in zip(result["other_answers"], result["similarities"]):
        row = list(row)
        values = [answer_value(a, k) if k else None for a, k in zip(answers, kinds)]
        for j, (ref, v) in enumerate(zip(ref_values, values)):
            if ref is not None and v is not None:
                row[j] = round(numeric_similarity(ref, v, kinds[j]), 4)
        sims.append(row)
        other_values.append(values)

    return {**result, "similarities": sims, "kinds": kinds,
            "reference_values": ref_values, "other_values": other_values}
