"""Extract structured synthesis facets from free-text synthesis descriptions.

Each synthesis paragraph becomes:
    precursors        list of formulas, e.g. ["CH3NH3Cl", "PbCl2"]   (MA/FA written out)
    solvents          list, e.g. ["DMF", "DMSO"]
    additives         list, e.g. ["oleic acid", "oleylamine"]
    method            one of METHODS, e.g. "solution_growth"
    product_form      one of FORMS, e.g. "single_crystal"
    atmosphere        "inert", "air" or "unknown"
    max_temperature_c highest temperature mentioned (crude)
    max_duration_h    longest step duration mentioned, in hours (crude)

These facets feed per-facet search queries and per-facet similarity
(e.g. solvent overlap between two syntheses), instead of comparing whole paragraphs.

Two backends:
    rules (default)  regex + dictionaries. No API needed; good on HybriD3's regular text.
    llm              any OpenAI-compatible chat API (OpenAI, a local Ollama, vLLM, ...).
                     Better on messy literature text. Configure with environment variables:
                         LLM_MODEL     e.g. gpt-4o-mini, or llama3.1:8b for Ollama
                         LLM_BASE_URL  e.g. http://localhost:11434/v1 for Ollama (omit for OpenAI)
                         LLM_API_KEY   your key ("ollama" for Ollama)

Usage:
    pip install pandas                      # + openai for --backend llm, psycopg for --db
    python extract_facets.py sample_records.csv --out facets.jsonl
    python extract_facets.py sample_records.csv --out facets.jsonl --backend llm
    python extract_facets.py sample_records.csv --out facets.jsonl --db   # also store in Postgres
"""

import argparse
import json
import os
import re

import pandas as pd

METHODS = [
    "hot_injection", "inverse_temperature_crystallization", "antisolvent_spin_coating",
    "spin_coating", "vapor_diffusion", "slow_evaporation", "solution_cooling",
    "solvothermal", "solid_state", "antisolvent_precipitation", "solution_growth", "other",
]
FORMS = ["nanocrystals", "single_crystal", "thin_film", "powder", "crystals", "unknown"]
ATMOSPHERES = ["inert", "air", "unknown"]

# --- dictionaries -----------------------------------------------------------------

# canonical solvent name -> regex (case-insensitive; (?-i:...) parts are case-sensitive)
SOLVENTS = {
    "DMF": r"\bDMF\b|dimethyl\s*formamide|(?-i:\bC3H7NO\b)",
    "DMSO": r"\bDMSO\b|dimethyl\s*sul(?:f|ph)oxide|(?-i:\bC2H6OS\b)",
    "GBL": r"\bGBL\b|butyrolactone|(?-i:\bC4H6O2\b)",
    "NMP": r"\bNMP\b|methyl-?2?-?pyrrolidone",
    "DMAc": r"\bDMAc\b|dimethyl\s*acetamide",
    "acetonitrile": r"acetonitrile|(?-i:\bCH3CN\b)",
    "methanol": r"methanol|(?-i:\bCH3OH\b)",
    "ethanol": r"ethanol|(?-i:\bC2H5OH\b)",
    "isopropanol": r"isopropanol|2-propanol|\bIPA\b",
    "water": r"\bwater\b|deionized|(?-i:\bH2O\b)",
    "HI": r"hydroiodic\s+acid|(?-i:\bHI\b)",
    "HBr": r"hydrobromic\s+acid|(?-i:\bHBr\b)",
    "HCl": r"hydrochloric\s+acid|(?-i:\bHCl\b)",
    "chlorobenzene": r"chlorobenzene|(?-i:\bC6H5Cl\b|\bCB\b)",
    "chloroform": r"chloroform|(?-i:\bCHCl3\b)",
    "toluene": r"toluene|(?-i:\bC7H8\b)",
    "diethyl ether": r"diethyl\s*ether",
    "ethyl acetate": r"ethyl\s*acetate",
    "dichloromethane": r"dichloromethane|\bDCM\b",
    "octadecene": r"octadecene|\bODE\b",
    "hexane": r"\bhexanes?\b",
}
ADDITIVES = {
    "oleic acid": r"oleic\s+acid|(?-i:\bOA\b)",
    "oleylamine": r"oleylamine|\bOAm\b|\bOLA\b",
    "H3PO2": r"hypophosphorous|(?-i:\bH3PO2\b)",
}
SOLVENT_RE = {k: re.compile(v, re.I) for k, v in SOLVENTS.items()}
ADDITIVE_RE = {k: re.compile(v, re.I) for k, v in ADDITIVES.items()}

CATIONS = {"MA": "CH3NH3", "FA": "CH(NH2)2", "GA": "C(NH2)3", "EA": "CH3CH2NH3",
           "BA": "C4H9NH3", "PEA": "C6H5C2H4NH3"}
NAMES = {"methylammonium": "CH3NH3", "formamidinium": "CH(NH2)2", "guanidinium": "C(NH2)3",
         "cesium": "Cs", "caesium": "Cs", "lead": "Pb", "tin": "Sn"}
HALIDES = {"iodide": "I", "bromide": "Br", "chloride": "Cl"}
DIVALENT = {"Pb", "Sn"}

ABBREV_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(CATIONS) + r")(I|Br|Cl)(?![A-Za-z0-9])")
NAME_RE = re.compile(r"(" + "|".join(NAMES) + r")(?:\s*\(II\))?\s+(?:hydro)?(iodide|bromide|chloride)", re.I)
FORMULA_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*)+(?![A-Za-z0-9])")
ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga
Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd
Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn""".split())
NOT_PRECURSORS = {"N2", "O2", "H2", "CO2", "Ar", "NCs"}  # NCs = nanocrystals (N + Cs)
OLEATE_RE = re.compile(r"\b(Cs|Pb|Zn|Mn|Na|K)[\s-]*oleate\b", re.I)
PRODUCT_RE = re.compile(r".+(Pb|Sn|Ge|Bi|Sb).*(I|Br|Cl)\d*$")  # metal not first, e.g. CsPbBr3

METHOD_RULES = [  # first match wins, so more specific methods come first
    ("hot_injection", r"hot[\s-]*injection|(?:swiftly|rapidly|quickly)\s+injected|injected\s+(?:swiftly|rapidly|quickly)"),
    ("inverse_temperature_crystallization", r"inverse[\s-]*temperature|retrograde\s+solubility"),
    ("antisolvent_spin_coating", r"spin[\s-]*coat.*(?:anti[\s-]*solvent|dripp|quench)|anti[\s-]*solvent.*spin[\s-]*coat"),
    ("spin_coating", r"spin[\s-]*coat"),
    ("vapor_diffusion", r"vapou?r[\s-]*(?:assisted\s+)?diffusion|anti[\s-]*solvent\s+vapou?r"),
    ("slow_evaporation", r"evaporat"),
    ("solution_cooling", r"(?:slowly|gradually)\s+cooled|cooled\s+(?:slowly|down)|cooling\s+rate"),
    ("solvothermal", r"hydrothermal|solvothermal|autoclave"),
    ("solid_state", r"ball[\s-]*mill|grind|mechanochem|solid[\s-]*state"),
    ("antisolvent_precipitation", r"anti[\s-]*solvent"),
    ("solution_growth", r"dissolv|solution"),
]
FORM_RULES = [
    ("nanocrystals", r"nano[\s-]*(?:crystal|cube|plate|platelet|wire|sheet)s?|quantum\s+dots?|\bNCs?\b|\bQDs?\b"),
    ("single_crystal", r"single[\s-]*crystal"),
    ("thin_film", r"\bfilms?\b|spin[\s-]*coat"),
    ("powder", r"powder|polycrystalline"),
    ("crystals", r"crystal"),
]
ATMOSPHERE_RULES = [
    ("inert", r"glove[\s-]*box|inert\s+atmosphere|(?:under|in)\s+(?:an?\s+)?(?:N2|nitrogen|argon|Ar)\b"),
    ("air", r"ambient|(?:in|under)\s+(?:open\s+)?air\b"),
]

TEMP_RE = re.compile(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:[°º]\s*C\b|℃|deg(?:rees)?\s*C|C\b)")
ROOM_TEMP_RE = re.compile(r"room\s+temperature", re.I)
DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*[-–~]\s*~?\s*(\d+(?:\.\d+)?))?\s*"
    r"(h|hrs?|hours?|min|mins|minutes?|d|days?|weeks?)\b"
)
TO_HOURS = {"h": 1, "hr": 1, "hrs": 1, "hour": 1, "hours": 1, "min": 1 / 60, "mins": 1 / 60,
            "minute": 1 / 60, "minutes": 1 / 60, "d": 24, "day": 24, "days": 24,
            "week": 168, "weeks": 168}


# --- normalisation (shared by both backends) ------------------------------------------

def canonical_solvent(name: str) -> str:
    for canon, rx in SOLVENT_RE.items():
        if rx.search(name):
            return canon
    return name.strip().lower()


def canonical_additive(name: str) -> str:
    for canon, rx in ADDITIVE_RE.items():
        if rx.search(name):
            return canon
    return name.strip().lower()


def canonical_precursor(token: str) -> str:
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):  # "(PbCl2)" -> "PbCl2"
        token = token[1:-1]
    m = OLEATE_RE.fullmatch(token)
    if m:
        return m.group(1).capitalize() + "-oleate"
    m = ABBREV_RE.fullmatch(token)
    if m:
        return CATIONS[m.group(1)] + m.group(2)
    m = NAME_RE.fullmatch(token)
    if m:
        cation = NAMES[m.group(1).lower()]
        return cation + HALIDES[m.group(2).lower()] + ("2" if cation in DIVALENT else "")
    return token


def is_formula(token: str) -> bool:
    """Looks like a chemical formula: only element symbols, plus a digit or a two-letter
    element. Rejects acronyms made of one-letter symbols ("UV", "PSS", "NC"); the price
    is missing digit-free ones like "KI"."""
    if re.fullmatch(r"\(?[IVX]+\)?", token):  # roman numerals, e.g. "(II)"
        return False
    symbols = re.findall(r"[A-Z][a-z]?", token)
    if not symbols or not all(s in ELEMENTS for s in symbols):
        return False
    has_digit = any(c.isdigit() for c in token)
    return (len(symbols) >= 2 or has_digit) and (has_digit or any(len(s) == 2 for s in symbols))


def clean(facets: dict, product_formula: str | None = None) -> dict:
    """Normalise names, drop solvents/product from precursors, and coerce categories."""
    solvents = sorted({canonical_solvent(s) for s in facets.get("solvents") or []})
    additives = sorted({canonical_additive(a) for a in facets.get("additives") or []})
    excluded = set(solvents) | set(additives) | NOT_PRECURSORS | {product_formula}
    precursors = set()
    for p in facets.get("precursors") or []:
        p = canonical_precursor(p)
        if p in excluded or canonical_solvent(p) in SOLVENTS or PRODUCT_RE.fullmatch(p):
            continue
        precursors.add(p)

    def number(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def category(x):  # "Thin film" -> "thin_film"
        return re.sub(r"[\s-]+", "_", str(x).strip().lower()) if x else None

    method = category(facets.get("method"))
    form = category(facets.get("product_form"))
    atmosphere = category(facets.get("atmosphere"))
    return {
        "precursors": sorted(precursors),
        "solvents": solvents,
        "additives": additives,
        "method": method if method in METHODS else "other",
        "product_form": form if form in FORMS else "unknown",
        "atmosphere": atmosphere if atmosphere in ATMOSPHERES else "unknown",
        "max_temperature_c": number(facets.get("max_temperature_c")),
        "max_duration_h": number(facets.get("max_duration_h")),
    }


# --- rules backend ------------------------------------------------------------------

def first_match(rules, text, default):
    for label, pattern in rules:
        if re.search(pattern, text, re.I | re.S):
            return label
    return default


def extract_rules(text: str) -> dict:
    # Drop the "Product: ..." line so the product formula is not read as a precursor.
    body = re.sub(r"Product:.*?(?=Description:|\n|\.\s|$)", " ", text)

    precursors = [m.group(0) for m in ABBREV_RE.finditer(body)]
    precursors += [m.group(0) for m in NAME_RE.finditer(body)]
    precursors += [m.group(0) for m in OLEATE_RE.finditer(body)]
    precursors += [t for t in FORMULA_RE.findall(body) if is_formula(t)]

    temps = [float(m.group(1)) for m in TEMP_RE.finditer(text) if m.group(1)]
    if ROOM_TEMP_RE.search(text):
        temps.append(25.0)
    durations = [
        float(m.group(2) or m.group(1)) * TO_HOURS[m.group(3)] for m in DURATION_RE.finditer(text)
    ]

    return {
        "precursors": precursors,
        "solvents": [k for k, rx in SOLVENT_RE.items() if rx.search(text)],
        "additives": [k for k, rx in ADDITIVE_RE.items() if rx.search(text)],
        "method": first_match(METHOD_RULES, text, "other"),
        "product_form": first_match(FORM_RULES, text, "unknown"),
        "atmosphere": first_match(ATMOSPHERE_RULES, text, "unknown"),
        "max_temperature_c": max(temps) if temps else None,
        "max_duration_h": round(max(durations), 3) if durations else None,
    }


# --- LLM backend --------------------------------------------------------------------

PROMPT = """Extract these fields from the perovskite synthesis description below.
Reply with a single JSON object and nothing else.

- "precursors": chemical formulas of the reactants, e.g. "PbI2", "CH3NH3I". Write MA as CH3NH3 and FA as CH(NH2)2. Exclude solvents, ligands and the final product.
- "solvents": solvents, including acids used as solvent and antisolvents, as short names, e.g. "DMF", "DMSO", "GBL", "HI", "water", "chlorobenzene".
- "additives": ligands or other additives, e.g. "oleic acid", "oleylamine", "H3PO2". Empty list if none.
- "method": one of {methods}
- "product_form": one of {forms}
- "atmosphere": one of {atmospheres}
- "max_temperature_c": the highest process temperature in °C as a number, or null.
- "max_duration_h": the longest single step in hours as a number, or null.

Synthesis description:
\"\"\"{text}\"\"\""""


def extract_llm(text: str, client, model: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You extract structured data from materials synthesis text."},
            {"role": "user", "content": PROMPT.format(
                methods=METHODS, forms=FORMS, atmospheres=ATMOSPHERES, text=text)},
        ],
    )
    content = response.choices[0].message.content or ""
    start, end = content.find("{"), content.rfind("}")  # tolerate stray text or code fences
    return json.loads(content[start:end + 1]) if start >= 0 else {}


# --- storage ------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS synthesis_facets (
    record_id         TEXT PRIMARY KEY,
    precursors        TEXT[],
    solvents          TEXT[],
    additives         TEXT[],
    method            TEXT,
    product_form      TEXT,
    atmosphere        TEXT,
    max_temperature_c REAL,
    max_duration_h    REAL,
    extractor         TEXT          -- 'rules' or the LLM model name
);
"""


def save_db(rows: list[dict]) -> None:
    import psycopg

    cols = ["record_id", *rows[0]["facets"], "extractor"]
    sql = (f"INSERT INTO synthesis_facets ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT (record_id) DO UPDATE SET "
           + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[1:]))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                sql, [(r["record_id"], *r["facets"].values(), r["extractor"]) for r in rows]
            )
    print(f"stored {len(rows)} rows in synthesis_facets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--out", default="facets.jsonl")
    parser.add_argument("--id-col", default="record_id")
    parser.add_argument("--field", default="syn_text")
    parser.add_argument("--formula-col", default="formula", help="product formula, if present")
    parser.add_argument("--backend", choices=["rules", "llm"], default="rules")
    parser.add_argument("--db", action="store_true", help="also write to Postgres (DATABASE_URL)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    df = df[df[args.field].notna() & (df[args.field].str.strip() != "")]

    if args.backend == "llm":
        from openai import OpenAI

        client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"),
                        api_key=os.environ.get("LLM_API_KEY", "none"))
        model = os.environ["LLM_MODEL"]
        extractor = model
    else:
        extractor = "rules"

    rows = []
    for _, rec in df.iterrows():
        text = rec[args.field]
        raw = extract_llm(text, client, model) if args.backend == "llm" else extract_rules(text)
        product = rec.get(args.formula_col) if args.formula_col in df.columns else None
        rows.append({"record_id": rec[args.id_col], "facets": clean(raw, product), "extractor": extractor})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"record_id": r["record_id"], **r["facets"]}) + "\n")
    print(f"wrote {len(rows)} records to {args.out}\n")

    for r in rows:
        fct = r["facets"]
        print(f"{r['record_id']}\n  precursors={fct['precursors']} solvents={fct['solvents']} "
              f"additives={fct['additives']}\n  method={fct['method']} form={fct['product_form']} "
              f"atmosphere={fct['atmosphere']} T={fct['max_temperature_c']} t_h={fct['max_duration_h']}")

    if args.db and rows:
        save_db(rows)


if __name__ == "__main__":
    main()
