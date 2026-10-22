"""Group local PL ground truth by DOI without merging material/measurement rows.

Run from any directory with Python 3; no extra packages are required.
Formula comparisons use exact source strings, not chemical canonicalization.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
# This export stores a DOI, sometimes with a URL prefix or comma-separated ISSN.
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,]+", re.IGNORECASE)


def normalize_doi(value):
    matches = DOI_PATTERN.findall(value)
    if len(matches) != 1:
        raise ValueError(f"Expected one DOI in doi_isbn, found {len(matches)}: {value!r}")
    return matches[0].lower()


def group_records(rows, prefix):
    papers = defaultdict(list)
    all_dois = set()
    all_formula_counts = Counter()
    for row in rows:
        doi = normalize_doi(row["doi_isbn"])
        all_dois.add(doi)
        all_formula_counts[doi, row["formula"]] += 1
        if doi.startswith(prefix):
            papers[doi].append(row)

    grouped = []
    paper_audits = []
    for doi, records in sorted(papers.items()):
        by_formula = defaultdict(list)
        for row in records:
            by_formula[row["formula"]].append(row)
        collisions = []
        for formula, matches in sorted(by_formula.items()):
            if len(matches) > 1:
                collisions.append({
                    "formula": formula,
                    "dataset_ids": [row["dataset_id"] for row in matches],
                    "system_ids": [row["system_id"] for row in matches],
                    "distinct_synthesis_count": len({row["synthesis"] for row in matches}),
                    "distinct_experimental_count": len({row["experimental"] for row in matches}),
                })
        # Keep every original field, original text, and original row order.
        # A record can be a measurement of a material, so do not deduplicate.
        grouped.append({"doi": doi, "records": records})
        paper_audits.append({
            "doi": doi,
            "record_count": len(records),
            "unique_formula_count": len(by_formula),
            "formulas": {formula: len(matches) for formula, matches in sorted(by_formula.items())},
            "formula_collisions": collisions,
        })

    selected = [row for paper in grouped for row in paper["records"]]
    audit = {
        "doi_prefix": prefix,
        "formula_comparison": "Exact source strings; no chemical canonicalization.",
        "source_record_count": len(rows),
        "source_paper_count": len(all_dois),
        "source_repeated_doi_formula_groups": sum(n > 1 for n in all_formula_counts.values()),
        "selected_record_count": len(selected),
        "selected_paper_count": len(grouped),
        "selected_unique_formula_count": len({row["formula"] for row in selected}),
        "selected_unique_system_count": len({row["system_id"] for row in selected}),
        "selected_doi_formula_is_unique": all(not p["formula_collisions"] for p in paper_audits),
        "selected_missing_or_blank_fields": {
            field: sum(not row.get(field, "").strip() for row in selected)
            for field in ("formula", "synthesis", "experimental")
        },
        "papers": paper_audits,
    }
    return grouped, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/PL_descriptions.jsonl")
    parser.add_argument("--doi-prefix", default="10.1016/")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/local/pl_ground_truth_10_1016")
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                    normalize_doi(row["doi_isbn"])
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid source record at line {line_number}: {error}") from error
                rows.append(row)
    dataset_ids = [row["dataset_id"] for row in rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Source dataset_id values are not unique; resolve before grouping.")
    prefix = args.doi_prefix.strip().lower()
    if not prefix:
        parser.error("--doi-prefix must not be empty")
    grouped, audit = group_records(rows, prefix)
    if not grouped:
        parser.error(f"No records match DOI prefix {prefix!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = args.output_dir / "papers.jsonl"
    audit_path = args.output_dir / "audit.json"
    grouped_path.write_text(
        "".join(json.dumps(paper, ensure_ascii=False) + "\n" for paper in grouped),
        encoding="utf-8",
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Grouped {audit['selected_record_count']} records into {len(grouped)} papers.")
    print(f"DOI + exact formula unique: {audit['selected_doi_formula_is_unique']}")
    print(f"Grouped records: {grouped_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
