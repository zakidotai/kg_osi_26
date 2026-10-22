import requests
import json
import os
import shutil
import re
from collections import defaultdict
from google.colab import files


#response = requests.get(url, timeout=1800)
# ============================================================
# 1. Settings
# ============================================================

BASE_URL = "https://materials.hybrid3.duke.edu/materials"
DATASETS_URL = f"{BASE_URL}/datasets/"

INPUT_FILE = "PL_descriptions.jsonl"
BACKUP_FILE = "PL_descriptions_backup.jsonl"

ATOMIC_PROPERTY_ID = 7

# Upload file if not already in Colab
if not os.path.exists(INPUT_FILE):
    print(f"Please upload {INPUT_FILE}")
    files.upload()

# ============================================================
# 2. Read existing PL records
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pl_records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(pl_records)} PL datasets")


def get_id(value):
    if isinstance(value, dict):
        return value.get("id", value.get("pk"))
    return value


def normalize_id(value):
    value = get_id(value)
    return str(value) if value is not None else None


# ============================================================
# 3. Build PL lookup using (system_id, reference_id)
# ============================================================

pl_lookup = defaultdict(list)

for record in pl_records:

    system_id = normalize_id(record.get("system_id"))
    reference_id = normalize_id(record.get("reference_id"))

    if system_id is not None and reference_id is not None:
        key = (system_id, reference_id)
        pl_lookup[key].append(record)

print(f"Unique PL (system, reference) pairs: {len(pl_lookup)}")

# ============================================================
# 4. Scan database to find matching atomic structure datasets
# ============================================================

atomic_matches = defaultdict(set)

session = requests.Session()

url = DATASETS_URL
page = 1
total_checked = 0
atomic_checked = 0

while url:

    response = session.get(url, timeout=1800)
    response.raise_for_status()

    data = response.json()

    datasets = data.get("results", [])
    url = data.get("next")

    for ds in datasets:

        total_checked += 1

        prop_id = normalize_id(ds.get("primary_property"))

        if prop_id != str(ATOMIC_PROPERTY_ID):
            continue

        atomic_checked += 1

        system_id = normalize_id(ds.get("system"))
        reference_id = normalize_id(ds.get("reference"))

        # Match system ID and reference ID
        key = (system_id, reference_id)

        if key in pl_lookup:

            atomic_id = get_id(
                ds.get("pk", ds.get("id"))
            )

            atomic_matches[key].add(atomic_id)

    print(f"Page {page} completed")
    page += 1

print("\n========== DATABASE SCAN ==========")
print(f"Total datasets checked: {total_checked}")
print(f"Atomic structure datasets checked: {atomic_checked}")
print(f"Matching (system, reference) pairs: {len(atomic_matches)}")

# ============================================================
# 5. Extract lattice parameters
# ============================================================

def parse_number(value):
    if value is None:
        return None

    # Supports values such as "11.1763" or "11.1763 (±0.002)"
    match = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        str(value)
    )

    return float(match.group()) if match else None


def extract_lattice(data):

    lattice_subsets = []

    parameters = [
        "a", "b", "c",
        "alpha", "beta", "gamma"
    ]

    for subset in data.get("subsets", []):

        datapoints = subset.get("datapoints", [])

        # Based on atomic structure dataset 2877:
        # First six datapoints = a, b, c, alpha, beta, gamma
        if len(datapoints) < 6:
            continue

        lattice = {}

        for i, parameter in enumerate(parameters):

            values = datapoints[i].get("values", [])

            if values:
                value = values[0].get("formatted")
                lattice[parameter] = parse_number(value)
            else:
                lattice[parameter] = None

        lattice_subsets.append({
            "subset_id": subset.get("pk"),
            "crystal_system": subset.get("crystal_system"),
            "lattice_parameters": lattice
        })

    return lattice_subsets


# ============================================================
# 6. Fetch complete JSON of matching atomic datasets
# ============================================================

matching_atomic_ids = set()

for ids in atomic_matches.values():
    matching_atomic_ids.update(ids)

print(f"\nUnique matching atomic datasets: {len(matching_atomic_ids)}")

atomic_details = {}
failed_ids = []

for i, atomic_id in enumerate(
    sorted(matching_atomic_ids, key=int), start=1
):

    detail_url = f"{DATASETS_URL}{atomic_id}/"

    try:

        response = session.get(detail_url, timeout=60)
        response.raise_for_status()

        data = response.json()

        # Verify atomic structure property
        prop_id = normalize_id(data.get("primary_property"))

        if prop_id != str(ATOMIC_PROPERTY_ID):
            print(f"Skipping {atomic_id}: property is not 7")
            continue

        system_id = normalize_id(data.get("system"))
        reference_id = normalize_id(data.get("reference"))

        key = (system_id, reference_id)

        # Verify SAME system and reference as a PL dataset
        if key not in pl_lookup:
            print(f"Skipping {atomic_id}: no matching PL pair")
            continue

        atomic_details[atomic_id] = {
            "system_id": system_id,
            "reference_id": reference_id,
            "lattice_subsets": extract_lattice(data)
        }

        print(
            f"[{i}/{len(matching_atomic_ids)}] "
            f"Atomic dataset {atomic_id} done"
        )

    except Exception as e:

        failed_ids.append(atomic_id)
        print(f"Failed dataset {atomic_id}: {e}")


# ============================================================
# 7. Update existing PL records
# ============================================================

for record in pl_records:

    system_id = normalize_id(record.get("system_id"))
    reference_id = normalize_id(record.get("reference_id"))

    key = (system_id, reference_id)

    atomic_ids = sorted(
        atomic_matches.get(key, []),
        key=int
    )

    atomic_structures = []

    for atomic_id in atomic_ids:

        detail = atomic_details.get(atomic_id)

        if detail is None:
            continue

        atomic_structures.append({
            "atomic_dataset_id": atomic_id,
            "lattice_subsets": detail["lattice_subsets"]
        })

    # Preserve existing fields; add these new fields
    record["has_atomic_structure"] = bool(atomic_ids)
    record["atomic_dataset_ids"] = atomic_ids
    record["atomic_structures"] = atomic_structures


# ============================================================
# 8. Backup and save updated JSONL
# ============================================================

shutil.copyfile(INPUT_FILE, BACKUP_FILE)

with open(INPUT_FILE, "w", encoding="utf-8") as f:

    for record in pl_records:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

# ============================================================
# 9. Summary
# ============================================================

n_matched = sum(
    r["has_atomic_structure"]
    for r in pl_records
)

n_with_lattice = sum(
    any(
        all(
            value is not None
            for value in subset["lattice_parameters"].values()
        )
        for entry in r["atomic_structures"]
        for subset in entry["lattice_subsets"]
    )
    for r in pl_records
)

print("\n========== FINAL RESULTS ==========")
print(f"Total PL datasets: {len(pl_records)}")
print(f"PL datasets with matching atomic structure: {n_matched}")
print(f"PL datasets with complete lattice parameters: {n_with_lattice}")
print(f"PL datasets without matching atomic structure: {len(pl_records) - n_matched}")
print(f"Failed atomic dataset IDs: {failed_ids}")

print(f"\nUpdated file: {INPUT_FILE}")
print(f"Backup file: {BACKUP_FILE}")

# ============================================================
# 10. Download updated file
# ============================================================

files.download(INPUT_FILE)
