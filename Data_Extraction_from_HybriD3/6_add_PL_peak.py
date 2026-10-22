
import requests
import json
import os
import shutil
import time
from google.colab import files

# ============================================================
# 1. Settings
# ============================================================

BASE_URL = "https://materials.hybrid3.duke.edu/materials"

INPUT_FILE = "PL_descriptions.jsonl"
BACKUP_FILE = "PL_descriptions_before_peak.jsonl"

MAX_RUNTIME = 30 * 60  # 30 minutes
START_TIME = time.monotonic()

# Upload file if needed
if not os.path.exists(INPUT_FILE):
    print(f"Please upload {INPUT_FILE}")
    files.upload()

# ============================================================
# 2. Read existing PL descriptions
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pl_records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Total PL datasets: {len(pl_records)}")

# ============================================================
# 3. Extract fixed values and PL peak position
# ============================================================

def extract_pl_information(data):

    pl_subsets = []

    wavelength_unit = (
        data.get("secondary_unit") or {}
    ).get("label", "nm")

    for subset in data.get("subsets", []):

        # ----------------------------------------------------
        # Extract all fixed values as strings
        # ----------------------------------------------------

        fixed_values = []

        for fixed in subset.get("fixed_values", []):

            prop = fixed.get("physical_property") or {}
            unit = fixed.get("unit") or {}

            name = prop.get("name", "Unknown")
            value = fixed.get("formatted", "")
            unit_label = unit.get("label", "")

            fixed_values.append(
                f"{name}: {value} {unit_label}".strip()
            )

        # ----------------------------------------------------
        # Find PL peak position
        # ----------------------------------------------------

        max_intensity = float("-inf")
        peak_wavelength = None

        for point in subset.get("datapoints", []):

            wavelength = None
            intensity = None

            for item in point.get("values", []):

                try:
                    value = float(item.get("formatted"))
                except (TypeError, ValueError):
                    continue

                if item.get("qualifier") == "secondary":
                    wavelength = value

                elif item.get("qualifier") == "primary":
                    intensity = value

            if wavelength is not None and intensity is not None:

                if intensity > max_intensity:
                    max_intensity = intensity
                    peak_wavelength = wavelength

        # ----------------------------------------------------
        # Store peak position with unit
        # ----------------------------------------------------

        if peak_wavelength is not None:
            peak_position = f"{peak_wavelength} {wavelength_unit}"
        else:
            peak_position = None

        pl_subsets.append({
            "fixed_values": fixed_values,
            "PL_peak_position": peak_position
        })

    return pl_subsets


# ============================================================
# 4. Retrieve each PL dataset from API
# ============================================================

pl_lookup = {}
failed_ids = []
timed_out = False

with requests.Session() as session:

    for i, record in enumerate(pl_records, start=1):

        dataset_id = record.get("dataset_id")

        remaining = MAX_RUNTIME - (
            time.monotonic() - START_TIME
        )

        if remaining <= 0:
            print("30-minute runtime limit reached.")
            timed_out = True
            break

        url = f"{BASE_URL}/datasets/{dataset_id}/"

        try:
            response = session.get(
                url,
                params={"format": "json"},
                timeout=(min(10, remaining), min(60, remaining))
            )

            response.raise_for_status()
            data = response.json()

            # Extract fixed values and peak positions
            pl_lookup[str(dataset_id)] = extract_pl_information(data)

            print(
                f"[{i}/{len(pl_records)}] "
                f"Dataset {dataset_id} done"
            )

        except (requests.RequestException, ValueError) as error:

            failed_ids.append(dataset_id)

            print(
                f"Could not retrieve dataset {dataset_id}: {error}"
            )

# ============================================================
# 5. Insert PL information after compound_name
# ============================================================

updated_records = []

for record in pl_records:

    updated_record = {}

    dataset_id = str(record.get("dataset_id"))

    # None if request failed or timed out
    pl_info = pl_lookup.get(dataset_id)

    # Insert after compound_name, or formula as fallback
    insert_after = (
        "compound_name"
        if "compound_name" in record
        else "formula"
    )

    for key, value in record.items():

        # Preserve every original field
        if key == "PL_subsets":
            continue

        updated_record[key] = value

        if key == insert_after:
            # Preserve previous PL data if the API request failed
            updated_record["PL_subsets"] = (
                pl_info
                if pl_info is not None
                else record.get("PL_subsets")
            )

    # Fallback if neither compound_name nor formula exists
    if "PL_subsets" not in updated_record:
        updated_record["PL_subsets"] = (
            pl_info
            if pl_info is not None
            else record.get("PL_subsets")
        )

    updated_records.append(updated_record)

# ============================================================
# 6. Backup and update original JSONL
# ============================================================

shutil.copyfile(INPUT_FILE, BACKUP_FILE)

with open(INPUT_FILE, "w", encoding="utf-8") as f:

    for record in updated_records:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

# ============================================================
# 7. Summary
# ============================================================

n_with_pl = sum(
    bool(record.get("PL_subsets"))
    for record in updated_records
)

n_with_peak = sum(
    any(
        subset.get("PL_peak_position") is not None
        for subset in (record.get("PL_subsets") or [])
    )
    for record in updated_records
)

n_total_subsets = sum(
    len(record.get("PL_subsets") or [])
    for record in updated_records
)

print("\n========== RESULTS ==========")

print(f"Total PL datasets: {len(updated_records)}")
print(f"Total PL subsets: {n_total_subsets}")
print(f"Datasets with PL information: {n_with_pl}")
print(f"Datasets with PL peak position: {n_with_peak}")
print(f"Failed dataset IDs: {failed_ids}")
print(f"Runtime limit reached: {timed_out}")

print(f"\nUpdated file: {INPUT_FILE}")
print(f"Backup file: {BACKUP_FILE}")

# ============================================================
# 8. Download updated file
# ============================================================

files.download(INPUT_FILE)
