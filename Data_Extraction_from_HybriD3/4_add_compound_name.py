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
BACKUP_FILE = "PL_descriptions_before_compound_name.jsonl"

MAX_RUNTIME = 30 * 60  # 30 minutes
START_TIME = time.monotonic()

if not os.path.exists(INPUT_FILE):
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
# 3. Collect unique system IDs
# ============================================================

system_ids = list(dict.fromkeys(
    str(record["system_id"])
    for record in pl_records
    if record.get("system_id") is not None
))

print(f"Unique systems: {len(system_ids)}")

# ============================================================
# 4. Fetch compound names from HybriD3 API
# ============================================================

system_lookup = {}
failed_ids = []

session = requests.Session()

for i, system_id in enumerate(system_ids, start=1):

    remaining = MAX_RUNTIME - (time.monotonic() - START_TIME)

    if remaining <= 0:
        print("30-minute runtime limit reached.")
        break

    url = f"{BASE_URL}/systems/{system_id}/"

    try:
        response = session.get(
            url,
            timeout=(10, min(60, remaining))
        )
        response.raise_for_status()

        data = response.json()

        compound_name = data.get("compound_name", "")

        system_lookup[system_id] = compound_name

        print(
            f"[{i}/{len(system_ids)}] "
            f"System {system_id}: {compound_name}"
        )

    except requests.RequestException as e:
        failed_ids.append(system_id)
        print(f"Failed system {system_id}: {e}")

# ============================================================
# 5. Add compound_name immediately after formula
# ============================================================

updated_records = []

for record in pl_records:

    updated_record = {}

    system_id = record.get("system_id")
    system_key = (
        str(system_id)
        if system_id is not None
        else None
    )

    compound_name = system_lookup.get(system_key)

    for key, value in record.items():

        # Preserve existing field
        updated_record[key] = value

        # Insert compound_name after formula
        if key == "formula":
            updated_record["compound_name"] = compound_name

    updated_records.append(updated_record)

# ============================================================
# 6. Backup and overwrite JSONL
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

with_name = sum(
    bool(record.get("compound_name"))
    for record in updated_records
)

print("\n========== RESULTS ==========")

print(f"Total PL datasets: {len(updated_records)}")
print(f"PL datasets with compound name: {with_name}")
print(f"PL datasets without compound name: {len(updated_records) - with_name}")
print(f"Failed system IDs: {failed_ids}")

print(f"\nUpdated file: {INPUT_FILE}")
print(f"Backup file: {BACKUP_FILE}")

# ============================================================
# 8. Download updated file
# ============================================================

files.download(INPUT_FILE)
