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
REFERENCES_URL = f"{BASE_URL}/references/"

INPUT_FILE = "PL_descriptions.jsonl"
BACKUP_FILE = "PL_descriptions_before_doi.jsonl"

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

print(f"Loaded {len(pl_records)} PL datasets")

# ============================================================
# 3. Collect unique reference IDs
# ============================================================

reference_ids = sorted({
    str(record["reference_id"])
    for record in pl_records
    if record.get("reference_id") is not None
})

print(f"Unique references: {len(reference_ids)}")

# ============================================================
# 4. Fetch DOI information
# ============================================================

reference_lookup = {}
failed_ids = []

session = requests.Session()

for i, reference_id in enumerate(reference_ids, start=1):

    remaining = MAX_RUNTIME - (time.monotonic() - START_TIME)

    if remaining <= 0:
        print("30-minute runtime limit reached.")
        break

    url = f"{REFERENCES_URL}{reference_id}/"

    try:
        response = session.get(
            url,
            timeout=(10, min(60, remaining))
        )

        response.raise_for_status()
        data = response.json()

        doi_isbn = data.get("doi_isbn")

        reference_lookup[reference_id] = doi_isbn

        print(
            f"[{i}/{len(reference_ids)}] "
            f"Reference {reference_id}: {doi_isbn}"
        )

    except requests.RequestException as e:
        failed_ids.append(reference_id)
        print(f"Failed reference {reference_id}: {e}")

# ============================================================
# 5. Add doi_isbn immediately after reference_id
# ============================================================

updated_records = []

for record in pl_records:

    updated_record = {}

    reference_id = record.get("reference_id")
    reference_key = (
        str(reference_id)
        if reference_id is not None
        else None
    )

    for key, value in record.items():

        # Keep original content
        updated_record[key] = value

        # Insert DOI immediately after reference_id
        if key == "reference_id":
            updated_record["doi_isbn"] = reference_lookup.get(
                reference_key
            )

    updated_records.append(updated_record)

# ============================================================
# 6. Backup and save updated JSONL
# ============================================================

# Backup original file
shutil.copyfile(INPUT_FILE, BACKUP_FILE)

# Overwrite with updated records
with open(INPUT_FILE, "w", encoding="utf-8") as f:

    for record in updated_records:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

# ============================================================
# 7. Summary
# ============================================================

with_doi = sum(
    bool(record.get("doi_isbn"))
    for record in updated_records
)

print("\n========== RESULTS ==========")

print(f"Total PL datasets: {len(updated_records)}")
print(f"Unique references: {len(reference_ids)}")
print(f"PL datasets with DOI/ISBN: {with_doi}")
print(f"PL datasets without DOI/ISBN: {len(updated_records) - with_doi}")

print(f"Failed reference IDs: {failed_ids}")

print(f"\nUpdated file: {INPUT_FILE}")
print(f"Backup file: {BACKUP_FILE}")

# ============================================================
# 8. Download updated file
# ============================================================

files.download(INPUT_FILE)
