import requests
import json
from google.colab import files

# ============================================================
# HybriD3 API
# ============================================================

BASE_URL = "https://materials.hybrid3.duke.edu/materials"
OUTPUT_FILE = "PL_descriptions.jsonl"

# pl_dataset_idd is your existing list of 211 dataset IDs

# ============================================================
# Extract descriptions
# ============================================================

def extract_descriptions(entries):
    descriptions = []

    for entry in entries or []:
        description = entry.get("description", "")
        if description:
            descriptions.append(description.strip())

    return "\n".join(descriptions)

# ============================================================
# Loop over all 211 PL datasets
# ============================================================

results = []
failed_ids = []

for i in range(len(pl_dataset_ids)):

    dataset_id = pl_dataset_ids[i]
    url = f"{BASE_URL}/datasets/{dataset_id}/"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()

        system = data.get("system") or {}
        reference = data.get("reference") or {}

        record = {
            "dataset_id": dataset_id,
            "system_id": system.get("id"),
            "formula": system.get("formula", ""),
            "synthesis": extract_descriptions(data.get("synthesis", [])),
            "experimental": extract_descriptions(data.get("experimental", [])),
            "reference_id": reference.get("id")
        }

        results.append(record)

        print(f"[{i+1}/{len(pl_dataset_ids)}] Dataset {dataset_id} done")

    except Exception as e:
        failed_ids.append(dataset_id)
        print(f"Failed dataset {dataset_id}: {e}")

# ============================================================
# Save all data to one JSONL file
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# Summary
# ============================================================

print("\n========== RESULTS ==========")
print(f"Total PL datasets: {len(pl_dataset_ids)}")
print(f"Successfully extracted: {len(results)}")
print(f"Failed: {len(failed_ids)}")

print(f"With synthesis: {sum(bool(r['synthesis']) for r in results)}")
print(f"With experimental: {sum(bool(r['experimental']) for r in results)}")

if failed_ids:
    print("Failed IDs:", failed_ids)

# Download file in Colab
files.download(OUTPUT_FILE)
