#collect dataset id for pl
import requests
response = requests.get(url, timeout=1800)

BASE_URL = "https://materials.hybrid3.duke.edu/materials"
DATASETS_URL = f"{BASE_URL}/datasets/"

PL_PROPERTY_ID = 4


def as_id(field):
    """Field may be an int or a nested dict."""
    if isinstance(field, dict):
        return field.get("id", field.get("pk"))
    return field


pl_dataset_ids = []

url = DATASETS_URL
page = 1

while url:

    print(f"Reading page {page}...")

    r = requests.get(
        url,
        timeout=(30, 1500)
    )
    r.raise_for_status()

    data = r.json()

    for ds in data["results"]:

        property_id = as_id(
            ds.get("primary_property")
        )

        if property_id == PL_PROPERTY_ID:

            dataset_id = ds.get("pk")

            if dataset_id is None:
                dataset_id = ds.get("id")

            pl_dataset_ids.append(dataset_id)

    url = data.get("next")
    page += 1


print()
print(
    f"Number of datasets with property_id = "
    f"{PL_PROPERTY_ID}: {len(pl_dataset_ids)}"
)

print("Dataset IDs:")
print(pl_dataset_ids)
