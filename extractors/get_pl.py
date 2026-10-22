def get_pl(i, j, d: dict) -> float:
    subsets = d.get("PL_subsets") or []
    if not subsets or not subsets[0].get("PL_peak_position"):
        return float("nan")
    value, *unit = subsets[0]["PL_peak_position"].split()
    value = float(value)
    if unit and unit[0] == "eV":
        value = 1239.84 / value  # convert to nm
    return value
