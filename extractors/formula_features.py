"""Structured features from the record itself, not from the prose.

formula_features: element matcher over the chemical formula (halide mix, metal site,
    halide-to-metal ratio as a dimensionality proxy, organic content).
MeasurementConditions: PL measurement temperature and excitation wavelength from the
    same PL_subsets[0] entry that get_pl reads the target from.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Z][a-z]?|\(|\)|\[|\]|\d*\.\d+|\d+")

HALIDES = ("Cl", "Br", "I")
METALS = ("Pb", "Sn", "Bi", "Ag", "Mn", "Sb", "Cd", "Ge")  # every other metal goes to "other"
A_SITE = ("Cs", "Rb", "K", "Na")  # inorganic A-site cations, not counted as metals
NONMETALS = {"H", "C", "N", "O", "F", "S", "P", "Se", "B", "Si", *HALIDES}


def parse_formula(formula: str) -> Counter:
    """element counts per formula unit; handles nested ()/[] groups and fractional amounts.
    A guest or additive after ':' (e.g. 'C8H24N2PbI4:I2') is dropped, and a variable
    composition like 'Br(x)I(3-x)' is taken at its midpoint x = 3/2."""
    formula = formula.split(":")[0]
    n = re.search(r"\(([\d.]+)-x\)", formula)
    if n:
        x = float(n.group(1)) / 2
        formula = formula.replace(n.group(0), str(x)).replace("(x)", str(x))
    is_num = lambda t: t is not None and (t[0].isdigit() or t[0] == ".")
    stack = [Counter()]
    tokens = _TOKEN.findall(formula)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in "([":
            stack.append(Counter())
            i += 1
            continue
        if is_num(tok):  # stray number, e.g. a leading coefficient
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        mult = float(nxt) if is_num(nxt) else 1.0
        if tok in ")]":
            for el, n in stack.pop().items():
                stack[-1][el] += n * mult
        else:
            stack[-1][tok] += mult
        i += 2 if is_num(nxt) else 1
    return stack[0]


def formula_features(i, j, d) -> list:
    """20 values, all roughly O(1) so the GP lengthscale bounds suit them"""
    c = parse_formula(d.get("formula") or "")
    n_x = sum(c[x] for x in HALIDES)
    metals = {el: n for el, n in c.items() if el not in NONMETALS and el not in A_SITE}
    n_m = sum(metals.values())

    halide_frac = [c[x] / n_x if n_x else 0.0 for x in HALIDES]
    metal_frac = [metals.get(m, 0.0) / n_m if n_m else 0.0 for m in METALS]
    metal_frac.append(sum(n for el, n in metals.items() if el not in METALS) / n_m if n_m else 0.0)

    per_m = lambda el: math.log1p(c[el] / n_m) if n_m else 0.0
    return [
        *halide_frac,                                   # Cl, Br, I share of the halide site
        *metal_frac,                                    # metal-site composition
        n_x / n_m / 4 if n_m else 0.0,                  # X/M: ~0.75 for 3D ABX3, 1 for 2D A2BX4, 1.5 for 0D BX6
        sum(c[a] for a in A_SITE) / n_m if n_m else 0.0,  # inorganic A-site (Cs, Rb, ...) per metal
        *[per_m(el) for el in ("C", "N", "O", "S", "F")],  # organic size and heteroatoms per metal
        c["H"] / c["C"] / 2 if c["C"] else 0.0,         # H/C: lower for aromatic spacers
    ]


def _fixed_value(record, name):
    """numeric value of e.g. 'temperature: 77.0 K' in PL_subsets[0], else None"""
    subsets = record.get("PL_subsets") or []
    for v in (subsets[0].get("fixed_values") or []) if subsets else []:
        key, _, val = v.partition(":")
        if key.strip() == name:
            num = re.search(r"\d*\.?\d+", val)  # tolerates prefixes like '≈293.0 K'
            return float(num.group()) if num else None
    return None


class MeasurementConditions:
    """[T / 300 K, T missing, excitation / 400 nm, excitation missing].
    Missing values are filled with the median over the records that report them."""

    def __init__(self, records):
        def median(xs):
            xs = sorted(x for x in xs if x is not None)
            return xs[len(xs) // 2] if xs else 0.0

        self.T = [_fixed_value(d, "temperature") for d in records]
        self.exc = [_fixed_value(d, "excitation wavelength") for d in records]
        self.T_fill, self.exc_fill = median(self.T), median(self.exc)

    def get_f(self, i, j, d):
        T, exc = self.T[i], self.exc[i]
        return [
            (T if T is not None else self.T_fill) / 300, float(T is None),
            (exc if exc is not None else self.exc_fill) / 400, float(exc is None),
        ]
