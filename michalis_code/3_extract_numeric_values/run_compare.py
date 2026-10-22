"""Call Sven's KG-GP-Embed /compare and re-score the temperature/time questions with
numeric_answers.apply_numeric.

Uses the Space's REST API directly (gradio_client hung at STARTING for us).
Put this next to numeric_answers.py and run (PowerShell):
    python run_compare.py
Optional, for more GPU quota:  $env:HF_TOKEN = "hf_..."
"""

import json
import os

import httpx  # installed with gradio_client

from numeric_answers import apply_numeric

BASE = "https://svenpvoigt-kg-gp-embed.hf.space/gradio_api"

QUESTIONS = [
    "What are the solvents?",
    "What was the temperature?",
    "How long was synthesis allowed to go on?",
]
REFERENCE_TEXT = ""  # empty = the Space's default reference (MAPbCl3 crystals, 50 °C, 6-8 h)
OTHER_TEXTS = [      # the Space's two default example texts
    "Starting materials: (PEA)2PbCl4 crystals, DMF, quartz substrates\n\nProduct: Thin film on quartz\n\n"
    "Description: Dissolve the 2D single crystals into DMF at a concentration 6%∼10% relative to the total "
    "weight. Spin-coat at 3000 rpm for 30 s on quartz substrates. Anneal in air at 100 °C for 10 min before "
    "measurement.",
    "Starting materials: (AEQT)PbBr4, quartz, acetone, ethanol\n\nProduct: (AEQT)PbBr4 thin film. Film "
    "thickness between 500 Å and 900 Å.\n\nDescription: To prepare (AEQT)PbBr4, sonicate quartz samples in 2% "
    "(w/v) detergent solution in water (20 min), then sonicate in acetone (20 min) and ethanol (20 min). Boil "
    "in ethanol (5 min) and place in a 130 °C oven to dry. Add 0.003 g of (AEQT)PbBr4 compounds to 0.1 mL of "
    "methanol, and sonicate again for 10 min. Place the charge dropwise via a syringe on the tantalum heater of "
    "the SSTA chamber. Close chamber and evacuate all solvent with a rotary mechanical pump. Switch on a "
    "turbomolecular pump, and pump system to approximately 10^{-7} Torr. To initiate the evaporation, pass a "
    "large current of approximately 65 A through the heater for about 4 s. Anneal the (AEQT)PbBr4 films at "
    "180 °C for 15 min.\n\nComment: Single source thermal ablation (SSTA) technique.",
]


def parse_events(lines) -> dict:
    """Read the Space's event stream; return the result of the 'complete' event."""
    event = None
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event == "complete":
            return json.loads(line[5:])[0]
        elif line.startswith("data:") and event == "error":
            raise RuntimeError(f"Space returned an error: {line[5:].strip()}")
    raise RuntimeError("Stream ended without a result.")


def compare(questions: list[str], reference_text: str, other_texts: list[str]) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    r = httpx.post(f"{BASE}/call/compare", json={"data": [questions, reference_text, other_texts]},
                   headers=headers, timeout=30)
    r.raise_for_status()
    url = f"{BASE}/call/compare/{r.json()['event_id']}"
    with httpx.stream("GET", url, headers=headers, timeout=httpx.Timeout(30, read=600)) as stream:
        return parse_events(stream.iter_lines())


if __name__ == "__main__":
    result = compare(QUESTIONS, REFERENCE_TEXT, OTHER_TEXTS)
    scored = apply_numeric(result)

    for j, q in enumerate(result["questions"]):
        print(f"Q{j + 1} {q}   (scored as: {scored['kinds'][j] or 'text'})")
        print(f"   reference: {result['reference_answers'][j]}")
        for i, answers in enumerate(result["other_answers"]):
            before, after = result["similarities"][i][j], scored["similarities"][i][j]
            change = f"{before} -> {after}" if before != after else f"{before}"
            print(f"   text {i + 1} [{change}]: {answers[j]}")
        print()
    print("reference values:", scored["reference_values"])
    print("other values:    ", scored["other_values"])