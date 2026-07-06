"""Bonus: A Paper-2-Style Determinism Check, Run Locally
=============================================================

Paper 2 (Barrie, Palmer & Spirling) found that setting a *closed* model's
temperature to 0 does NOT make it deterministic -- repeated calls to
`gpt-4o` at temperature=0 still disagreed with each other. They also found
that open-weight, locally versioned models did NOT have this problem: at
temperature=0 with a pinned version, repeated runs were identical.

This script lets you check that claim yourself, locally, using the same
real `promptstability` machinery as starter.py -- just with the SAME prompt,
run repeatedly, at temperature=0 and a fixed seed, instead of comparing
reworded prompts.

Run from inside `module_1_prompt_stability/`:
    python ollama_repeated_check.py
"""

import json
from pathlib import Path

import ollama
import pandas as pd
from promptstability import PromptStabilityAnalysis

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "llama3.2:3b"
SEED = 42  # a fixed seed, on top of temperature=0 -- see reflection Q3
N_ITEMS = 15
REPEATS = 5  # how many times to re-annotate the identical prompt

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "integer", "enum": [0, 1]}},
    "required": ["label"],
}

ORIGINAL_PROMPT = (
    "The text provided is a party manifesto for a political party in the "
    "United Kingdom. Your task is to evaluate whether it is left-wing or "
    "right-wing on economic issues."
)
PROMPT_POSTFIX = "Respond with 0 for left-wing, or 1 for right-wing."


def annotate_ollama_deterministic(text: str, prompt: str) -> str:
    """Same idea as starter.py's annotate_ollama, but pinned to temperature=0
    and a fixed seed -- the exact condition Paper 2 tested on closed models
    and found NOT to guarantee determinism there."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        format=LABEL_SCHEMA,
        options={"temperature": 0, "seed": SEED},
    )
    return response["message"]["content"]


def parse_label(raw: str):
    try:
        return json.loads(raw)["label"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main():
    df = pd.read_csv(DATA_DIR / "manifestos_sample.csv").head(N_ITEMS)
    data = list(df["sentence_text"].values)
    print(f"Model: {MODEL} (local, via Ollama). temperature=0, seed={SEED}.")
    print(f"Re-annotating the same {len(data)} items {REPEATS} times with the identical prompt...\n")

    psa = PromptStabilityAnalysis(
        annotation_function=annotate_ollama_deterministic,
        data=data,
        parse_function=parse_label,
        load_generation_models=False,
    )

    ka_map, annotated = psa.intra_pss(
        ORIGINAL_PROMPT, PROMPT_POSTFIX, iterations=REPEATS, bootstrap_samples=500, plot=False,
    )
    annotated.to_csv(OUTPUT_DIR / "module1_ollama_repeated_annotations.csv", index=False)

    # Exact-match check: for each item, are all REPEATS labels identical?
    wide = annotated.pivot_table(index="id", columns="iteration", values="annotation")
    all_identical = wide.nunique(axis=1) == 1
    n_perfectly_stable = int(all_identical.sum())

    final_run = max(ka_map.keys())
    print(f"Krippendorff's alpha across {REPEATS} runs: {ka_map[final_run]['Average Alpha']:.3f}")
    print(f"Items with an identical label on all {REPEATS} runs: {n_perfectly_stable}/{len(data)}")

    if n_perfectly_stable == len(data):
        print(
            "\nEvery item got the exact same label every time -- consistent with Paper 2's finding "
            "that open-weight, locally versioned models ARE deterministic at temperature=0, unlike "
            "the closed models they tested (Finding 3 in the talk)."
        )
    else:
        print(
            "\nSome items flipped across runs -- inspect outputs/module1_ollama_repeated_annotations.csv "
            "to see which ones, and consider reflection question 5 below."
        )


if __name__ == "__main__":
    main()
