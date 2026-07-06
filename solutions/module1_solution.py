"""Module 1: Prompt Stability -- REFERENCE SOLUTION
========================================================

This is a completed reference version of
`module_1_prompt_stability/starter.py`, with the TODO(1) prompt variants
filled in as one valid way to complete the exercise. Your own wording does
not need to match -- and per the exercise's own README, there is no single
"correct" shape for the resulting plot; compare patterns, not exact numbers.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import ollama
import pandas as pd
from promptstability import PromptStabilityAnalysis

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "llama3.2:3b"  # pull first: `ollama pull llama3.2:3b`
TEMPERATURE = 0.1  # Paper 1's own conservative default for annotation calls
N_ITEMS = 15  # how many manifesto sentences to annotate (keep small & fast)
INTRA_ITERATIONS = 5  # repeated runs of the *same* prompt

# ---------------------------------------------------------------------------
# Ollama structured output: constrain the model to return exactly
# {"label": 0} or {"label": 1} as valid JSON, via Ollama's `format` parameter
# (a JSON schema). This replaces free-text parsing entirely -- no regex, no
# "hope the model says just a number."
# ---------------------------------------------------------------------------
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "integer", "enum": [0, 1]}},
    "required": ["label"],
}

# ---------------------------------------------------------------------------
# The real annotation prompt, verbatim from Barrie et al.'s `promptstability`
# replication package (data/output/original_prompts.tex), used there for
# exactly this manifesto left/right-wing coding task.
# ---------------------------------------------------------------------------
ORIGINAL_PROMPT = (
    "The text provided is a party manifesto for a political party in the "
    "United Kingdom. Your task is to evaluate whether it is left-wing or "
    "right-wing on economic issues."
)
PROMPT_POSTFIX = "Respond with 0 for left-wing, or 1 for right-wing."


def annotate_ollama(text: str, prompt: str) -> str:
    """The `annotation_function` the real package calls as
    `annotation_function(item, prompt)` -- `prompt` already has the postfix
    composed in by the package for intra_pss (see ORIGINAL_PROMPT/PROMPT_POSTFIX
    above); for manual_inter_pss, each variant's full text (with its own
    postfix) is supplied directly from PROMPT_VARIANTS below."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        format=LABEL_SCHEMA,
        options={"temperature": TEMPERATURE},
    )
    return response["message"]["content"]


def parse_label(raw: str):
    """`parse_function`: turn Ollama's structured JSON string into an int
    label (or None if something went wrong -- worth inspecting, see
    reflection question 4)."""
    try:
        return json.loads(raw)["label"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# TODO(1): write 2 prompt variants at each of the THREE tiers marked below
# (0.5, 2.0, 4.0 -- trivial / moderate / drastic rewording). Each must ask
# the SAME question as ORIGINAL_PROMPT, just worded differently, and must
# include a version of PROMPT_POSTFIX itself (manual_inter_pss sends each
# prompt_text as-is -- unlike intra_pss, it does not auto-append the
# postfix for you).
#
# The other three temperatures (1.0, 3.0, 5.0) are pre-written and not part
# of the exercise -- they're there to give the resulting plot more
# resolution across the rewording spectrum (Paper 1 itself uses 25
# temperature levels; we use 6 to keep this fast). You don't need to edit
# them, but you're welcome to.
#
# Run the script once with the variants below as a baseline, then rewrite
# YOUR three tiers in your own words and re-run to compare.
# ---------------------------------------------------------------------------
PROMPT_VARIANTS = [
    {
        "prompt_id": 0,
        "temperature": 0.0,
        "prompt_text": f"{ORIGINAL_PROMPT} {PROMPT_POSTFIX}",
        "original_prompt": True,
    },
    # --- TODO tier: trivial reword (temperature 0.5) ---
    {
        "prompt_id": 1,
        "temperature": 0.5,
        "prompt_text": (
            "This is a UK party manifesto excerpt. Decide if it leans "
            f"left-wing or right-wing economically. {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    {
        "prompt_id": 2,
        "temperature": 0.5,
        "prompt_text": (
            "Here is a passage from a British political party manifesto. "
            f"Assess whether its economic stance is left-wing or right-wing. {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    # --- pre-written, not a TODO: temperature 1.0 ---
    {
        "prompt_id": 7,
        "temperature": 1.0,
        "prompt_text": (
            "In this excerpt from a UK manifesto, would you say the "
            f"economics lean left or right? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    {
        "prompt_id": 8,
        "temperature": 1.0,
        "prompt_text": (
            "Judging by this British manifesto passage, is the economic "
            f"angle left-wing or right-wing? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    # --- TODO tier: moderate reword (temperature 2.0) ---
    {
        "prompt_id": 3,
        "temperature": 2.0,
        "prompt_text": (
            "Read this excerpt from a UK party's manifesto and judge whether "
            f"its economic position sits more on the left or the right. {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    {
        "prompt_id": 4,
        "temperature": 2.0,
        "prompt_text": (
            "Considering the economic content of this British manifesto "
            f"excerpt, would you call it left-leaning or right-leaning? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    # --- pre-written, not a TODO: temperature 3.0 ---
    {
        "prompt_id": 9,
        "temperature": 3.0,
        "prompt_text": (
            "How would you characterize the economic angle here -- more "
            f"left-wing or more right-wing? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    {
        "prompt_id": 10,
        "temperature": 3.0,
        "prompt_text": (
            "Based on this manifesto snippet, does the economic thinking "
            f"lean left or right? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    # --- TODO tier: drastic reword (temperature 4.0) ---
    {
        "prompt_id": 5,
        "temperature": 4.0,
        "prompt_text": f"What's the vibe of this text -- left or right, economically speaking? {PROMPT_POSTFIX}",
        "original_prompt": False,
    },
    {
        "prompt_id": 6,
        "temperature": 4.0,
        "prompt_text": (
            "Politically, does this feel more like a left-wing or right-wing "
            f"thing to say about the economy? {PROMPT_POSTFIX}"
        ),
        "original_prompt": False,
    },
    # --- pre-written, not a TODO: temperature 5.0 ---
    {
        "prompt_id": 11,
        "temperature": 5.0,
        "prompt_text": f"Left or right? Just vibes, economically. {PROMPT_POSTFIX}",
        "original_prompt": False,
    },
    {
        "prompt_id": 12,
        "temperature": 5.0,
        "prompt_text": f"Sounds more lefty or righty to you, money-wise? {PROMPT_POSTFIX}",
        "original_prompt": False,
    },
]


def main():
    df = pd.read_csv(DATA_DIR / "manifestos_sample.csv").head(N_ITEMS)
    data = list(df["sentence_text"].values)
    print(f"Loaded {len(data)} manifesto sentences. Model: {MODEL} (via Ollama, local).")

    psa = PromptStabilityAnalysis(
        annotation_function=annotate_ollama,
        data=data,
        parse_function=parse_label,
        load_generation_models=False,  # skip loading the PEGASUS paraphraser -- we supply our own variants
    )

    print(f"\n--- Intra-PSS: same prompt, {INTRA_ITERATIONS} repeated runs ---")
    ka_intra, annotated_intra = psa.intra_pss(
        ORIGINAL_PROMPT, PROMPT_POSTFIX, iterations=INTRA_ITERATIONS, bootstrap_samples=500, plot=False,
    )
    annotated_intra.to_csv(OUTPUT_DIR / "module1_intra_annotations.csv", index=False)
    for run_count, stats in sorted(ka_intra.items()):
        print(f"  after {run_count} runs: alpha = {stats['Average Alpha']:.3f} "
              f"(95% CI {stats['CI Lower']:.3f}-{stats['CI Upper']:.3f})")

    n_temps = len({v["temperature"] for v in PROMPT_VARIANTS if not v["original_prompt"]})
    print(f"\n--- Inter-PSS: {len(PROMPT_VARIANTS) - 1} reworded variants across {n_temps} temperatures ---")
    variants_path = OUTPUT_DIR / "module1_prompt_variants.csv"
    pd.DataFrame(PROMPT_VARIANTS).to_csv(variants_path, index=False)
    ka_inter, annotated_inter = psa.manual_inter_pss(str(variants_path), bootstrap_samples=500, plot=False)
    annotated_inter.to_csv(OUTPUT_DIR / "module1_inter_annotations.csv", index=False)
    for temp, stats in sorted(ka_inter.items()):
        print(f"  temperature {temp}: alpha = {stats['Average Alpha']:.3f} "
              f"(95% CI {stats['CI Lower']:.3f}-{stats['CI Upper']:.3f})")

    print(
        "\nThere is no single 'correct' shape for the line below -- see the "
        "README's 'What to expect' section. Look at your own intra-PSS vs. "
        "inter-PSS numbers above and discuss what they suggest, whatever "
        "they show."
    )

    temps = sorted(ka_inter.keys())
    alphas = [ka_inter[t]["Average Alpha"] for t in temps]
    plt.figure(figsize=(7, 4))
    plt.plot(temps, alphas, marker="o", color="#4C72B0")
    plt.axhline(0.8, color="gray", linestyle="--", label="0.8 heuristic reference (Krippendorff, 2004)")
    plt.ylim(0, 1.05)
    plt.ylabel("Inter-prompt stability (Krippendorff's alpha)")
    plt.xlabel("Paraphraser \"temperature\" (rewording distance)")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "module1_inter_pss_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    print(f"\nSaved annotation CSVs and the prompt-variants CSV to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
