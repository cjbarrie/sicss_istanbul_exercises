"""Module 2: LLM Annotation -> Downstream Inference -- REFERENCE SOLUTION
==============================================================================

This is a completed reference version of
`module_2_annotation_inference/starter.py`, extended with the stretch-goal
third condition ("moderate") to show what going further looks like. The
"strict" and "loose" logic is unchanged from the starter exercise.

Question we're asking: if two reasonable-sounding prompts produce
somewhat different annotations of the same texts, can that change the
substantive conclusion of a downstream analysis (e.g. "is Party X more
populist than Party Y")?

This mirrors the general pattern documented in Barrie, Palmer & Spirling
(2024), "Replication for Large Language Models": when they substituted
LLM-coded annotations into two published studies' original regressions,
coefficients gained or lost statistical significance depending on which
model/prompt produced the annotations -- in one case, a coefficient that
was significant for *no* human coder became significant for *every* LLM
version tested. The toy exercise below is a hand-built illustration of that
general phenomenon, at a much smaller scale, using a public dataset of
US campaign-speech excerpts -- it is NOT a reproduction of either paper's
specific numbers.

MODES
-----
Mock mode (default): no API key needed, runs instantly, uses
`mock_llm2.mock_annotate_populism`, a small stand-in that simulates two
annotation "conditions" (a narrow/strict definition of populism vs. a
broad/loose one).

Real API mode (optional):
    export OPENAI_API_KEY="sk-..."
    export USE_REAL_API=1
    python starter.py

Run this script from inside `module_2_annotation_inference/`:
    python starter.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from mock_llm2 import mock_annotate_populism

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# TODO(2) in the starter exercise: try changing this seed and re-running to
# see whether the qualitative story survives a different noise realization.
SEED = 42
USE_REAL_API = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_REAL_API", "0") == "1"

PROMPT_POSTFIX = "Respond with 0 for not populist, or 1 for populist. Respond nothing else."

# ---------------------------------------------------------------------------
# TODO(1): these two prompts operationalize "populism" differently -- one
# narrowly (closer to the classic definition used by the human coders who
# produced gold_label), one broadly. Read them, then predict: which one do
# you expect to label MORE speeches as populist overall? Write your guess
# down before running the script.
# ---------------------------------------------------------------------------
PROMPTS = {
    "strict": (
        "This is an excerpt from a US political campaign speech. Populism, "
        "narrowly defined, juxtaposes a fundamentally corrupt elite against "
        "a virtuous people and explicitly promises to restore power to the "
        "people. Does this excerpt use that specific rhetorical structure?"
    ),
    "loose": (
        "This is an excerpt from a US political campaign speech. Does this "
        "excerpt criticize elites, insiders, or 'the system' in ANY way, or "
        "position the speaker as being on the side of ordinary people?"
    ),
    # Stretch-goal condition: an intermediate definition, added here to show
    # that the strict/loose gap in the starter exercise isn't all-or-nothing
    # -- a "moderate" prompt lands, as you'd hope, somewhere in between.
    "moderate": (
        "This is an excerpt from a US political campaign speech. Does this "
        "excerpt clearly frame politics as a struggle between a corrupt "
        "establishment and ordinary people, even if less explicitly than a "
        "textbook definition of populism would require?"
    ),
}


def call_openai(text: str, prompt: str, postfix: str, temperature: float = 0.1,
                 model: str = "gpt-4o-mini"):
    """Real API call (Tier A). Only used when USE_REAL_API=1."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": f"{prompt} {postfix}"},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content.strip()
    try:
        return int(content[0])
    except (ValueError, IndexError):
        return None


def annotate(text: str, party: str, condition: str, gold_label: int) -> int:
    """See the starter exercise's "Check Your Understanding" for what this
    function's branching and mock_annotate_populism's use of gold_label
    actually mean."""
    if USE_REAL_API:
        return call_openai(text, PROMPTS[condition], PROMPT_POSTFIX)
    return mock_annotate_populism(text, party, condition, gold_label, seed=SEED)


def build_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate every text under both conditions. Stores one row per
    (text, condition) -- this is the reproducible record of what the
    'annotators' (prompt conditions) actually said, kept separate from the
    downstream analysis so you can always re-derive the analysis from it."""
    rows = []
    for _, r in df.iterrows():
        for condition in PROMPTS:
            label = annotate(r["text"], r["party"], condition, r["gold_label"])
            rows.append({
                "uid": r["uid"], "party": r["party"], "condition": condition,
                "label": label, "gold_label": r["gold_label"],
            })
    long_df = pd.DataFrame(rows).dropna(subset=["label"])
    long_df.to_csv(OUTPUT_DIR / "module2_annotations.csv", index=False)
    return long_df


def compare_group_rates(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index=["uid", "party"], columns="condition", values="label").reset_index()
    wide = wide.merge(df[["uid", "gold_label"]], on="uid")
    wide.to_csv(OUTPUT_DIR / "module2_wide_comparison.csv", index=False)

    rates = wide.groupby("party")[["gold_label"] + list(PROMPTS.keys())].mean()
    rates.to_csv(OUTPUT_DIR / "module2_group_rates.csv")
    return wide, rates


def run_downstream_regressions(wide: pd.DataFrame) -> pd.DataFrame:
    """The 'downstream analysis': does party predict the populism label?
    Run the SAME regression spec on gold labels and on every annotation
    condition in PROMPTS, and compare the coefficient on being Republican
    across all of them."""
    results = []
    for col in ["gold_label"] + list(PROMPTS.keys()):
        model = smf.logit(f"{col} ~ C(party)", data=wide).fit(disp=0)
        results.append({
            "annotation_source": col,
            "coef_rep": model.params["C(party)[T.rep]"],
            "p_value_rep": model.pvalues["C(party)[T.rep]"],
            "significant_at_0.05": model.pvalues["C(party)[T.rep]"] < 0.05,
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "module2_regression_comparison.csv", index=False)
    return results_df


def main():
    print("Mode:", "REAL API (OpenAI)" if USE_REAL_API else "MOCK (offline, no API key needed)")

    df = pd.read_csv(DATA_DIR / "populism_sample.csv")
    print(f"Loaded {len(df)} speech excerpts ({df['party'].value_counts().to_dict()})")

    long_df = build_annotations(df)
    wide, rates = compare_group_rates(df, long_df)

    print(f"\nPopulism rate by party (gold human label vs. {len(PROMPTS)} LLM prompt conditions):")
    print(rates.round(3))

    for condition in PROMPTS:
        agree = (wide["gold_label"] == wide[condition]).mean()
        print(f"Agreement with gold label -- {condition}: {agree:.2%}")

    results_df = run_downstream_regressions(wide)
    print("\nDownstream logistic regression: populism_label ~ party")
    print(results_df.round(3).to_string(index=False))
    print("\nCheckpoint: gold_label and 'strict' should show a small, non-significant")
    print("Republican coefficient; 'loose' should show a larger, statistically significant")
    print("one; 'moderate' (stretch goal) should land in between in coefficient size, and")
    print("may already cross the significance threshold -- showing how little extra")
    print("prompt bias is needed to tip a conclusion over the line.")

    plt.figure(figsize=(6, 4))
    rates.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Populism rate")
    plt.title("Populism rate by party: gold vs. strict vs. loose vs. moderate prompt")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "module2_group_rates_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    print("Saved annotation + regression CSVs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
