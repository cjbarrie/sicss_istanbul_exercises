"""Module 2: LLM Annotation -> Downstream Inference
=======================================================

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

# ---------------------------------------------------------------------------
# TODO(2): after you've looked at the results once, change SEED to a
# different integer (e.g. 7 or 123) and re-run. Every downstream number
# (rates, coefficients, significance) comes from mock_llm2's seeded noise --
# does the QUALITATIVE story (loose inflates the Republican rate more than
# strict does) survive a different seed, or was it a one-off with seed 42?
# ---------------------------------------------------------------------------
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
    # TODO(3) [optional, do this after TODO(1) and TODO(2)]: add a third
    # condition here with its own key and prompt text (e.g. "moderate" -- a
    # definition somewhere between strict and loose). If you add one, also
    # add its name to the two TODO(3) spots further down this file
    # (`compare_group_rates` and `run_downstream_regressions`) so it flows
    # through the rest of the analysis.
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
    """CHECK YOUR UNDERSTANDING: this function branches on USE_REAL_API. If
    you never set that environment variable, which branch always runs? Now
    look at `mock_annotate_populism`'s signature in mock_llm2.py -- it takes
    `gold_label` as an argument. Does that mean the mock is "cheating" by
    looking at the right answer, or is `gold_label` used differently there
    than a real LLM call would use it? (Open mock_llm2.py to check.)"""
    if USE_REAL_API:
        return call_openai(text, PROMPTS[condition], PROMPT_POSTFIX)
    return mock_annotate_populism(text, party, condition, gold_label, seed=SEED)


def build_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate every text under both conditions. Stores one row per
    (text, condition) -- this is the reproducible record of what the
    'annotators' (prompt conditions) actually said, kept separate from the
    downstream analysis so you can always re-derive the analysis from it.

    CHECK YOUR UNDERSTANDING: this loops over `for condition in PROMPTS`,
    not a hardcoded list of ["strict", "loose"]. If you complete TODO(3) and
    add a third key to PROMPTS, does this function need any changes to pick
    it up?"""
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
    # CHECK YOUR UNDERSTANDING: in mock mode, can dropna(subset=["label"])
    # above ever actually drop a row? Look at mock_annotate_populism -- does
    # it have a code path that returns None? Now check call_openai's
    # except clause -- when would REAL API mode produce a None label?
    return long_df


def compare_group_rates(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index=["uid", "party"], columns="condition", values="label").reset_index()
    wide = wide.merge(df[["uid", "gold_label"]], on="uid")
    wide.to_csv(OUTPUT_DIR / "module2_wide_comparison.csv", index=False)

    # TODO(3) [optional]: if you added a third condition to PROMPTS, add its
    # name to this list too, or it won't show up in the rates table/plot.
    rates = wide.groupby("party")[["gold_label", "strict", "loose"]].mean()
    rates.to_csv(OUTPUT_DIR / "module2_group_rates.csv")
    return wide, rates


def run_downstream_regressions(wide: pd.DataFrame) -> pd.DataFrame:
    """The 'downstream analysis': does party predict the populism label?
    Run the SAME regression spec on gold labels, strict-condition labels,
    and loose-condition labels, and compare the coefficient on being
    Republican across the three."""
    results = []
    # TODO(3) [optional]: if you added a third condition to PROMPTS, add its
    # name to this list too, so it gets its own regression.
    for col in ["gold_label", "strict", "loose"]:
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

    print("\nPopulism rate by party (gold human label vs. two LLM prompt conditions):")
    print(rates.round(3))

    agree_strict = (wide["gold_label"] == wide["strict"]).mean()
    agree_loose = (wide["gold_label"] == wide["loose"]).mean()
    print(f"\nAgreement with gold label -- strict: {agree_strict:.2%}, loose: {agree_loose:.2%}")

    results_df = run_downstream_regressions(wide)
    print("\nDownstream logistic regression: populism_label ~ party")
    print(results_df.round(3).to_string(index=False))
    print("\nCheckpoint: gold_label and 'strict' should show a small, non-significant")
    print("Republican coefficient; 'loose' should show a larger, often statistically")
    print("significant one -- the SAME underlying texts, a different substantive conclusion.")

    plt.figure(figsize=(6, 4))
    rates.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Populism rate")
    plt.title("Populism rate by party: gold vs. strict vs. loose prompt")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "module2_group_rates_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    print("Saved annotation + regression CSVs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
