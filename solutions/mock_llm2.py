"""Offline stand-in for an LLM annotator, used when no API key is available.

Like module_1's mock_llm.py, this is NOT a real language model -- it is a
small deterministic function designed to demonstrate a real phenomenon from
Barrie, Palmer & Spirling (2024), "Replication for Large Language Models":
LLM annotations under different prompts/conditions can diverge from each
other in ways that are *not* random noise, but systematically related to
group membership -- which then distorts downstream group comparisons and
regression coefficients.

Two mock "conditions" are implemented:
  - "strict": a narrowly-worded definition of populism that tracks the
    human gold label closely (small random noise only).
  - "loose": a broadly-worded definition that also picks up a lot of
    generic anti-establishment rhetoric -- which, in this particular
    teaching sample, appears more often in Republican speeches than
    Democratic ones. This inflates the apparent Republican populism rate
    relative to "strict," without touching the actual (fixed) party
    balance in the data.

This is a deliberately simple, hand-built illustration of the general
pattern in Paper 2's case studies (different annotation conditions shifting
downstream regression coefficients) -- it does not reproduce their specific
numbers or datasets.
"""

import random


def mock_annotate_populism(text: str, party: str, condition: str, gold_label: int,
                            seed: int = 42) -> int:
    rng = random.Random(f"{seed}-{condition}-{text}")
    base = 0.72 if gold_label == 1 else 0.28

    if condition == "strict":
        noise = rng.uniform(-0.30, 0.30)
        score = base + noise
    elif condition == "loose":
        party_bias = 0.30 if party == "rep" else 0.05
        noise = rng.uniform(-0.22, 0.22)
        score = base + party_bias + noise
    elif condition == "moderate":
        # Stretch-goal example: a third condition landing between strict
        # and loose, illustrating that the distortion is a matter of degree,
        # not an all-or-nothing effect of "a bad prompt."
        party_bias = 0.20 if party == "rep" else 0.0
        noise = rng.uniform(-0.24, 0.24)
        score = base + party_bias + noise
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return 1 if score > 0.5 else 0
