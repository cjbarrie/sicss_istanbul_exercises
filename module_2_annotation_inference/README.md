# Module 2: LLM Annotation -> Downstream Inference

**Estimated time:** 60-75 minutes
**Prerequisites:** Module 1 (helpful but not required), `exercises/requirements.txt` installed

## The idea

Module 1 asked whether the *same* prompt is stable. This module asks a
different question: if two *different-but-reasonable* prompts produce
somewhat different annotations, does that change what you'd conclude from a
downstream statistical analysis? Barrie, Palmer & Spirling (2024) found real
cases of this in published-study replications -- e.g. a coefficient that was
not statistically significant for *any* human coder became significant for
*every* LLM version they tested, just by swapping which coder (human or
model) produced the input labels. The exercise below is a small, hand-built
illustration of that same general risk -- not a reproduction of either
paper's numbers.

## What you're annotating

`exercises/data/populism_sample.csv` -- 60 excerpts from US presidential
campaign speeches (2000-2016, both parties), each with a human-coded
`gold_label` (1 = populist rhetoric present, 0 = not), drawn from the public
dataset used in the `LLReplication` replication code.

## Setup

Mock mode (default, no API key, instant):

```bash
cd exercises/module_2_annotation_inference
python starter.py
```

Real API mode (optional, ~120 calls to `gpt-4o-mini`, a few cents):

```bash
export OPENAI_API_KEY="sk-..."
export USE_REAL_API=1
python starter.py
```

## Task

1. Open `starter.py` and read the two prompts in the `PROMPTS` dictionary
   (marked `TODO(1)`): `strict` (narrow definition of populism) and `loose`
   (broad "anti-establishment or pro-people" definition). Before running
   anything, write down your prediction: which prompt will label *more*
   speeches as populist, and will that extra labeling fall evenly across
   both parties or concentrate in one?
2. Run `starter.py` and compare the printed populism rates by party (gold
   vs. strict vs. loose) to your prediction.
3. Look at the downstream regression table (`populism_label ~ party`, fit
   three times: once on gold labels, once on strict-condition labels, once
   on loose-condition labels). Compare the coefficient on being Republican
   and whether it's statistically significant in each version.
4. Open `outputs/module2_wide_comparison.csv` and find 3-5 specific
   excerpts where `strict` and `loose` disagree. Read the text. Does the
   disagreement look reasonable to you, given how the two prompts define
   populism?
5. **Stretch goal:** write a third prompt condition of your own (e.g. one
   that also varies a decoding-style setting, or targets a different
   construct) and add it to the `PROMPTS` dict and the regression comparison.

## Checkpoints (what to expect, mock mode)

- **Populism rate by party:** gold ≈ 24% (Dem) / 29% (Rep) -- a small,
  unremarkable gap. `strict` should track this gold pattern fairly closely.
  `loose` should show a *much* larger gap (Rep rate roughly double Dem's).
- **Agreement with gold label:** strict ≈ 85-90%, loose ≈ 75-80%. Loose is
  somewhat less accurate, but not dramatically so -- the real story isn't
  "loose is bad," it's "loose is biased in a specific, party-correlated way."
- **Downstream regression (`populism_label ~ party`):** gold and strict
  should both show a small, **not statistically significant** Republican
  coefficient (p > 0.3). Loose should show a **larger, often significant**
  coefficient (p often < 0.05). Same 60 texts, same regression spec, two
  different substantive conclusions about whether Republicans are "more
  populist" in this sample.


## Reflection questions

1. In your own words, why does the `loose` prompt inflate the Republican
   populism rate specifically, rather than inflating both parties' rates
   equally? (Hint: think about what "criticizing elites/the system" tends
   to sound like across the speeches in this sample, not about the model.)
2. The `loose` condition is *less accurate* against the gold label overall,
   but that's not the only thing that changed -- its *errors* are not
   evenly distributed across parties. Why does that matter more for a
   downstream group comparison than the overall accuracy number would
   suggest?
3. If you only had the `loose`-condition annotations and no gold label to
   check against (the realistic situation in most LLM-annotation projects),
   what would have tipped you off that something was wrong with the
   pipeline, rather than the substantive finding being real?
4. Paper 2 recommends validating a subsample against human/expert coding
   before trusting an LLM annotation pipeline (recommendation #2 in the
   slides). How large a validation subsample do you think would have been
   needed here to catch the strict-vs-loose divergence, and why?
5. Which of the two prompts here is closer to how *you* would have written
   the task instructions if you were coding this dataset yourself, before
   seeing this exercise? What does that tell you about how easy it is to
   introduce this kind of bias unintentionally?
