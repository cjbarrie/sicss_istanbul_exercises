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

1. **Before running anything**, read through `starter.py` and work through
   "Check Your Understanding" below with your group.
2. Open `starter.py` and read the two prompts in the `PROMPTS` dictionary
   (marked `TODO(1)`): `strict` (narrow definition of populism) and `loose`
   (broad "anti-establishment or pro-people" definition). Write down your
   prediction: which prompt will label *more* speeches as populist, and
   will that extra labeling fall evenly across both parties or concentrate
   in one?
3. Run `starter.py` and compare the printed populism rates by party (gold
   vs. strict vs. loose) to your prediction, and look at the downstream
   regression table (`populism_label ~ party`, fit on gold/strict/loose).
4. Find `TODO(2)` (the `SEED` constant). Change it, re-run, and see whether
   the qualitative pattern (loose skews Republican more than strict) holds
   up with a different seed.
5. **Stretch goal (`TODO(3)`):** write a third prompt condition of your own
   and wire it into the two other spots marked `TODO(3)` further down the
   file.

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


## Check Your Understanding

Work through these by reading the code -- not by running it or checking a
specific number:

1. `annotate()` branches on `USE_REAL_API`. If you never set that
   environment variable, which branch always runs?
2. `mock_annotate_populism` (in `mock_llm2.py`) takes `gold_label` as an
   argument. Does that mean the mock is "cheating" by looking at the right
   answer, or is `gold_label` used differently there than a real LLM call
   would ever have access to it?
3. `run_downstream_regressions` fits the exact same formula
   (`f"{col} ~ C(party)"`) three times, changing only which column supplies
   the outcome. Why does holding the regression spec fixed matter for the
   comparison this exercise is trying to make?
4. If you complete the stretch goal and add a third condition to `PROMPTS`,
   which functions need no changes at all, and which need a one-line edit?
   (Hint: check the comprehension note in `build_annotations` and the two
   `TODO(3)` markers.)

## Reflection questions

1. Even when two annotation conditions have similar *overall* accuracy
   against a gold standard, why might they still lead to different
   downstream conclusions in a group comparison or regression? What does
   that imply about relying on overall accuracy alone to validate a
   pipeline?
2. If you only had one annotation condition's output and no gold label to
   check against (the realistic situation in most real projects), what
   would tip you off that something might be wrong with the pipeline,
   rather than the substantive finding being real?
3. Which of the two prompts here is closer to how *you* would have
   instinctively written the task instructions, before seeing this
   exercise? What does that suggest about how easily this kind of bias
   could be introduced without anyone noticing?
