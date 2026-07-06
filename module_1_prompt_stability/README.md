# Module 1: Prompt Stability

**Estimated time:** 60-75 minutes
**Prerequisites:** Ollama installed and running, a Python 3.8-3.10 environment (see setup below)

## The idea

An LLM annotation pipeline is only as trustworthy as it is *stable*: does it
give (roughly) the same answer if you (a) run the identical prompt again, or
(b) reword the prompt slightly while asking the same underlying question?
This is the "Prompt Stability Score" (PSS) idea from Barrie, Palaiologou &
Toernberg (2024), and this module uses their **actual, published Python
package** (`promptstability`, `pip install promptstability`) -- not a
reimplementation. We measure agreement across repeated runs / reworded
prompts using **Krippendorff's alpha (KA)** -- the same statistic used for
human inter-coder reliability. A conventional heuristic reference point is
**KA ≈ 0.8** (Krippendorff, 2004) -- not a hard pass/fail line, but a useful
benchmark.

Everything runs **entirely on your own machine** via [Ollama](https://ollama.com)
and a small open-weight model. No API key, no cost, no data leaves your
laptop.

## What you're annotating

`exercises/data/manifestos_sample.csv` -- 40 sentences from real UK party
manifestos (Benoit et al.'s Manifesto Project data, via the `promptstability`
replication package), each with a `gold_label` (0 = left-wing, 1 = right-wing
on economic issues) assigned by the original human coders. You will not need
the gold label for this module -- it's there for your own curiosity /
Module 2 comparisons.

## Setup

**1. Install Ollama and pull a small model** (one-time, ~2GB download):

```bash
# macOS: brew install ollama, or download from https://ollama.com
ollama pull llama3.2:3b
```

Check the Ollama server is running with `ollama list` -- it should show
`llama3.2:3b` in the list.

**2. `promptstability` requires Python 3.8-3.10** (it will not install on
3.11+ as of this writing). Create a **separate** virtual environment just
for this module, right here in `module_1_prompt_stability/` (this is
deliberately a different venv from any you made for Module 2 or the shared
`exercises/requirements.txt` -- keeping them separate avoids exactly the
confusion below):

```bash
cd exercises/module_1_prompt_stability   # from the repo root; adjust if you're already inside exercises/
python3.10 --version             # or python3.9 / python3.8 -- whichever you have
python3.10 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
python --version                 # VERIFY: must print 3.8.x, 3.9.x, or 3.10.x -- if it
                                  # prints 3.11+ or 3.13, stop here (see Troubleshooting)
pip install -r requirements.txt  # the file in THIS folder, not ../requirements.txt
```

If `python3.10` isn't found, get it from [python.org](https://www.python.org/downloads/)
or via your OS package manager (`brew install python@3.10` on macOS, or
`pyenv install 3.10.13` if you use pyenv -- with pyenv, run
`~/.pyenv/versions/3.10.13/bin/python3.10 -m venv venv` to be certain which
interpreter actually builds the venv, since `pyenv shell`/`pyenv global`
don't always affect what a bare `python3` resolves to in every shell).

**If you accidentally installed into a Python 3.11+ venv**, `promptstability`
won't be there and everything downstream will fail with confusing errors
(e.g. `ModuleNotFoundError: No module named 'ollama'`, because pip silently
installed the wrong/shared requirements file instead of erroring). Delete
that `venv/` folder and redo the steps above with a real 3.8-3.10 interpreter
-- don't try to debug forward from there.

**Cost/time note:** `promptstability` depends on `torch` and `transformers`
(for a paraphrasing feature this exercise doesn't use -- see below), so the
`pip install` is a genuinely large download (a few hundred MB to ~1-2GB
depending on your platform). Budget a few minutes for this on a workshop
wifi network. Nothing about *running* the exercise is slow -- only the
one-time install.

## Task

1. Run `starter.py` once, unmodified:
   ```bash
   cd exercises/module_1_prompt_stability
   python starter.py
   ```
   Look at the printed alpha values and the saved plot at
   `outputs/module1_inter_pss_plot.png`. **Read "What to expect" below
   before deciding whether your result looks right** -- there is no single
   correct shape for this plot.
2. Open `starter.py` and find `PROMPT_VARIANTS` (marked `TODO(1)`). Rewrite
   the six variant strings **at the three tiers marked TODO** (0.5, 2.0, 4.0
   -- trivial/moderate/drastic rewording) in your own words -- keep asking
   the same question (left-wing vs. right-wing on economic issues), but
   phrase it yourself. (The other three temperature levels, 1.0/3.0/5.0, are
   pre-written and just add resolution to the plot -- you don't need to
   touch them.)
3. Re-run the script. Compare your new alpha values and plot to the original
   -- as a *comparison*, not a check for whether either run is "correct."
4. **Bonus:** run `python ollama_repeated_check.py` -- a Paper-2-style check
   of whether a local model is truly deterministic at temperature=0.

**Moving on to Module 2?** Run `deactivate` first. Module 2 uses a
*different* virtual environment (the shared one at the `exercises/` root),
and staying in this venv will cause `ModuleNotFoundError: No module named
'statsmodels'` (or similar) over there -- see the top-level `exercises/README.md`,
"How to run."

## How the real package is wired up here

`starter.py` builds a `promptstability.PromptStabilityAnalysis` object exactly
as the package expects, with an `annotation_function` that calls your local
Ollama model instead of OpenAI:

- **Structured output**: the annotation function passes a JSON schema to
  Ollama's `format` parameter (`{"label": 0}` or `{"label": 1}`, nothing
  else) -- this is Ollama's built-in structured-output feature, so there's
  no regex-parsing of free text and (almost) no malformed responses.
- **`intra_pss`**: calls the real package's own repeated-run logic.
- **`manual_inter_pss`**: the real package's method for scoring
  inter-prompt stability from a CSV of *your own* prompt variants, instead
  of its built-in PEGASUS auto-paraphraser (which needs a further, heavier
  model download). `load_generation_models=False` in the constructor skips
  loading that paraphraser entirely -- you never need it for this exercise.

## What to expect (read this before judging your results)

**There is no single "correct" number or shape here, and no wrong answer
worth worrying about.** This calls a live local model with no fixed random
seed for the annotation calls themselves, on a small sample (15 items, 2
prompt variants per temperature). That combination is genuinely noisy --
we ran this exact script, unmodified, seven separate times while building
it, and no two runs looked alike. Two things *did* hold up consistently
across all seven runs, and are worth treating as reliable:

- **Intra-PSS climbs to something high** (we saw 0.83-1.00 across every
  run) -- repeating the identical prompt is consistently more stable than
  rewording it.
- **Inter-PSS is usually, but not always, lower than intra-PSS**, and the
  line across temperatures is often bumpy rather than a clean downward
  slope -- in our testing, the *overall direction* (trivial reword more
  stable than drastic reword) held about 70% of the time, not every time.

If your inter-PSS plot bounces around, goes up at the highest temperature,
or doesn't look like a Paper-1 textbook figure, **that's a normal and
expected result, not a mistake** -- it's exactly the kind of small-sample
noise that motivates Paper 1's own point that PSS estimates need enough
items and enough prompt variants to be trustworthy. Discuss what your
specific run shows rather than trying to reproduce a specific curve.

`ollama_repeated_check.py` is the one place in this module where we *do*
expect a consistent result: **alpha = 1.000, every item identical across
all 5 runs**, every time we tested it. That's because it pins both
`temperature=0` and a fixed `seed` -- true determinism with a pinned local
model, which is exactly the property Paper 2 found closed models like
`gpt-4o` do *not* reliably have. If you see anything other than perfect
agreement there, that (unlike the inter-PSS plot) is worth flagging in
discussion.


## Reflection questions

1. Compare your intra-PSS to your inter-PSS. Even if the inter-PSS *line*
   wasn't a clean decline, was inter-PSS generally lower than intra-PSS?
   Why might rewording a prompt introduce more disagreement than simply
   re-running it, whether or not that shows up as a smooth trend at this
   sample size?
2. Look at `outputs/module1_inter_annotations.csv`. Are there specific
   manifesto sentences whose label flips across *every* variant, versus ones
   that stay stable no matter how the prompt is worded? What does that
   suggest about which sentences are inherently ambiguous vs. clearly
   one-sided -- independent of whatever the overall alpha values were?
3. Paper 1 argues low (or noisy) stability is a *diagnostic*, not proof the
   labels are wrong. If you saw a low or erratic inter-PSS in a real
   project, what two or three things would you check or change -- sample
   size, number of prompt variants, the prompt itself, the model -- before
   concluding anything about whether the model "works" for this task?
4. This module ran on only 15 items and 2 variants per temperature to keep
   it fast. Based on what you saw, do you think doubling the items would
   settle the curve down, doubling the variants per temperature would,
   both, or neither? What would you actually try first, and why?
5. Ollama's structured output should make malformed responses rare. Did you
   see any `None` labels in the output CSVs anyway (check for missing rows)?
   What would you do about them in a real pipeline?
6. `ollama_repeated_check.py` pins both `temperature=0` *and* a `seed`. Paper
   2 found that temperature=0 alone did not guarantee determinism for closed
   models. Why might a `seed` parameter matter in addition to temperature,
   and why is a closed API less likely to expose (or honor) one the way a
   local model does?
