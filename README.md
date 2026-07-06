# SICSS Istanbul: Testing LLM Replication Fragility

Hands-on coding exercises accompanying a talk on "Replication with Language
Models." Two modules let you test, hands-on, two kinds of fragility in
LLM-based social science research: prompt sensitivity (Module 1) and how
annotation differences propagate into downstream statistical conclusions
(Module 2).

## Learning objectives

By the end of this workshop, you should be able to:

1. Explain the difference between intra-prompt stability (same prompt,
   repeated) and inter-prompt stability (reworded prompt), and compute both
   using Krippendorff's alpha -- with the actual `promptstability` package.
2. Recognize that a low stability score is a diagnostic signal, not proof
   that an LLM's labels are wrong.
3. Run repeated local-model annotations via Ollama and check whether
   temperature=0 really means "deterministic" (it does locally; it didn't
   for the closed models Paper 2 tested).
4. Demonstrate, with your own small dataset, how two reasonable-but-different
   annotation prompts can change a downstream regression's substantive
   conclusion (e.g., statistical significance flipping on or off).
5. Describe at least two concrete steps you'd take, in a real project, to
   check whether your own LLM-annotation pipeline is fragile in these ways.

## Estimated timing (2-3 hours total)

| Segment | Time |
|---|---|
| Intro / framing recap | 10-15 min |
| Module 1: Prompt Stability (incl. Ollama setup) | 75-90 min |
| Break | 10-15 min |
| Module 2: Annotation -> Downstream Inference | 60-75 min |
| Group wrap-up / discussion of reflection questions | 15-20 min |

Module 1 now includes a real, local-model install (Ollama + a small model),
so budget extra time for that one-time download versus a purely mock exercise.
Compressible to ~2 hours by pre-installing Ollama before the session starts.

## File structure

```
exercises/
├── README.md                          <- you are here
├── requirements.txt                    <- shared deps for Module 2 (any modern Python)
├── data/
│   ├── manifestos_sample.csv           <- 40 UK manifesto sentences (Module 1)
│   └── populism_sample.csv             <- 60 US campaign-speech excerpts (Module 2)
├── module_1_prompt_stability/
│   ├── README.md                       <- task instructions, checkpoints, reflection Qs
│   ├── requirements.txt                <- promptstability + ollama (needs Python 3.8-3.10)
│   ├── starter.py                      <- real `promptstability` package + local Ollama model
│   └── ollama_repeated_check.py        <- bonus: Paper-2-style temperature=0 determinism check
├── module_2_annotation_inference/
│   ├── README.md
│   ├── starter.py
│   └── mock_llm2.py
├── solutions/
│   ├── README.md
│   ├── module1_solution.py / ollama_repeated_check_solution.py
│   └── module2_solution.py / mock_llm2.py   <- includes a stretch-goal 3rd condition
└── outputs/                            <- whatever each starter.py writes when you run it
```

## Setup

**Module 2** needs only Python 3.9+ and pip, from this repo's root:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Module 1** needs two additional things. **Follow the exact commands in
`module_1_prompt_stability/README.md`, not the summary below** -- getting
the venv location or requirements file wrong is the single most common
setup error in this workshop:

1. [Ollama](https://ollama.com) installed, with a small model pulled:
   `ollama pull llama3.2:3b` (~2GB, one-time).
2. A **separate Python 3.8-3.10** virtual environment, created *inside*
   `module_1_prompt_stability/` (not shared with Module 2's venv above),
   installing `module_1_prompt_stability/requirements.txt` (not the shared
   `exercises/requirements.txt` above -- they are two different files with
   two different purposes). The module's own README has a version-check
   step to catch a wrong-Python-version venv immediately rather than
   several steps later as a confusing `ModuleNotFoundError`.

This is a deliberate change from a purely "lightweight mock" design: Module 1
now uses the actual published `promptstability` package pointed at a real
model, so the install is a bit heavier (torch/transformers as
`promptstability`'s own dependencies) -- but the payoff is that everything
you see is real model behavior, not a simulation, and it never leaves your
machine or costs anything.

## API key guidance

**You do not need any API key for either module.** Module 1 runs entirely
locally via Ollama. Module 2 defaults to a "mock mode" that simulates
realistic LLM instability offline, instantly, and at zero cost -- with an
optional real-API path if your group has an OpenAI key and wants to see it
with a live closed model:

```bash
export OPENAI_API_KEY="sk-..."
export USE_REAL_API=1
```

Never commit an API key to any file or share it in chat -- set it as an
environment variable only.

## How to run

**Module 1 and Module 2 use two different virtual environments.** Moving
between them means switching venvs, not just changing directory -- this is
the single most common mistake when running both modules in one sitting:

```bash
cd module_1_prompt_stability
source venv/bin/activate                 # Module 1's OWN venv (Python 3.8-3.10)
python starter.py
python ollama_repeated_check.py          # bonus
deactivate                               # leave Module 1's venv before switching

cd ../module_2_annotation_inference
# activate the OTHER venv here -- the shared one you made at the exercises/
# root in Setup above (e.g. `source ../venv/bin/activate`), NOT Module 1's.
# If you skip this and Module 1's venv is still active, you'll get
# `ModuleNotFoundError: No module named 'statsmodels'` (or pandas/openai) --
# Module 1's venv never installed Module 2's dependencies.
source ../venv/bin/activate
python starter.py
```

Each script prints its results to the terminal and saves annotation CSVs
(and, for Module 1, a plot) to `../outputs/`. Module 1 is a live local model
with no fixed seed, so its numbers and plot shape will differ from anyone
else's, and from your own last run -- see that module's README ("What to
expect") before treating any particular result as right or wrong. Detailed
task instructions and reflection questions are in each module's own
`README.md` -- start there.

## Troubleshooting

- **`promptstability` won't install / pip complains about Python version** --
  it requires Python 3.8-3.10. Run `python3.10 --version` (or 3.9/3.8) to
  check what's available, and build the venv with that interpreter
  specifically (`python3.10 -m venv venv`), not whatever `python3` defaults to.
- **`ModuleNotFoundError: No module named 'ollama'` (or `'promptstability'`)**
  -- almost always means you installed the *wrong* requirements file into
  Module 1's venv. From inside `module_1_prompt_stability/`, the command is
  `pip install -r requirements.txt` (the file in that same folder). If you
  instead ran `pip install -r ../requirements.txt`, that installs the
  *shared* `exercises/requirements.txt` (Module 2's dependencies), which
  does not include `ollama` or `promptstability` -- reactivate the correct
  venv and rerun with the right path. Also check `python --version` inside
  the activated venv actually shows 3.8-3.10; if it shows 3.11+ or 3.13, the
  venv was built with the wrong interpreter and needs to be recreated (see
  Module 1's README).
- **`ModuleNotFoundError: No module named 'statsmodels'` (or `'pandas'`,
  `'openai'`) in Module 2** -- you're almost certainly still in Module 1's
  virtual environment. Run `deactivate`, `cd` into
  `module_2_annotation_inference/`, activate the *shared* venv you created
  at the `exercises/` root in Setup (e.g. `source ../venv/bin/activate`),
  and rerun. `which python` (or `where python` on Windows) will show you
  which venv is currently active if you're not sure.
- **Ollama connection errors** (`ConnectionError`, `model not found`) --
  confirm the Ollama app/service is running (`ollama list` should print
  without error) and that you've pulled the model
  (`ollama pull llama3.2:3b`).
- **`FileNotFoundError` for a CSV** -- run the scripts from *inside* their
  own module folder (e.g. `module_1_prompt_stability/`), not from the
  `exercises/` root. Paths are resolved relative to the script's own
  location, so this should work from any machine/location as long as the
  folder structure above stays intact.
- **No output plot appears / script seems to hang** -- matplotlib is
  running in a non-interactive backend by design (it saves a `.png` instead
  of popping up a window); look in `../outputs/` for the file rather than
  waiting for a window.
- **`ValueError: API key not found`** (Module 2 only) -- you set
  `USE_REAL_API=1` without also setting `OPENAI_API_KEY`. Either set the
  key, or unset `USE_REAL_API` to fall back to mock mode.
- **Module 1's plot looks noisy, bumpy, or doesn't decline smoothly** --
  expected, not a bug. It's a live local model on a small sample with no
  fixed seed; we reran the identical script seven times while building this
  exercise and got seven different-looking plots. Intra-PSS settling high is
  the one consistently reliable signal. See Module 1's README ("What to
  expect") -- there's no target shape to match. Module 2's mock mode, by
  contrast, is seeded and should reproduce exactly.
- **Nobody in the group can get Ollama or Python running at all** -- pair up
  with another group and work through their `outputs/` files and the
  reflection questions together.
