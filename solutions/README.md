# Reference Solutions

Completed reference versions of both modules' starter scripts. **Try the
exercises yourself first** -- these are here for facilitators, for checking
your own answers afterward, or for groups that get stuck.

- `module1_solution.py` -- Module 1, using the real `promptstability` package
  and a local Ollama model, with the `PROMPT_VARIANTS` TODO filled in as one
  valid way to complete the exercise. Your own wording doesn't need to match;
  compare the *pattern* (inter-PSS generally lower/noisier than intra-PSS),
  not the exact numbers -- this is a live local model, not a fixed seed.
- `ollama_repeated_check_solution.py` -- identical to the bonus script in
  `module_1_prompt_stability/` (no TODOs to fill in); included here for
  completeness.
- `module2_solution.py` + `mock_llm2.py` -- Module 2, with the stretch-goal
  third ("moderate") annotation condition added, to show what extending the
  exercise looks like.

Run either module's solution the same way as the corresponding starter script:

```bash
cd exercises/solutions
python module1_solution.py            # needs Ollama running + promptstability installed
python ollama_repeated_check_solution.py
python module2_solution.py
```

Both write to the shared `exercises/outputs/` directory. Module 1's solution
calls a live local model, so its numbers will vary slightly run to run;
Module 2's mock mode is seeded, so its numbers are identical every run.
