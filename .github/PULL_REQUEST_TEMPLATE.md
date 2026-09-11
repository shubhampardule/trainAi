<!--
CONTRIBUTING.md asks for three things in a pull request. They are the three
headings below, and they are asked for because a reviewer cannot get any of them
from the diff.
-->

## What changed and why

<!-- The why is the part that is not in the diff. If it fixes an issue, link it. -->

## What you ran

<!--
Real commands and their real results. "Should be fine" is not an answer to this
heading; neither is a green tick with nothing behind it.

    pytest -m "not gpu"        # the default gate: no GPU, no network, a few seconds
    ruff format --check . && ruff check .

If you touched anything under `model/`, `train/` or `hardware/benchmark.py`, the
GPU suite is yours to run - nobody's CI has a GPU:

    pytest -m gpu              # say which card

A real training run is worth mentioning if you did one, with the numbers.
-->

- [ ] `pytest -m "not gpu"` passes
- [ ] `ruff format --check .` and `ruff check .` are clean
- [ ] `pytest -m gpu` - ran it / no GPU-relevant change / could not (say which, and on what card)

## Anything you could not verify

<!--
An untested path labelled untested is fine. An untested path presented as working
is not - that is the rule a PR gets asked to change over regardless of how good
the rest is. Hardware you do not have, a platform you cannot reach, a flag you
added blind: say so here and in the docs.

Write "nothing" if there is nothing.
-->

## The rules that are load-bearing

<!-- Tick what applies. Struck-through lines you can delete; N/A is a fine answer. -->

- [ ] Every new error is a `TrainAIError` subclass with a `hint=` saying what to *do*
- [ ] No `import torch` at package import time (it stays inside the function that needs it)
- [ ] Binary units throughout, via the `fmt_*` helpers in `console.py`
- [ ] Non-ASCII output routed through the `BULLET` / `DASH` / `ELLIPSIS` / `SPINNER` constants; source files stay pure ASCII
- [ ] Any claim about what fits in memory or how fast it runs comes from a real counter, not a formula
- [ ] No new required dependency (or: the reason is in the "what changed" section above)
- [ ] `CHANGELOG.md` updated, if a user would notice this
