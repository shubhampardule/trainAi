"""``trainai quickstart`` -- the orchestration, not the five things it orchestrates.

Every step is a ``run_*`` function with its own tests, so what is left to check here is
the wiring: the order, what is reused, what is re-measured, what is passed through and
what is deliberately *not*. Those are checked against fakes rather than by training a
model, because a test that trains is a test of the trainer.

The one end-to-end run is recorded in the commit that added this file: real corpus, real
GPU, both branches of the confirmation. It is not in the suite because a real plan
measures candidates by running training steps, which is a minute even for the smallest.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import flat
from trainai.cli import main
from trainai.cli import quickstart as qs
from trainai.cli.main import app
from trainai.data.binarize import MANIFEST_NAME

runner = CliRunner()


class _Recorder:
    """Stand-ins for the five steps, recording the order and the keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.kwargs: dict[str, dict[str, Any]] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        returns = {
            "run_prepare": SimpleNamespace(total_tokens=317_284, vocab_size=8192),
            "run_plan": SimpleNamespace(
                preset_name="tiny",
                estimated_seconds=15.0,
                train_config=SimpleNamespace(steps=75),
            ),
            "run_train": None,
            "run_eval": SimpleNamespace(
                step=75,
                result_for=lambda split: SimpleNamespace(loss=5.4657, perplexity=236.44),
            ),
            "run_chat": None,
        }
        for name, value in returns.items():
            self._patch(monkeypatch, name, value)
        monkeypatch.setattr(
            qs.DatasetManifest,
            "load",
            staticmethod(lambda _: SimpleNamespace(total_tokens=1, vocab_size=2)),
        )

    def _patch(self, monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
        def fake(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            self.kwargs[name] = kwargs
            return value

        monkeypatch.setattr(qs, name, fake)


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    monkeypatch.setattr(qs.console, "input", lambda *_: "y")
    return recorder


# --------------------------------------------------------------------------- #
# The sequence
# --------------------------------------------------------------------------- #
def test_the_five_steps_run_in_the_documented_order(
    steps: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prepare, plan, train, evaluate, sample -- and the banner says five."""
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")

    assert steps.calls == ["run_prepare", "run_plan", "run_train", "run_eval", "run_chat"]
    shown = flat(capsys.readouterr().out)
    assert "Step 1 of 5" in shown
    assert "Step 5 of 5" in shown
    assert len(steps.calls) == qs.TOTAL_STEPS


def test_declining_the_confirmation_trains_nothing(
    steps: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Answering "n" has to stop *before* the expensive step, not after it.

    And it has to leave the user able to continue: the dataset and the plan are both
    still on disk and still valid, so the message names the one command that resumes.
    """
    monkeypatch.setattr(qs.console, "input", lambda *_: "n")
    out = tmp_path / "qs"

    qs.run_quickstart("corpus.txt", out=str(out))

    assert steps.calls == ["run_prepare", "run_plan"]
    shown = flat(capsys.readouterr().out)
    assert "Nothing was trained" in shown
    assert "--plan" in shown


@pytest.mark.parametrize("answer", ["", "y", "yes", "  Y  "])
def test_an_empty_answer_means_yes(
    steps: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str
) -> None:
    """Enter trains, unlike the prompt on `trainai setup --install`.

    The two differ on purpose: pip rewrites a dependency shared by the whole
    environment, while this writes into a directory the user named on the command line.
    """
    monkeypatch.setattr(qs.console, "input", lambda *_: answer)

    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")

    assert "run_train" in steps.calls


def test_yes_asks_nothing(
    steps: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(*_: Any) -> str:
        raise AssertionError("--yes must not read stdin")

    monkeypatch.setattr(qs.console, "input", refuse)

    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once", assume_yes=True)

    assert "run_train" in steps.calls


# --------------------------------------------------------------------------- #
# What is reused and what is measured again
# --------------------------------------------------------------------------- #
def _already_prepared(out: Path) -> None:
    (out / "dataset").mkdir(parents=True)
    (out / "dataset" / MANIFEST_NAME).write_text("{}", encoding="utf-8", newline="\n")


def test_an_existing_dataset_is_reused_and_the_plan_is_not(
    steps: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The asymmetry is the design, so it is the thing worth pinning.

    Preparing the same corpus twice with the same settings and the same seed produces the
    same bytes, so a second pass could only be slower. A plan is a measurement of this
    machine at one moment -- free VRAM moves when a browser opens -- and training on a
    stale measurement is the failure this project exists to avoid.
    """
    out = tmp_path / "qs"
    _already_prepared(out)

    qs.run_quickstart("corpus.txt", out=str(out), prompt="Once")

    assert steps.calls == ["run_plan", "run_train", "run_eval", "run_chat"]
    assert "already prepared" in flat(capsys.readouterr().out)


def test_force_prepares_again(steps: _Recorder, tmp_path: Path) -> None:
    out = tmp_path / "qs"
    _already_prepared(out)

    qs.run_quickstart("corpus.txt", out=str(out), prompt="Once", force=True)

    assert steps.calls[0] == "run_prepare"
    assert steps.kwargs["run_prepare"]["force"] is True
    assert steps.kwargs["run_train"]["force"] is True


# --------------------------------------------------------------------------- #
# Pass-through
# --------------------------------------------------------------------------- #
def test_an_unset_precision_is_left_to_the_plan(steps: _Recorder, tmp_path: Path) -> None:
    """The bug this is here to prevent, found while writing the command.

    ``run_train`` treats any value it is given as an override of the plan, and the plan
    carries a precision that was *measured* on this machine. Defaulting this command's
    flag to ``"auto"`` -- as every other command does -- would therefore hand ``train`` an
    override on every run and discard the measurement. So the default here is ``None``,
    and only the steps with no plan to consult get ``"auto"``.
    """
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")

    assert steps.kwargs["run_train"]["precision"] is None
    assert steps.kwargs["run_train"]["device"] is None
    assert steps.kwargs["run_plan"]["precision"] == "auto"
    assert steps.kwargs["run_eval"]["precision"] == "auto"


def test_an_explicit_precision_reaches_every_step(steps: _Recorder, tmp_path: Path) -> None:
    qs.run_quickstart(
        "corpus.txt", out=str(tmp_path / "qs"), prompt="Once", precision="fp32", device="cpu"
    )

    for step in ("run_plan", "run_train", "run_eval", "run_chat"):
        assert steps.kwargs[step]["precision"] == "fp32", step
        assert steps.kwargs[step]["device"] == "cpu", step


def test_the_evaluation_scores_both_splits(steps: _Recorder, tmp_path: Path) -> None:
    """A first run on a small corpus overfits, and one number cannot show that."""
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")

    assert steps.kwargs["run_eval"]["split"] == "both"


def test_the_reading_flags_reach_the_prepare_step(steps: _Recorder, tmp_path: Path) -> None:
    """A corpus that needs a flag to be read at all needs it on the step that reads it."""
    qs.run_quickstart(
        "corpus.csv",
        out=str(tmp_path / "qs"),
        prompt="Once",
        encoding="latin-1",
        csv_text_column="body",
    )

    assert steps.kwargs["run_prepare"]["encoding"] == "latin-1"
    assert steps.kwargs["run_prepare"]["csv_text_column"] == "body"


def test_tokens_is_omitted_rather_than_guessed(steps: _Recorder, tmp_path: Path) -> None:
    """``--tokens`` unset must not become a number here; ``chat``'s own default owns it."""
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")
    assert "max_new_tokens" not in steps.kwargs["run_chat"]

    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs2"), prompt="Once", tokens=32)
    assert steps.kwargs["run_chat"]["max_new_tokens"] == 32


def test_steps_reaches_train_and_the_resume_command(
    steps: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An overridden step count has to survive into the command printed after "n"."""
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once", steps=10)
    assert steps.kwargs["run_train"]["steps"] == 10

    monkeypatch.setattr(qs.console, "input", lambda *_: "n")
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs2"), steps=10)

    assert "--steps 10" in flat(capsys.readouterr().out)


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
def test_a_given_prompt_is_used_verbatim(steps: _Recorder, tmp_path: Path) -> None:
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once upon")

    assert steps.kwargs["run_chat"]["prompt"] == "Once upon"


def _prompt_for(text: str, tmp_path: Path) -> str:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8", newline="\n")
    return qs._prompt_from_corpus(str(corpus), qs._Reading())


def test_the_default_prompt_is_the_corpus_own_opening(tmp_path: Path) -> None:
    """Taken from the corpus, because a base model continues text rather than answering.

    A hard-coded English prompt would be the wrong beginning-of-something for a corpus in
    another language or another subject, and would make the first sample look broken when
    the model was fine.
    """
    assert _prompt_for("First Citizen: Before we proceed", tmp_path).startswith("First Citizen:")


def test_the_prompt_is_collapsed_to_one_quotable_line(tmp_path: Path) -> None:
    """It is echoed inside a printed command, so a newline or a quote would break it.

    A newline would split the copyable line in two; a double quote would end the shell
    argument early and leave the rest as separate words.
    """
    prompt = _prompt_for('  He said\n\t"hello"  there\n', tmp_path)

    assert prompt == "He said hello there"


def test_the_prompt_is_truncated(tmp_path: Path) -> None:
    assert len(_prompt_for("word " * 200, tmp_path)) == qs.PROMPT_CHARS


def test_an_empty_opening_falls_back(tmp_path: Path) -> None:
    """Unreachable in practice: `data prepare` refuses a corpus of only whitespace."""
    assert _prompt_for("   \n\n  ", tmp_path) == qs.FALLBACK_PROMPT


# --------------------------------------------------------------------------- #
# The printed equivalents
# --------------------------------------------------------------------------- #
def test_each_step_prints_the_command_it_stands_in_for(
    steps: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The claim in the banner -- "nothing here is a black box" -- is testable, so test it.

    This is what makes the command a shortcut *through* the CLI rather than a wizard
    around it: a user who wants to change one number copies the printed line and stops
    using this command.
    """
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once")

    shown = flat(capsys.readouterr().out)
    for command in (
        "trainai data prepare corpus.txt",
        "trainai plan",
        "trainai train",
        "trainai eval",
        "trainai chat",
    ):
        assert command in shown, command


def test_the_reading_flags_are_printed_only_when_they_are_not_defaults() -> None:
    """The printed line stays copyable only while it stays short."""
    assert qs._Reading().flags() == ""
    assert qs._Reading(csv_text_column="body").flags() == " --csv-text-column body"
    assert "--encoding" in qs._Reading(encoding="latin-1").flags()


def test_the_final_report_names_the_held_out_loss(
    steps: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    qs.run_quickstart("corpus.txt", out=str(tmp_path / "qs"), prompt="Once", assume_yes=True)

    shown = flat(capsys.readouterr().out)
    assert "5.4657" in shown
    assert "236.44" in shown
    assert "trainai export" in shown


# --------------------------------------------------------------------------- #
# The command surface
# --------------------------------------------------------------------------- #
def test_the_help_says_there_is_no_json_and_there_is_not() -> None:
    """Five reports concatenated is not a document, so there is deliberately no --json.

    Anything scripted should call the individual commands, each of which emits its own
    object. The absence is documented rather than left to be discovered, so both halves
    are checked: the sentence, and the flag actually being rejected.

    The sentence is read from the docstring Typer renders rather than from the rendered
    help, which CI proved is not a string this can assert on: the help renderer styles
    its output whether or not anything is a terminal, so the phrase arrives with escape
    sequences between its words, wrapped at whatever width the job happened to use.
    Rich's own markup is stripped for the same reason -- the emphasis around the flag is
    presentation, and asserting through it would pin the styling rather than the claim.
    """
    documented = " ".join(re.sub(r"\[/?[a-z ]+\]", "", main.quickstart.__doc__ or "").split())
    assert "no --json" in documented

    refused = runner.invoke(app, ["quickstart", "corpus.txt", "--json"])
    assert refused.exit_code != 0
