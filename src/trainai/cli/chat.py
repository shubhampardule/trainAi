"""``trainai chat`` -- an interactive playground for a finished run.

The name is what people look for, and it is also the command's main hazard. What a
run produces is a **base** language model: it continues text. Nothing in its training
data was a question followed by an answer, so typing "what is the capital of France?"
gets a continuation of that sentence -- more questions, probably -- and a user who
expected an assistant reads a working model as a broken one. So the banner says what
this is before the first prompt, and says it in one short paragraph rather than a
footnote nobody reads.

Three other things shape the loop:

**Each line is a fresh completion, and ``/more`` is how you continue.** Accumulating a
transcript would imply a dialogue format the model has never seen. Continuing its own
last output, on the other hand, is exactly what a base model does well -- so that is
available, explicitly, as its own command.

**A prompt longer than the model's context is truncated, and it says so.** The sampler
keeps the last ``seq_len`` tokens, which is the right choice and an invisible one; a
user pasting three pages into a 256-token model should not have to wonder why the
output ignores the beginning.

**Speed is measured, not estimated.** Every completion reports the tokens it produced
and the rate it produced them at, because that number is the one thing a user cannot
look up -- it belongs to their machine. The count is tokens, not printed fragments: a
token that completes half a character prints nothing and still counts.

Ctrl-C stops a generation and returns to the prompt. It exits only at an empty prompt,
which is the behaviour of every REPL and the one people try first.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from typing import Any

from trainai.console import (
    DASH,
    console,
    emit_json,
    fmt_count,
    fmt_int,
    print_bullets,
    print_kv,
    printable,
    rule,
)
from trainai.errors import UsageError
from trainai.infer import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_WHICH,
    InferenceSession,
)

__all__ = ["run_chat"]

#: A model this small produces fragments rather than sentences. Below this parameter
#: count the banner says so, because "it does not work" is the wrong conclusion to
#: draw from a 2M-parameter model and the right one to draw from a bug.
SMALL_MODEL_PARAMETERS = 20_000_000


@dataclass(frozen=True)
class Sampling:
    """The knobs a ``/`` command can change between completions."""

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    top_p: float | None = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    seed: int | None = None
    stop_at_eot: bool = True

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("Tokens", f"{fmt_int(self.max_new_tokens)} at most"),
            ("Temperature", f"{self.temperature:g}"),
            ("Top-p", "off" if self.top_p is None else f"{self.top_p:g}"),
            ("Top-k", "off" if self.top_k is None else str(self.top_k)),
            ("Penalty", f"{self.repetition_penalty:g}"),
            ("Seed", "unset (a new sample each time)" if self.seed is None else str(self.seed)),
            ("End-of-text", "stops generation" if self.stop_at_eot else "ignored"),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "stop_at_eot": self.stop_at_eot,
        }


@dataclass(frozen=True)
class Completion:
    """One generation, with what it cost. ``seconds`` is wall clock on this machine."""

    prompt: str
    text: str
    tokens: int
    seconds: float
    truncated_from: int | None = None
    interrupted: bool = False

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.text,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
            "prompt_truncated_from": self.truncated_from,
            "interrupted": self.interrupted,
        }


def run_chat(
    target: str,
    *,
    prompt: str | None = None,
    which: str = DEFAULT_WHICH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int | None = DEFAULT_TOP_K,
    top_p: float | None = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    seed: int | None = None,
    ignore_eot: bool = False,
    device: str = "auto",
    precision: str = "auto",
    tokenizer: str | None = None,
    json_output: bool = False,
) -> None:
    """Generate from a run: once from ``prompt``, or interactively.

    With no ``prompt`` and a piped stdin, the pipe is the prompt -- so
    ``echo "Once upon" | trainai chat runs/mine`` works and can be scripted. With no
    ``prompt`` and a terminal, this is the interactive playground.
    """
    sampling = Sampling(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_at_eot=not ignore_eot,
    )
    _check_sampling(sampling)

    once = prompt if prompt is not None else _piped_prompt()
    if once is None and json_output:
        raise UsageError(
            "--json needs a prompt to answer with.",
            hint='Pass --prompt "some text", or pipe it in. The interactive playground '
            "prints as it generates, which is not machine-readable.",
            details={"json": True},
        )

    session = InferenceSession.open(
        target,
        which=which,
        device=device,
        precision=precision,
        tokenizer=tokenizer,
    )

    if once is not None:
        _one_shot(session, once, sampling, json_output=json_output)
        return
    _interactive(session, sampling)


# --------------------------------------------------------------------------- #
# The two modes
# --------------------------------------------------------------------------- #
def _one_shot(
    session: InferenceSession, prompt: str, sampling: Sampling, *, json_output: bool
) -> None:
    """Generate once. Under ``--json`` the only thing printed is the JSON."""
    if json_output:
        completion = _generate(session, prompt, sampling, echo=False)
        emit_json(
            {
                "version": 1,
                **completion.to_dict(),
                "sampling": sampling.to_dict(),
                "model": {
                    "parameters": session.model.parameter_count(),
                    "vocab_size": session.tokenizer.vocab_size,
                    "context": session.model_config.seq_len,
                },
                "checkpoint": str(session.layout.checkpoint_path),
                "which": session.layout.which,
                "step": session.step,
                "device": str(session.device),
                "precision": session.precision_note,
                "notes": [
                    "This is a base language model: it continues text rather than "
                    "answering questions."
                ],
            }
        )
        return

    completion = _generate(session, prompt, sampling, echo=True)
    console.print()
    console.print(f"[dim]{_cost(completion)}[/]")


def _interactive(session: InferenceSession, sampling: Sampling) -> None:
    """The playground. Returns when the user exits or stdin ends.

    Ctrl-D ends it, so does ``/exit``, and so does Ctrl-C twice in a row. A single
    Ctrl-C at the prompt only discards the half-typed line, because both things this
    command tells the user say it does not leave: the banner offers "``/exit`` or Ctrl-D
    to leave" without naming Ctrl-C, and ``/help`` promises "Ctrl-C stops a generation
    without leaving". It used to leave anyway, which cost a whole session -- the model
    off the device (about 7s to put back, on this machine), every ``/temp`` and ``/seed``
    that had been set, and the ``/more`` history -- for a keystroke that in Python's own
    REPL just clears the line. Two in a row still leaves, announced in between, so
    Ctrl-C remains a way out for anyone who reaches for it first and bounds the loop if
    a broken terminal raises on every read.
    """
    _banner(session)
    last = ""
    interrupts = 0

    while True:
        try:
            line = console.input("\n[bold cyan]>[/] ")
        except EOFError:
            console.print("\n[dim]Bye.[/]")
            return
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts > 1:
                console.print("\n[dim]Bye.[/]")
                return
            console.print("\n[dim]Interrupted. Ctrl-C again, Ctrl-D or /exit to leave.[/]")
            continue
        # Any line that arrives makes the next Ctrl-C a first one again, so two presses
        # with real work between them do not add up to an exit.
        interrupts = 0

        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            command = _handle_command(text, sampling, session, last)
            if command is None:
                return
            sampling, prompt = command
            if prompt is None:
                continue
        else:
            prompt = text

        console.print()
        completion = _generate(session, prompt, sampling, echo=True)
        console.print()
        console.print(f"[dim]{_cost(completion)}[/]")
        last = prompt + completion.text


def _handle_command(
    line: str, sampling: Sampling, session: InferenceSession, last: str
) -> tuple[Sampling, str | None] | None:
    """Apply a ``/`` command.

    Returns ``(sampling, prompt)`` -- a prompt when the command asks for generation,
    ``None`` for the prompt when it does not -- or ``None`` entirely to exit.
    """
    parts = line.split()
    name, argument = parts[0].lower(), (parts[1] if len(parts) > 1 else None)

    if name in ("/exit", "/quit", "/q"):
        return None
    if name in ("/help", "/?"):
        _help()
        return sampling, None
    if name in ("/settings", "/set"):
        print_kv("Sampling", sampling.rows())
        return sampling, None
    if name == "/more":
        if not last:
            console.print("[yellow]Nothing to continue yet.[/] Give it something first.")
            return sampling, None
        return sampling, last

    try:
        updated = _apply_setting(name, argument, sampling)
        if updated is None:
            console.print(f"[yellow]Unknown command {name}.[/] Type /help for the list.")
            return sampling, None
        _check_sampling(updated)
    except UsageError as error:
        # Printed rather than raised: a typo in an interactive session should cost the
        # user a line, not the loaded model.
        console.print(f"[yellow]{error}[/]  [dim]{error.hint}[/]")
        return sampling, None
    print_kv("Sampling", updated.rows())
    return updated, None


def _apply_setting(name: str, argument: str | None, sampling: Sampling) -> Sampling | None:
    """A setting command applied, or ``None`` if ``name`` is not one.

    Raises:
        UsageError: The command is a setting but its value is missing or not a number.
    """
    off = argument is not None and argument.lower() in ("off", "none", "unset")
    if name in ("/temp", "/temperature"):
        return replace(sampling, temperature=_number(name, argument, float))
    if name in ("/top-p", "/topp", "/p"):
        return replace(sampling, top_p=None if off else _number(name, argument, float))
    if name in ("/top-k", "/topk", "/k"):
        return replace(sampling, top_k=None if off else _number(name, argument, int))
    if name in ("/penalty", "/repeat"):
        return replace(sampling, repetition_penalty=_number(name, argument, float))
    if name in ("/tokens", "/max"):
        return replace(sampling, max_new_tokens=_number(name, argument, int))
    if name == "/seed":
        return replace(sampling, seed=None if off else _number(name, argument, int))
    if name == "/eot":
        return replace(sampling, stop_at_eot=not off)
    return None


def _number(name: str, argument: str | None, cast: Any) -> Any:
    if argument is None:
        raise UsageError(
            f"{name} needs a value.",
            hint="For example: /temp 0.7. Type /settings to see the current ones.",
            details={"command": name},
        )
    try:
        return cast(argument)
    except ValueError:
        raise UsageError(
            f"{name} needs a number, got {argument!r}.",
            hint="For example: /temp 0.7, or /top-k off to disable one.",
            details={"command": name, "value": argument},
        ) from None


# --------------------------------------------------------------------------- #
# Generating
# --------------------------------------------------------------------------- #
def _generate(
    session: InferenceSession, prompt: str, sampling: Sampling, *, echo: bool
) -> Completion:
    """Stream one completion, printing as it arrives when ``echo``.

    Ctrl-C mid-generation keeps what was produced: the tokens were really generated and
    the timing over them is real, so the completion is returned rather than discarded.
    """
    truncated = _truncation(session, prompt)
    if truncated is not None and echo:
        console.print(
            f"[yellow]The prompt is {fmt_int(truncated)} tokens and this model's context "
            f"is {fmt_int(session.model_config.seq_len)}.[/] "
            "[dim]Only the last part of it was used.[/]"
        )

    pieces: list[str] = []
    tokens = 0
    interrupted = False
    started = time.perf_counter()
    try:
        for piece in session.stream_pieces(
            prompt,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
            stop_at_eot=sampling.stop_at_eot,
        ):
            # piece.tokens is cumulative and counts tokens whose text is still being
            # held back, which counting pieces would not -- see infer.StreamPiece.
            tokens = piece.tokens
            if not piece.text:
                continue
            pieces.append(piece.text)
            if echo:
                # markup=False: the model's own text is not Rich markup, and a
                # generated "[" would otherwise swallow the rest of the line.
                # printable(): a byte-level tokenizer can produce U+FFFD, which a
                # cp1252 console cannot encode -- see trainai.console.printable.
                console.print(
                    printable(piece.text), end="", markup=False, highlight=False, soft_wrap=True
                )
    except KeyboardInterrupt:
        interrupted = True
        if echo:
            console.print("\n[dim]Stopped.[/]")
    elapsed = time.perf_counter() - started

    return Completion(
        prompt=prompt,
        text="".join(pieces),
        tokens=tokens,
        seconds=elapsed,
        truncated_from=truncated,
        interrupted=interrupted,
    )


def _truncation(session: InferenceSession, prompt: str) -> int | None:
    """The prompt's token count, when it exceeds the context. ``None`` when it fits.

    The sampler keeps the last ``seq_len`` tokens, which is the right thing to do and
    leaves no trace. Counting here costs one encode -- microseconds against a forward
    pass -- and turns a silent truncation into a sentence.
    """
    count = len(session.tokenizer.encode(prompt))
    return count if count > session.model_config.seq_len else None


def _cost(completion: Completion) -> str:
    if completion.tokens == 0:
        return "no tokens produced"
    return (
        f"{fmt_int(completion.tokens)} tokens in {completion.seconds:.1f}s "
        f"{DASH} {completion.tokens_per_second:.1f} tokens/s"
    )


# --------------------------------------------------------------------------- #
# What the user is told before they type
# --------------------------------------------------------------------------- #
def _banner(session: InferenceSession) -> None:
    layout = session.layout
    parameters = session.model.parameter_count()
    rule("Playground")
    loaded = [
        (
            "Checkpoint",
            f"[bold]{layout.checkpoint_path.name}[/]  [dim]({layout.which}, step "
            f"{fmt_int(session.step)})[/]",
        ),
        (
            "Model",
            f"[bold]{fmt_count(parameters)}[/] parameters  {DASH}  "
            f"{session.model_config.describe()}",
        ),
        ("Device", f"{session.device}  {DASH}  {session.precision_note}"),
    ]
    if layout.note:
        loaded.append(("Note", f"[yellow]{layout.note}[/]"))
    print_kv("Loaded", loaded)

    notes = [
        "This is a [bold]base[/] language model. It continues text; it does not answer "
        "questions or follow instructions, because nothing in its training data was a "
        "dialogue. Give it the beginning of something.",
        "Each line is a fresh completion. [bold]/more[/] continues the last one.",
    ]
    if parameters < SMALL_MODEL_PARAMETERS:
        notes.append(
            f"At {fmt_count(parameters)} parameters this model produces fragments that "
            "read like its training data rather than sentences that mean anything. That "
            "is the size, not a fault."
        )
    if session.best_val_loss is not None:
        notes.append(
            f"Its best validation loss was {session.best_val_loss:.4f} at step "
            f"{fmt_int(session.best_val_step or 0)}."
        )
    notes.append("[bold]/help[/] for commands, [bold]/exit[/] or Ctrl-D to leave.")
    print_bullets("Worth knowing", notes)


def _help() -> None:
    print_bullets(
        "Commands",
        [
            "[bold]/more[/] -- continue the last completion instead of starting fresh",
            "[bold]/temp[/] N -- sampling temperature; 0 is greedy, 0.8 is the default",
            "[bold]/top-p[/] N | off -- nucleus sampling threshold",
            "[bold]/top-k[/] N | off -- keep only the N most likely tokens",
            "[bold]/penalty[/] N -- repetition penalty; 1.0 is off",
            "[bold]/tokens[/] N -- how many tokens to generate at most",
            "[bold]/seed[/] N | off -- fix the sample, or let it vary",
            "[bold]/eot[/] on | off -- stop at the end-of-text token, or keep going",
            "[bold]/settings[/] -- show all of the above",
            "[bold]/exit[/] -- leave, and so does Ctrl-D",
            "Ctrl-C -- stop a generation without leaving; at the prompt, twice to leave",
        ],
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check_sampling(sampling: Sampling) -> None:
    """Refuse impossible sampling settings with the range that is allowed.

    The model's own sampler checks temperature, but it checks it at the first forward
    pass -- after a checkpoint has been loaded onto the device. Everything here is
    knowable from the flags alone, so it is answered from the flags alone.
    """
    if sampling.max_new_tokens < 1:
        raise UsageError(
            f"--tokens must be at least 1, got {sampling.max_new_tokens}.",
            hint="This is how many tokens to generate; 200 is the default.",
            details={"max_new_tokens": sampling.max_new_tokens},
        )
    if sampling.temperature < 0:
        raise UsageError(
            f"--temperature must not be negative, got {sampling.temperature}.",
            hint="0 is greedy decoding, 0.8 is the usual sample, above 1.2 wanders.",
            details={"temperature": sampling.temperature},
        )
    if sampling.top_p is not None and not 0 < sampling.top_p <= 1:
        raise UsageError(
            f"--top-p must be above 0 and at most 1, got {sampling.top_p}.",
            hint="0.95 keeps the smallest set of tokens holding 95% of the probability.",
            details={"top_p": sampling.top_p},
        )
    if sampling.top_k is not None and sampling.top_k < 1:
        raise UsageError(
            f"--top-k must be at least 1, got {sampling.top_k}.",
            hint="Leave it unset to disable it, or use --top-k 40.",
            details={"top_k": sampling.top_k},
        )
    if sampling.repetition_penalty <= 0:
        raise UsageError(
            f"--penalty must be above 0, got {sampling.repetition_penalty}.",
            hint="1.0 disables it; 1.1 is the default; above 1.5 breaks grammar.",
            details={"repetition_penalty": sampling.repetition_penalty},
        )


def _piped_prompt() -> str | None:
    """Everything on stdin when it is a pipe, or ``None`` when it is a terminal.

    Makes the command scriptable without a second flag:
    ``echo "Once upon" | trainai chat runs/mine``. A terminal falls through to the
    interactive loop, which is what someone typing the command by hand wants.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return None
    try:
        piped = sys.stdin.read()
    except (OSError, ValueError):  # pragma: no cover - closed or detached stdin
        return None
    return piped if piped.strip() else None
