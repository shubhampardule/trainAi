"""``trainai chat`` -- an interactive playground for a finished run.

The name is what people look for, and it is also the command's main hazard. What a
run produces is a **base** language model: it continues text. On a prose corpus --
which is what this project ships and documents -- nothing in the training data was a
question followed by an answer, so typing "what is the capital of France?" gets a
continuation of that sentence, more questions probably, and a user who expected an
assistant reads a working model as a broken one. So the banner says what this is
before the first prompt, and says it in one short paragraph rather than a footnote
nobody reads.

What the banner does **not** do is claim to know the corpus. It used to say "nothing
in its training data was a dialogue" as a flat fact, which stopped being true the day
this repo started shipping ``chat.jsonl`` -- 651,448 ``User:``/``Assistant:``
examples. A checkpoint records its dataset's ``content_hash``, not its contents, so
this command cannot tell prose from dialogue and should not pretend to. It states the
dependency instead, and leaves the corpus to the person who chose it.

**Whether a line is wrapped in the chat template is the checkpoint's call.** A run
whose dataset was rendered as conversations records the template it was rendered in, so
this command can put ``User:``/``Assistant:`` around what you type and cut the reply
where the model starts writing somebody else's turn. A run trained on prose records
nothing and the text goes to the model exactly as typed. Neither is sniffed from the
text -- the same refusal to guess as everywhere else in the tool -- and the two flags
that override it exist for what a checkpoint cannot know: a corpus flattened into
``User:``/``Assistant:`` text by hand carries no template even though the model learned
one, and a chat model is still a base model worth probing raw.

Three other things shape the loop:

**In chat mode the conversation is remembered; in raw mode nothing is.** The corpus this
template renders is conversations -- the typed corpus this repo ships is 4,000 of 4,000
``user``/``assistant``/``user``/``assistant`` -- so a follow-up question with the exchange
before it in front of it is the layout the model was trained on, and one without it is
not. ``/new`` forgets. Raw mode accumulates nothing, because a transcript would imply a
dialogue format a prose-trained model has never seen; ``/more`` continues the last output
there, which is exactly what a base model does well, and in chat mode continues the reply.

**A prompt longer than the model's context is truncated, and it says so.** The sampler
keeps the last ``seq_len`` tokens, which is the right choice and an invisible one; a
user pasting three pages into a 256-token model should not have to wonder why the
output ignores the beginning. A *conversation* too long for the context loses whole
turns off its front instead, so that what is left is still a conversation.

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
from collections.abc import Callable, Sequence
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
from trainai.data.chat import (
    REPLY_ROLE,
    ROLE_LABELS,
    TEMPLATE_VERSION,
    TURN_BOUNDARIES,
    ChatFormatError,
    render_conversation,
    render_prompt,
    reply_content,
)
from trainai.errors import UsageError
from trainai.infer import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_WHICH,
    Finish,
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
class Mode:
    """Whether a typed line is wrapped in the chat template, and why it was.

    ``why`` is carried around rather than recomputed for display because the mode is
    usually a *default*, and a default the user cannot see is one they cannot correct.
    It reaches the banner, ``/settings`` and ``--json``, so "the model ignored my
    question" and "the model was never asked my question" stop looking alike.
    """

    chat: bool
    why: str

    @property
    def name(self) -> str:
        return "chat" if self.chat else "raw"

    @property
    def stops(self) -> tuple[str, ...]:
        """Where generation stops early. Nothing in raw mode: prose has no turns to end."""
        return TURN_BOUNDARIES if self.chat else ()

    def prompt_for(self, text: str) -> str:
        """What the model is actually given for a *single* line, which in chat mode is
        not what was typed.

        This is the one-shot path -- ``--prompt`` and a pipe, where there is no
        conversation to be part of. The interactive loop renders the history with it
        through :func:`_ask` instead, and both end up in
        :func:`~trainai.data.chat.render_prompt`.

        Rendered by :func:`~trainai.data.chat.render_prompt` rather than by pasting a
        label on here, for the reason that function exists: a second copy of the layout
        is a second thing to keep in agreement with the shards, and its disagreements
        are invisible.
        """
        if not self.chat:
            return text
        try:
            return render_prompt([{"role": "user", "content": text}])
        except ChatFormatError as error:
            # ChatFormatError is a ValueError, not a TrainAIError, so left alone it
            # would reach the user as a traceback rather than as a refusal.
            raise UsageError(
                f"In chat mode the prompt is a message, and an empty one renders as a "
                f"bare {ROLE_LABELS['user']}: label -- there would be nothing to answer.",
                hint='Ask something, as in --prompt "Why do the tides turn?", or pass '
                "--raw to send the text to the model exactly as typed.",
                details={"mode": "chat", "problem": error.problem, **error.fields},
            ) from error

    def rows(self) -> list[tuple[str, str]]:
        wrapping = (
            f"wraps what you type in [bold]{ROLE_LABELS['user']}:[/] / "
            f"[bold]{ROLE_LABELS[REPLY_ROLE]}:[/]"
            if self.chat
            else "the model sees exactly what you type"
        )
        return [("Mode", f"[bold]{self.name}[/] {DASH} {wrapping}  [dim]({self.why})[/]")]

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.name, "reason": self.why, "stop": list(self.stops)}


@dataclass(frozen=True)
class Prompt:
    """One thing to generate from: what the user gave, and what the model is given.

    The two differ in chat mode, and both are worth keeping: ``text`` is what a script
    passed in, ``model`` is what was encoded, truncated against the context and streamed
    from. ``/more`` is what makes them two fields rather than one and a rule -- a
    continuation is already model text, and wrapping it in a fresh label would ask the
    model to answer its own reply.

    ``dropped`` is how many messages came off the front of the conversation to make it fit
    the context, which the loop says out loud: see :func:`_fit`.
    """

    text: str
    model: str
    dropped: int = 0


#: One message of the conversation the playground is keeping, in the shape the renderer
#: takes. Plain dicts rather than a dataclass because they are handed straight to
#: :mod:`trainai.data.chat`, which validates them and is the only thing that reads them.
Turn = dict[str, str]


@dataclass(frozen=True)
class State:
    """Everything one line of the playground can change, and the next line can see.

    One value rather than four because each command changes a different part of it --
    ``/temp`` the sampling, ``/raw`` the mode, ``/new`` the conversation -- and a handler
    returning four unlabelled values is a handler whose callers get the order wrong.

    ``history`` is the conversation in chat mode, and is what makes a second question a
    follow-up rather than a fresh one. ``last`` is the raw-mode equivalent and cannot be
    the same field: raw mode has no turns to keep, only the text so far.
    """

    mode: Mode
    sampling: Sampling
    history: tuple[Turn, ...] = ()
    last: str = ""


@dataclass(frozen=True)
class Completion:
    """One generation, with what it cost. ``seconds`` is wall clock on this machine.

    ``finish`` is why generation ended, and ``None`` when nothing ended it -- a Ctrl-C
    mid-stream never reaches the answer, and reporting a reason for an abandoned
    generation would be inventing one.

    ``truncated_from`` and ``slides_after`` are the two ways a prompt can outgrow the
    context, and they are mutually exclusive on purpose: the first is a prompt that did
    not fit and was cut before generation, the second a prompt that fitted and then lost
    its front *during* generation. Both are ``None`` in the ordinary case.
    """

    prompt: str
    text: str
    tokens: int
    seconds: float
    model_prompt: str = ""
    truncated_from: int | None = None
    interrupted: bool = False
    finish: Finish | None = None
    slides_after: int | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model_prompt": self.model_prompt or self.prompt,
            "completion": self.text,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
            "prompt_truncated_from": self.truncated_from,
            "window_slides_after": self.slides_after,
            "interrupted": self.interrupted,
            # Nested rather than flattened: "stop" at the top level is already the list
            # of stop strings that were *configured*, and one key cannot be both.
            "finish": self.finish.to_dict() if self.finish else None,
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
    chat_format: bool = False,
    raw_format: bool = False,
    json_output: bool = False,
) -> None:
    """Generate from a run: once from ``prompt``, or interactively.

    With no ``prompt`` and a piped stdin, the pipe is the prompt -- so
    ``echo "Once upon" | trainai chat runs/mine`` works and can be scripted. With no
    ``prompt`` and a terminal, this is the interactive playground.

    ``chat_format`` and ``raw_format`` override what the checkpoint says about the chat
    template. Neither is the normal case: the default is what the run was trained as.
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
    _check_format_flags(chat_format=chat_format, raw_format=raw_format)

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
    mode = _resolve_mode(session, chat_format=chat_format, raw_format=raw_format)

    if once is not None:
        _one_shot(session, mode, once, sampling, json_output=json_output)
        return
    _interactive(session, mode, sampling)


def _resolve_mode(session: InferenceSession, *, chat_format: bool, raw_format: bool) -> Mode:
    """Whether to wrap what the user types in the chat template, and the reason.

    The checkpoint decides unless told otherwise, which is the only default that can be
    right: a dataset prepared with ``--jsonl-messages-field`` records the template its
    shards were rendered in, and one prepared from prose records nothing. Sniffing the
    user's text for a question mark, or the corpus for ``User:``, would be the guess this
    tool refuses everywhere else.

    A recorded template from a *different* version is refused rather than applied. That
    field exists to be bumped when the rendering changes, so a mismatch means this build
    would wrap the prompt in a layout the model was not trained on -- which produces
    worse replies and no error, the failure the template module was written to prevent.
    """
    template = session.chat_template
    if raw_format:
        return Mode(chat=False, why="--raw")
    if chat_format:
        if not template:
            return Mode(chat=True, why="--chat, though this checkpoint records no template")
        recorded = template.get("version")
        if recorded != TEMPLATE_VERSION:
            return Mode(chat=True, why=f"--chat, overriding the recorded version {recorded}")
        return Mode(chat=True, why="--chat")
    if not template:
        return Mode(chat=False, why="this checkpoint records no chat template")
    recorded = template.get("version")
    if recorded != TEMPLATE_VERSION:
        raise UsageError(
            f"This run was trained with chat template version {recorded}, and this "
            f"TrainAI renders version {TEMPLATE_VERSION}.",
            hint="Prompting a model in a layout it was not trained in makes it answer "
            "worse without failing, so this is refused rather than guessed. Use --chat "
            "to prompt it in this version's layout anyway, or --raw to send your text "
            "unchanged.",
            details={
                "recorded_template": template,
                "supported_version": TEMPLATE_VERSION,
                "checkpoint": str(session.layout.checkpoint_path),
            },
        )
    return Mode(chat=True, why="this run's dataset was rendered as conversations")


# --------------------------------------------------------------------------- #
# The conversation
# --------------------------------------------------------------------------- #
def _ask(session: InferenceSession, history: Sequence[Turn], text: str) -> Prompt:
    """A new question with the conversation so far in front of it.

    The corpus this template renders is conversations, not question/answer pairs -- the
    typed corpus in ``data/chat-typed`` is 4,000 of 4,000 ``user/assistant/user/assistant``
    -- so a follow-up with the exchange before it is the layout the model was trained on,
    and a follow-up without it is the one that is not.
    """
    messages = [*history, {"role": "user", "content": text}]
    model, dropped = _fit(session, messages, render_prompt)
    return Prompt(text=text, model=model, dropped=dropped)


def _continue(session: InferenceSession, history: Sequence[Turn]) -> Prompt:
    """``/more``: the same conversation, ending in the reply the model already began.

    Rendered with :func:`~trainai.data.chat.render_conversation` rather than
    :func:`~trainai.data.chat.render_prompt`, which is the difference between the two
    functions: one ends at the label, this one ends in the reply itself.
    """
    model, dropped = _fit(session, history, lambda turns: render_conversation(turns).text)
    return Prompt(text=model, model=model, dropped=dropped)


def _fit(
    session: InferenceSession,
    messages: Sequence[Turn],
    render: Callable[[list[Turn]], str],
) -> tuple[str, int]:
    """Render ``messages``, dropping whole ones off the front until they fit the context.

    The sampler truncates by *tokens*, keeping the last ``seq_len``, which for a
    conversation cuts wherever the count lands: mid-word, inside a message, leaving a
    prompt whose first label is half a label. That is a layout no corpus contains.
    Dropping whole messages instead means every prompt the model sees is one it was
    trained on, and the difference is invisible without this: both produce a reply.

    They come off in pairs where they have to. Dropping one can leave the conversation
    beginning with a reply, and a corpus conversation begins with a question, so the drop
    continues until the oldest surviving message starts a turn again.

    The last message is never dropped: a question the user just typed is the one thing
    that must not silently disappear, so a single message too long for the context is
    handed on as it is and :func:`_generate` reports the truncation it will get.

    Trimming is per prompt. The conversation itself keeps everything, so a short question
    after a long one has the whole thing behind it again rather than what the long one had
    room for -- what fits is a property of the prompt, not a decision about the past.

    Nothing is reserved for the answer, and that is the measured answer rather than a gap.
    Reserving ``r`` tokens means fitting the prompt to ``seq_len - r``; the sampler's
    window slides anyway once the reply outgrows ``r``, so a reserve does not buy the
    conversation room, it spends it sooner. Written out: with a prompt of ``P`` tokens in
    a context of ``C``, the window holds ``P`` tokens of conversation until generated
    token ``C - P`` and ``C - i`` from then on, and a larger budget never keeps fewer
    messages, so ``P`` is largest at ``r = 0``. The no-reserve window therefore holds at
    least as much of the conversation at *every* position. Counted over 216 held-out
    sessions against ``--tokens 200`` on a 128-token context: 0 of 172,800 positions where
    any reserve showed the model more.

    What a reserve does buy is the front alignment this function exists for, for as long
    as it lasts. Measured on ``data/chat-typed``: the slide put the window's front
    mid-message for 11.0% of generated positions, a reserve of 16 or more for none of
    them, and teacher-forced loss on the held-out replies moved from 0.0058 to 0.0055
    nats/token -- perplexity 1.006 against 1.005. That corpus is memorised, so the
    difference is noise, and it is the only chat corpus here; the honest summary is that
    the alignment is worth an amount this repository cannot measure, against a context
    cost that is arithmetic. Aligning the *slide* to message boundaries would buy both,
    and needs a corpus where old turns are load-bearing before it is worth the layering.

    The slide is reported rather than silent: :func:`_slides_after` says, once a reply is
    written, how many of its tokens had the whole prompt in the window, because after ``C``
    of them none of the conversation is left -- a reply that runs to the default
    ``--tokens 200`` on a 128-token context spends its last 72 tokens continuing nothing
    but itself. It is reported afterwards rather than predicted from ``--tokens``, since
    the budget is an upper bound most replies never reach.

    Returns the rendered text and how many messages were dropped.
    """
    kept = list(messages)
    dropped = 0
    while True:
        rendered = render(kept)
        if len(session.tokenizer.encode(rendered)) <= session.model_config.seq_len:
            return rendered, dropped
        if len(kept) <= 1:
            return rendered, dropped
        kept, dropped = kept[1:], dropped + 1
        while len(kept) > 1 and kept[0]["role"] == REPLY_ROLE:
            kept, dropped = kept[1:], dropped + 1


def _remember(history: tuple[Turn, ...], question: str, reply: str) -> tuple[Turn, ...]:
    """The conversation with one completed exchange added, or unchanged if there was none.

    What the model produced is turned back into a message by
    :func:`~trainai.data.chat.reply_content`, which is where the knowledge of what the
    template put around it lives -- the gap after the label and the separator generation
    stopped on both belong to the layout, and keeping either would render a second one.

    A reply with no text in it is not remembered, and neither is the question that got it.
    ``trainai.data.chat`` refuses an empty message -- it renders as a bare label, which is
    a thing to train a model out of, not into -- so recording one would raise on the
    *next* prompt, several lines from the cause. A model that said nothing has said
    nothing to remember.
    """
    text = reply_content(reply)
    if not text.strip():
        return history
    return (*history, {"role": "user", "content": question}, {"role": REPLY_ROLE, "content": text})


def _extend(history: tuple[Turn, ...], more: str) -> tuple[Turn, ...]:
    """The conversation with ``more`` appended to the reply it already ends in.

    ``/more`` continues a reply rather than starting a turn, so it grows the last one, and
    ``continuing=True`` is what says the generation began inside the content: there is no
    label gap in front of this text to take off. An empty continuation leaves the
    conversation exactly as it was.
    """
    text = reply_content(more, continuing=True)
    if not history or not text.strip():
        return history
    last = history[-1]
    grown = {**last, "content": f"{last['content']}{text}"}
    return (*history[:-1], grown)


# --------------------------------------------------------------------------- #
# The two modes
# --------------------------------------------------------------------------- #
def _one_shot(
    session: InferenceSession,
    mode: Mode,
    text: str,
    sampling: Sampling,
    *,
    json_output: bool,
) -> None:
    """Generate once. Under ``--json`` the only thing printed is the JSON."""
    prompt = Prompt(text=text, model=mode.prompt_for(text))
    if json_output:
        completion = _generate(session, mode, prompt, sampling, echo=False)
        emit_json(
            {
                "version": 1,
                **completion.to_dict(),
                **mode.to_dict(),
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
                "chat_template": session.chat_template,
                "notes": _notes(mode),
            }
        )
        return

    completion = _generate(session, mode, prompt, sampling, echo=True)
    console.print()
    ending = _ending(completion, sampling, interactive=False)
    if ending:
        console.print(f"[yellow]{ending}[/]")
    console.print(f"[dim]{_cost(completion)}[/]")


def _notes(mode: Mode) -> list[str]:
    """What a machine reader should know about the text it just got.

    Says less than the banner's :func:`_worth_knowing` and says it without markup: a
    machine reader needs to know what ``model_prompt`` is and why a completion can end in
    a newline, not to be advised how to prompt.
    """
    if mode.chat:
        return [
            f"The prompt was wrapped in this run's chat template: model_prompt is what "
            f"the model was given, and it ends in {ROLE_LABELS[REPLY_ROLE]}:.",
            "Generation stops where the model starts writing somebody else's turn, so "
            "the completion may end in the newline before that label.",
        ]
    return ["This is a base language model: it continues text rather than answering questions."]


def _interactive(session: InferenceSession, mode: Mode, sampling: Sampling) -> None:
    """The playground. Returns when the user exits or stdin ends.

    Ctrl-D ends it, so does ``/exit``, and so does Ctrl-C twice in a row. A single
    Ctrl-C at the prompt only discards the half-typed line, because both things this
    command tells the user say it does not leave: the banner offers "``/exit`` or Ctrl-D
    to leave" without naming Ctrl-C, and ``/help`` promises "Ctrl-C stops a generation
    without leaving". It used to leave anyway, which cost a whole session -- the model
    off the device (about 7s to put back, on this machine), every ``/temp`` and ``/seed``
    that had been set, and the conversation -- for a keystroke that in Python's own
    REPL just clears the line. Two in a row still leaves, announced in between, so
    Ctrl-C remains a way out for anyone who reaches for it first and bounds the loop if
    a broken terminal raises on every read.

    The loop is a fold over :class:`State`: read a line, produce a prompt, generate,
    record. ``asked`` is what distinguishes a typed question from a ``/`` command that
    generates, which is the difference between adding a turn and growing the last one.
    """
    _banner(session, mode)
    state = State(mode=mode, sampling=sampling)
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
        asked = not text.startswith("/")
        if asked:
            # No refusal to catch here: _ask only rejects an empty message, and the blank
            # line that would produce one was skipped above.
            prompt = (
                _ask(session, state.history, text)
                if state.mode.chat
                else Prompt(text=text, model=text)
            )
        else:
            command = _handle_command(text, state, session)
            if command is None:
                return
            state, prompt = command
            if prompt is None:
                continue

        if prompt.dropped:
            left_out = (
                "Its oldest message was left out"
                if prompt.dropped == 1
                else f"Its oldest {fmt_int(prompt.dropped)} messages were left out"
            )
            console.print(
                f"[yellow]The conversation no longer fits this model's context of "
                f"{fmt_int(session.model_config.seq_len)} tokens.[/] "
                f"[dim]{left_out}. /new starts over.[/]"
            )
        console.print()
        completion = _generate(session, state.mode, prompt, state.sampling, echo=True)
        console.print()
        if state.mode.chat and completion.tokens == 0 and not completion.interrupted:
            # Zero tokens in chat mode has one cause worth naming: the first thing the
            # model produced was the next turn's label, or end-of-text. Both mean it
            # considers the reply finished, which is the answer to "why did /more do
            # nothing" -- and without saying so, the only thing on screen is a cost line
            # reporting no tokens, which reads like a failure.
            console.print(
                "[dim]The model ended its turn straight away rather than adding anything.[/]"
            )
        ending = _ending(completion, state.sampling, interactive=True)
        if ending:
            console.print(f"[yellow]{ending}[/]")
        console.print(f"[dim]{_cost(completion)}[/]")
        state = _recorded(state, prompt, completion, asked=asked)


def _recorded(state: State, prompt: Prompt, completion: Completion, *, asked: bool) -> State:
    """The state one generation later: what to continue from next time.

    Chat mode keeps turns, so a follow-up is a follow-up. Raw mode keeps the text, because
    prose has no turns -- and it keeps every character of it, since a trailing newline in
    prose is a paragraph break the model chose rather than a separator the renderer will
    write again.
    """
    if not state.mode.chat:
        return replace(state, last=prompt.model + completion.text)
    if asked:
        return replace(state, history=_remember(state.history, prompt.text, completion.text))
    return replace(state, history=_extend(state.history, completion.text))


def _handle_command(
    line: str, state: State, session: InferenceSession
) -> tuple[State, Prompt | None] | None:
    """Apply a ``/`` command.

    Returns ``(state, prompt)`` -- a prompt when the command asks for generation, ``None``
    for the prompt when it does not -- or ``None`` entirely to exit.
    """
    parts = line.split()
    name, argument = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
    mode, sampling = state.mode, state.sampling

    if name in ("/exit", "/quit", "/q"):
        return None
    if name in ("/help", "/?"):
        _help()
        return state, None
    if name in ("/settings", "/set"):
        print_kv("Sampling", [*mode.rows(), *sampling.rows()])
        return state, None
    if name in ("/chat", "/raw"):
        switched = _resolve_mode(session, chat_format=name == "/chat", raw_format=name == "/raw")
        print_kv("Sampling", [*switched.rows(), *sampling.rows()])
        # The conversation is kept across the switch rather than cleared. Switching to
        # raw to probe the same model and switching back is the reason both flags exist,
        # and losing the conversation to it would make that a one-way door.
        return replace(state, mode=switched), None
    if name == "/new":
        return _cleared(state), None
    if name == "/more":
        return state, _more(state, session)

    try:
        updated = _apply_setting(name, argument, sampling)
        if updated is None:
            console.print(f"[yellow]Unknown command {name}.[/] Type /help for the list.")
            return state, None
        _check_sampling(updated)
    except UsageError as error:
        # Printed rather than raised: a typo in an interactive session should cost the
        # user a line, not the loaded model.
        console.print(f"[yellow]{error}[/]  [dim]{error.hint}[/]")
        return state, None
    print_kv("Sampling", updated.rows())
    return replace(state, sampling=updated), None


def _cleared(state: State) -> State:
    """``/new``: forget the conversation, keep the model and the sampling.

    Both halves are cleared whichever mode is on, so that ``/new`` means the same thing in
    both and a mode switch cannot resurrect what was just forgotten.
    """
    if not state.history and not state.last:
        console.print("[dim]Nothing to forget yet.[/]")
        return state
    count = len(state.history)
    what = (
        f"{fmt_int(count)} {'message' if count == 1 else 'messages'} forgotten"
        if count
        else "the last completion forgotten"
    )
    console.print(f"[dim]New conversation {DASH} {what}.[/]")
    return replace(state, history=(), last="")


def _more(state: State, session: InferenceSession) -> Prompt | None:
    """``/more``: what to continue, or ``None`` with a reason printed.

    Chat mode continues the conversation's own last reply, so the model is asked to write
    more of a turn it is already inside. Raw mode continues the text as text: it is
    already model output, so it is passed through rather than rendered again -- rendering
    it would ask the model to answer its own reply under a fresh label.
    """
    if state.mode.chat:
        if not state.history:
            console.print("[yellow]Nothing to continue yet.[/] Ask it something first.")
            return None
        return _continue(session, state.history)
    if not state.last:
        console.print("[yellow]Nothing to continue yet.[/] Give it something first.")
        return None
    return Prompt(text=state.last, model=state.last)


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
    session: InferenceSession, mode: Mode, prompt: Prompt, sampling: Sampling, *, echo: bool
) -> Completion:
    """Stream one completion, printing as it arrives when ``echo``.

    Ctrl-C mid-generation keeps what was produced: the tokens were really generated and
    the timing over them is real, so the completion is returned rather than discarded.

    The stop strings are the mode's, which in chat mode are the template's own labels.
    They are not this module's to invent -- a sampler prompted in one layout and stopped
    at another would cut in the wrong place, or not at all.
    """
    truncated = _truncation(session, prompt.model)
    if truncated is not None and echo:
        console.print(
            f"[yellow]The prompt is {fmt_int(truncated)} tokens and this model's context "
            f"is {fmt_int(session.model_config.seq_len)}.[/] "
            "[dim]Only the last part of it was used.[/]"
        )

    pieces: list[str] = []
    tokens = 0
    finish: Finish | None = None
    interrupted = False
    started = time.perf_counter()
    try:
        for piece in session.stream_pieces(
            prompt.model,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
            stop_at_eot=sampling.stop_at_eot,
            stop=mode.stops,
        ):
            # piece.tokens is cumulative and counts tokens whose text is still being
            # held back, which counting pieces would not -- see infer.StreamPiece.
            tokens = piece.tokens
            # Only the last piece has one, so this ends up holding the stream's own
            # answer rather than a running guess at it.
            finish = piece.finish or finish
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

    # After the reply, not before it: whether the window slid is a fact about the reply
    # that was written, and predicting it from --tokens would warn on every short answer
    # that then stopped at EOT well inside the context.
    slides_after = _slides_after(session, prompt.model, tokens)
    if slides_after is not None and echo:
        console.print(
            f"\n[yellow]The window slid after {fmt_int(slides_after)} tokens: the last "
            f"{fmt_int(tokens - slides_after)} were written without the start of the "
            f"prompt.[/] [dim]This model's context is "
            f"{fmt_int(session.model_config.seq_len)} tokens.[/]"
        )

    return Completion(
        prompt=prompt.text,
        text="".join(pieces),
        tokens=tokens,
        seconds=elapsed,
        model_prompt=prompt.model,
        truncated_from=truncated,
        interrupted=interrupted,
        finish=finish,
        slides_after=slides_after,
    )


def _slides_after(session: InferenceSession, prompt: str, produced: int) -> int | None:
    """How many tokens the reply ran before its window dropped the front of *prompt*.

    ``None`` when it never did -- the ordinary case, and the one worth saying nothing
    about. *produced* is what the reply actually came to rather than what ``--tokens``
    allowed, because most replies stop at EOT well inside the context and a note about a
    slide that did not happen is a yellow line every turn for nothing.

    When it did: :meth:`GPT.generate_stream` re-anchors its cache once a step would run
    past ``seq_len`` (measured in :func:`~trainai.model.gpt.slide_caches` against a fresh
    forward pass), so from then on the front of the conversation goes one token per
    generated token. Room for ``r`` tokens means tokens 1..``r`` see the whole prompt and
    token ``r + 1`` is the first that does not; after ``seq_len`` of them none of the
    prompt is left and the reply is continuing nothing but itself.

    ``None`` too when the prompt did not fit in the first place, which is
    :func:`_truncation`'s sentence rather than this one's. The front was discarded before
    generation started there, so saying it was discarded during generation as well is a
    second yellow line about one thing.

    Reserving room for the answer would not fix this and does not need to -- see
    :func:`_fit`, which counts why not.
    """
    room = session.model_config.seq_len - len(session.tokenizer.encode(prompt))
    if room < 0 or room >= produced:
        return None
    return room


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


def _ending(completion: Completion, sampling: Sampling, *, interactive: bool) -> str | None:
    """What to say about how the generation ended, or ``None`` when there is nothing.

    Only the token limit earns a line. A reply that ended at end-of-text or at somebody
    else's label is a reply that finished, and saying so every time is noise; a reply the
    budget cut off is *unfinished*, which is indistinguishable on screen from a model that
    stops mid-sentence because it is small. Without this the answer to "why did it stop
    there" is a guess, and the guess costs a re-run at a higher ``--tokens``.

    Nothing is said for an interrupted generation. The user pressed Ctrl-C, so they know
    why it stopped, and ``_generate`` already printed "Stopped."
    """
    if completion.interrupted or completion.finish is None or not completion.finish.cut:
        return None
    fix = (
        "/more continues it, /tokens raises the limit"
        if interactive
        else "--tokens raises the limit"
    )
    return (
        f"The model was still writing at the {fmt_int(sampling.max_new_tokens)}-token "
        f"limit, so this reply is unfinished. [dim]{fix}.[/]"
    )


# --------------------------------------------------------------------------- #
# What the user is told before they type
# --------------------------------------------------------------------------- #
def _banner(session: InferenceSession, mode: Mode) -> None:
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
        *mode.rows(),
    ]
    if layout.note:
        loaded.append(("Note", f"[yellow]{layout.note}[/]"))
    print_kv("Loaded", loaded)

    notes = _worth_knowing(mode)
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


def _worth_knowing(mode: Mode) -> list[str]:
    """The first two bullets of the banner, which the mode decides.

    In raw mode the warning is the one this command has always given: it continues text,
    and whether that looks like an answer depends on a corpus this command cannot see. In
    chat mode two of those unknowns are gone -- the checkpoint says the corpus was
    conversations, and this command is writing the labels -- so saying "give it the
    beginning of something rather than a request" there would be advice against the tool's
    own behaviour.
    """
    if mode.chat:
        return [
            f"Each line is sent as a [bold]{ROLE_LABELS['user']}[/] message and the model "
            f"answers as [bold]{ROLE_LABELS[REPLY_ROLE]}[/]. Generation stops where it "
            "starts writing the next turn. [bold]/raw[/] switches this off.",
            "The conversation is remembered, so a follow-up question has the exchange "
            "before it behind it. [bold]/new[/] forgets it; [bold]/more[/] continues the "
            "last reply.",
        ]
    return [
        "This is a [bold]base[/] language model: it continues text. Whether it answers "
        "a question depends on the corpus it saw, and this command cannot tell you "
        "which -- train on prose and it continues prose, so give it the beginning of "
        "something rather than a request.",
        "Each line is a fresh completion. [bold]/more[/] continues the last one.",
    ]


def _help() -> None:
    print_bullets(
        "Commands",
        [
            "[bold]/more[/] -- continue the last completion instead of starting fresh",
            "[bold]/new[/] -- forget the conversation and start over",
            "[bold]/chat[/] | [bold]/raw[/] -- wrap what you type in the chat template, "
            "or send it unchanged",
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
def _check_format_flags(*, chat_format: bool, raw_format: bool) -> None:
    """Refuse ``--chat --raw``, before a checkpoint is loaded onto a device.

    Silently preferring one of the two would be worse than it looks: the whole point of
    the pair is that the prompt the model sees is not guessable from the text, so a
    contradiction resolved by argument order is a contradiction the user never sees.
    """
    if chat_format and raw_format:
        raise UsageError(
            "--chat and --raw ask for opposite things: one wraps your text in the chat "
            "template, the other sends it unchanged.",
            hint="Pass one, or neither -- with neither, the checkpoint's own record of "
            "how its dataset was rendered decides.",
            details={"chat": True, "raw": True},
        )


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
