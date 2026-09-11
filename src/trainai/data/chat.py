"""The one chat format TrainAI understands, and which characters it counts as trained.

This module exists because of a silent-mismatch surface. A model trained with one
chat layout and prompted with another degrades without an error -- the same class of
failure as a tokenizer fingerprint mismatch, and the reason that one is checked. So
there is exactly **one** template, it is versioned, and :mod:`trainai.cli.chat` and
``trainai data prepare`` both read it from here rather than each spelling it out.

**The input is typed, not sniffed.** A conversation is a list of
``{"role": ..., "content": ...}`` objects. The obvious alternative -- take the
flattened text a corpus already holds and find the replies by searching for
``"Assistant:"`` -- was rejected: a reply that happens to contain that string, and a
model discussing itself will produce many, gets masked in the wrong place, silently.
That is the tabular-CSV mistake in a new costume, so roles are read from a field or
not at all, and an unrecognised role is an error rather than a guess.

**Rendering matches the corpus this repo already ships.** ``chat.jsonl`` holds
``User: ...\\nAssistant: ...`` turns separated by a blank line, so that is the
template: each message is ``{Label}: {content}``, messages are joined with a single
newline, and a blank line is inserted before any message that starts a new turn --
that is, a non-assistant message following an assistant one. Preparing the typed
records therefore produces byte-identical text to the flattened corpus, which is what
makes the flattened corpus and the masked one comparable at all.

**A trained span runs to the next label, not to the end of the reply.** The span for
an assistant message starts at the first character of its content and ends where the
next message's label begins -- so the separator after a reply is trained too. At
inference the harness writes ``Assistant:`` and the model completes; if the
terminator were masked out, the model would be trained to start a reply and never to
finish one, and generation would run to the token limit every time. The final message
in a conversation has no following label, so its span ends at the end of the text;
the stop signal there is the end-of-text token, which is appended after this module
has done its work.

Nothing here tokenizes. Spans are **character** offsets into ``text``; converting
them to token ranges is the encoder's job, and doing it here would put a tokenizer
dependency into the one part of ingestion that has none.

**Rendering has an inverse, and it lives here too.** A generation harness that keeps a
conversation has to turn what the model produced back into a message, and the two
characters that makes it get wrong -- the gap the model writes after the label, the
separator it stops on before the next one -- are the same two this module put there. So
:func:`reply_content` is here rather than in the harness, for the reason everything else
is: a second place that knows the layout is a second place it can drift from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "REPLY_ROLE",
    "ROLE_LABELS",
    "TEMPLATE_VERSION",
    "TRAINED_ROLES",
    "TURN_BOUNDARIES",
    "ChatFormatError",
    "Conversation",
    "describe_template",
    "render_conversation",
    "render_prompt",
    "reply_content",
]

#: Bumped if the rendering below ever changes. It reaches the dataset manifest, so a
#: model can be told apart from one trained under a different layout instead of the
#: mismatch showing up as mediocre replies.
TEMPLATE_VERSION: Final = 1

#: The roles this template can render, and the label each one is written with.
#: Insertion order is the order they are listed in error messages.
ROLE_LABELS: Final[dict[str, str]] = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}

#: Whose content is trained on. A single role today; a set because the loss mask is
#: defined in terms of "which roles" and a future template may have more than one.
TRAINED_ROLES: Final = frozenset({"assistant"})

#: The role a generation harness asks the model to speak as. The same role as the one
#: trained on today, and deliberately named separately: "what the loss covers" and "whose
#: turn comes next" are two questions, and a template that ever trains on two roles would
#: otherwise make the second one an arbitrary pick out of a set.
REPLY_ROLE: Final = "assistant"

_TURN_SEPARATOR: Final = "\n\n"
_LINE_SEPARATOR: Final = "\n"

#: Between a label's colon and the content after it. Written down rather than inlined
#: because :func:`render_prompt` has to end *before* it -- see the measurement there.
_LABEL_GAP: Final = " "

#: What it looks like when a model stops replying and starts writing somebody else's
#: turn. A trained model does this constantly -- it was shown whole conversations, so
#: continuing past its own reply into the next ``User:`` is exactly what the corpus
#: taught -- and a generation harness that does not cut there shows the user a reply
#: with an invented follow-up question stapled to it.
#:
#: One entry per label, with a *single* newline rather than the blank line
#: :func:`render_conversation` writes between turns. The shorter form matches both: a
#: reply followed by ``\n\nUser:`` still matches at its second newline, leaving one
#: trailing newline for the caller to strip. Matching the longer form only would miss
#: the case where the model writes the label after a single newline, which it does,
#: because a conversation's *within-turn* messages are joined with one.
TURN_BOUNDARIES: Final[tuple[str, ...]] = tuple(
    f"{_LINE_SEPARATOR}{label}:" for label in ROLE_LABELS.values()
)


class ChatFormatError(ValueError):
    """A conversation this template cannot render.

    Deliberately not a :class:`~trainai.errors.DatasetError`: this module never sees a
    path, a line number or an ``--on-error`` setting. It reports *what* is wrong and
    what to do about it, and the caller -- which does know where the record came from
    -- turns that into the error the user reads. ``fields`` carries whatever structured
    detail the caller should pass through to ``details``.
    """

    def __init__(self, problem: str, *, hint: str, fields: dict[str, Any] | None = None) -> None:
        super().__init__(problem)
        self.problem = problem
        self.hint = hint
        self.fields = fields or {}


@dataclass(frozen=True)
class Conversation:
    """Rendered text, plus the character spans within it that are trained on.

    Spans are half-open ``[start, end)``, sorted, non-overlapping, and expressed in
    characters of ``text``. There is one per trained message, not one merged run:
    merging them would lose the fact that a two-reply conversation has two replies,
    which is what makes the "trained share" number in ``data inspect`` meaningful.
    """

    text: str
    spans: tuple[tuple[int, int], ...]

    @property
    def trained_chars(self) -> int:
        return sum(end - start for start, end in self.spans)


def describe_template() -> dict[str, Any]:
    """The template as data, for the dataset manifest and for ``--json`` output."""
    return {
        "version": TEMPLATE_VERSION,
        "labels": dict(ROLE_LABELS),
        "trained_roles": sorted(TRAINED_ROLES),
    }


def render_conversation(messages: Any) -> Conversation:
    """Render a list of ``{"role", "content"}`` objects, with the trained spans marked.

    Raises :class:`ChatFormatError` for anything it cannot render exactly, including a
    conversation with no assistant message at all -- that one is a refusal rather than
    an empty span list on purpose. Such a record has nothing to learn from, so keeping
    it would put tokens in the dataset that every step is masked out of, and the
    "trained share" figure would quietly fall without anything being wrong upstream.
    """
    turns = _parse(messages)
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    previous: str | None = None
    pending = False

    for role, content in turns:
        separator = "" if previous is None else _separator(previous, role)
        if pending:
            # A reply is trained up to where the next label starts, separator included.
            start, _ = spans[-1]
            spans[-1] = (start, position + len(separator))
            pending = False
        label = f"{ROLE_LABELS[role]}:{_LABEL_GAP}"
        pieces.append(separator)
        pieces.append(label)
        position += len(separator) + len(label)
        pieces.append(content)
        if role in TRAINED_ROLES:
            spans.append((position, position + len(content)))
            pending = True
        position += len(content)
        previous = role

    return Conversation("".join(pieces), tuple(spans))


#: Content used to locate where a reply would begin. Never appears in the returned
#: prompt: the text is cut at the span that marks it, not by looking for this string
#: afterwards. A test passes this exact value as a *user's* message to pin the
#: difference down -- a prompt built by stripping the probe out would eat it.
_PROBE: Final = "(reply)"


def render_prompt(messages: Any) -> str:
    """The text to generate an assistant reply from: the turns so far, then the label.

    ``[{"role": "user", "content": "Hi!"}]`` renders as ``"User: Hi!\\nAssistant:"``.
    Pass the conversation up to and including the latest non-assistant message; to
    continue a reply the model already started, render the whole conversation with
    :func:`render_conversation` instead, which ends in the reply itself.

    Derived from :func:`render_conversation` rather than assembled again here. That is
    the whole point of the function: a prompt built from its own copy of the label, the
    colon and the separator rule is a prompt that can drift from what the shards hold,
    and the drift is invisible -- the model answers, a little worse, for a reason no
    error names. Rendering one throwaway reply and cutting at its span costs a string
    copy and cannot disagree.

    **It stops at the colon, not after the space the label puts there**, and that one
    character is the difference between a prompt the model recognises and noise. A
    byte-level BPE merges a space into the word after it, so the shards hold
    ``"Assistant:"`` followed by one ``" Hey"`` token and never a lone space token; a
    prompt ending in the space asks the model to continue from a token sequence that
    appears nowhere in its training data, and what comes back is garbage without an
    error. Measured on the 4,000-conversation corpus in ``data/chat-typed`` with its own
    trained tokenizer: ending after the space, 0 of 4,000 prompts were a token prefix of
    the conversation as the model saw it; ending at the colon, 4,000 of 4,000 were. The
    model writes the space itself, as part of the first word.

    Refusals are :func:`render_conversation`'s, because the format is the same one.
    """
    turns = _parse(messages, trained_required=False)
    if turns[-1][0] == REPLY_ROLE:
        raise ChatFormatError(
            f"the conversation already ends with {ROLE_LABELS[REPLY_ROLE]}, so there is "
            "no reply to ask for",
            hint=(
                "Pass the turns up to the latest user message. To continue a reply the "
                "model has already begun, render the whole conversation instead."
            ),
            fields={"last_role": turns[-1][0]},
        )
    whole = render_conversation(
        [{"role": role, "content": content} for role, content in turns]
        + [{"role": REPLY_ROLE, "content": _PROBE}]
    )
    return whole.text[: whole.spans[-1][0] - len(_LABEL_GAP)]


def reply_content(generated: str, *, continuing: bool = False) -> str:
    """The message content inside text a model generated, with the template's own off.

    Two characters of what comes back from a sampler belong to the layout rather than to
    the reply, and both are at the edges:

    * **The gap after the label.** :func:`render_prompt` stops at the colon, so the model
      writes that space itself as part of its first token -- see the measurement there.
      Storing it as content and rendering the conversation again puts two spaces after the
      label, which is a layout no corpus contains.
    * **The separator before the next label.** Generation stops at ``\\nUser:``, and the
      newline it stopped on is the one :func:`render_conversation` writes between messages.
      Keeping it means rendering writes a second one.

    Neither is visible in the reply the user reads, and both change the *next* prompt. That
    is the failure this module exists to prevent, so the inverse of rendering lives next to
    rendering instead of in the caller.

    ``continuing`` is for text that continues a reply already in progress -- generated from
    :func:`render_conversation`, which ends inside the content rather than at a label.
    There is no gap to remove there, and removing one would eat a space the model meant.
    """
    text = generated if continuing else generated.removeprefix(_LABEL_GAP)
    return text.rstrip(_LINE_SEPARATOR)


def _separator(previous: str, role: str) -> str:
    """A blank line starts a new turn; anything within a turn is a single newline."""
    if previous in TRAINED_ROLES and role not in TRAINED_ROLES:
        return _TURN_SEPARATOR
    return _LINE_SEPARATOR


def _parse(messages: Any, *, trained_required: bool = True) -> list[tuple[str, str]]:
    """Validate the record and return ``(role, content)`` pairs, roles normalised.

    ``trained_required`` is what separates a corpus record from a prompt. A record with
    no reply in it is refused, for the reason below; a *prompt* consists of exactly that
    and is the normal case, so :func:`render_prompt` turns the requirement off.
    """
    if not isinstance(messages, list | tuple):
        raise ChatFormatError(
            f"the messages field holds {type(messages).__name__}, not a list",
            hint=(
                'Each record needs a list of {"role": ..., "content": ...} objects. '
                "For a corpus that is already flattened into one string per record, "
                "drop --jsonl-messages-field and use --jsonl-field instead -- it "
                "trains on every token, with no mask."
            ),
            fields={"type": type(messages).__name__},
        )
    if not messages:
        raise ChatFormatError(
            "the messages list is empty",
            hint="A conversation needs at least one user message and one reply.",
        )

    known = ", ".join(ROLE_LABELS)
    turns: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        where = f"message {index}"
        if not isinstance(message, dict):
            raise ChatFormatError(
                f"{where} is a {type(message).__name__}, not an object",
                hint=f'Every message is {{"role": ..., "content": ...}}, with role one of {known}.',
                fields={"message": index},
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ChatFormatError(
                f'{where} has no string "role" (it has: {_keys(message)})',
                hint=f"Every message needs a role, one of {known}.",
                fields={"message": index, "available_fields": sorted(map(str, message))},
            )
        normalised = role.strip().lower()
        if normalised not in ROLE_LABELS:
            raise ChatFormatError(
                f"{where} has role {role!r}, which this template does not render",
                hint=(
                    f"Roles are {known}. TrainAI will not guess what an unknown role "
                    "means: rendering it under the wrong label, or dropping it, changes "
                    "what the model is trained to produce without saying so. Rewrite the "
                    "role, or drop the record with --on-error skip."
                ),
                fields={"message": index, "role": role, "known_roles": list(ROLE_LABELS)},
            )
        if not isinstance(content, str):
            raise ChatFormatError(
                f'{where} ({normalised}) has no string "content" (it has: {_keys(message)})',
                hint='Every message needs its text in a string "content" field.',
                fields={"message": index, "role": normalised},
            )
        if not content.strip():
            raise ChatFormatError(
                f"{where} ({normalised}) has empty content",
                hint=(
                    "An empty message renders as a bare label, which teaches the model "
                    "to produce one. Drop the record with --on-error skip, or fix it."
                ),
                fields={"message": index, "role": normalised},
            )
        turns.append((normalised, content))

    if trained_required and not any(role in TRAINED_ROLES for role, _ in turns):
        raise ChatFormatError(
            "the conversation has no "
            + " or ".join(sorted(TRAINED_ROLES))
            + " message, so there is nothing in it to train on",
            hint=(
                "With a loss mask, only these roles' text is learned from; a record "
                "without one contributes tokens the model is never trained to predict. "
                "Drop it with --on-error skip."
            ),
            fields={"roles": sorted({role for role, _ in turns})},
        )
    return turns


def _keys(message: dict[Any, Any]) -> str:
    keys = sorted(str(key) for key in message)
    return ", ".join(keys) if keys else "(no keys)"
