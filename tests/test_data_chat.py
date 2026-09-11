"""Tests for :mod:`trainai.data.chat` and the chat mode it gives ``ingest``.

The contract worth guarding is narrow and easy to break invisibly: the rendered text
must equal the flattened corpus this repo already ships, and the spans must cover the
replies and nothing else. Get the first wrong and a masked run cannot be compared
with an unmasked one; get the second wrong and the model is trained on the prompts it
will be given at inference, or on nothing at all, and in both cases the loss curve
looks perfectly normal.

So the assertions here are mostly about *characters*, spelled out literally rather
than computed from the same code under test. A span test that recomputes the offsets
the renderer computed would pass whatever the renderer did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainai.data.chat import (
    REPLY_ROLE,
    ROLE_LABELS,
    TEMPLATE_VERSION,
    TRAINED_ROLES,
    TURN_BOUNDARIES,
    ChatFormatError,
    describe_template,
    render_conversation,
    render_prompt,
    reply_content,
)
from trainai.data.ingest import Document, IngestOptions, Ingestor, IngestStats
from trainai.errors import DatasetFormatError, UsageError

#: The shape of one record in ``data/corpus-assistant/chat.jsonl``, typed. The
#: flattened text below it is what that file actually holds, quoted verbatim, so this
#: pair is the one assertion that pins the template to the shipped corpus.
CONVERSATION = [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hey! What would you like help with?"},
    {"role": "user", "content": "what is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 = 4."},
]
FLATTENED = (
    "User: Hi!\n"
    "Assistant: Hey! What would you like help with?\n"
    "\n"
    "User: what is 2 + 2?\n"
    "Assistant: 2 + 2 = 4."
)


def chat_corpus(directory: Path, conversations: list[object], field: str = "messages") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "chat.jsonl"
    lines = [json.dumps({field: conversation}) for conversation in conversations]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def ingest(root: Path, **options: object) -> tuple[list[Document], IngestStats]:
    ingestor = Ingestor(IngestOptions(**options))  # type: ignore[arg-type]
    return list(ingestor.documents(ingestor.discover(root))), ingestor.stats


# --------------------------------------------------------------------------- #
# The template
# --------------------------------------------------------------------------- #
def test_rendering_matches_the_flattened_corpus_byte_for_byte() -> None:
    """The template is not a new format; it is the one ``chat.jsonl`` already holds.

    Verified beyond this test on the real file: 20,000 of 20,000 records parsed back
    into messages and re-rendered identically. If this assertion ever has to change,
    ``TEMPLATE_VERSION`` changes with it -- every model trained under the old layout
    is a model that will be prompted with the new one.
    """
    assert render_conversation(CONVERSATION).text == FLATTENED


def test_the_trained_spans_cover_the_replies_and_their_terminator() -> None:
    """Spelled out as literal strings, not as offsets recomputed from the renderer."""
    conversation = render_conversation(CONVERSATION)

    kept = [conversation.text[start:end] for start, end in conversation.spans]
    assert kept == ["Hey! What would you like help with?\n\n", "2 + 2 = 4."]
    # The blank line after a reply is trained on: at inference the harness writes
    # "Assistant:" and the model completes, so if the terminator were masked the
    # model would learn to start a reply and never to end one.
    assert kept[0].endswith("\n\n")


def test_nothing_a_prompt_contains_is_trained_on() -> None:
    """The complement of the spans, which is what the loss mask actually removes."""
    conversation = render_conversation(CONVERSATION)

    masked = ""
    position = 0
    for start, end in conversation.spans:
        masked += conversation.text[position:start]
        position = end
    masked += conversation.text[position:]

    assert masked == "User: Hi!\nAssistant: User: what is 2 + 2?\nAssistant: "
    for reply in ("Hey!", "2 + 2 = 4."):
        assert reply not in masked


def test_spans_are_sorted_disjoint_and_inside_the_text() -> None:
    conversation = render_conversation(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second"},
        ]
    )

    previous = 0
    for start, end in conversation.spans:
        assert 0 <= previous <= start < end <= len(conversation.text)
        previous = end
    assert conversation.trained_chars == sum(end - start for start, end in conversation.spans)


def test_a_system_message_opens_the_conversation_without_a_blank_line() -> None:
    conversation = render_conversation(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert conversation.text == "System: Be brief.\nUser: hello\nAssistant: hi"


def test_roles_are_case_insensitive_but_not_open_ended() -> None:
    """``"User"`` and ``"user"`` are the same role; ``"tool"`` is not a role at all."""
    assert (
        render_conversation(
            [{"role": "USER", "content": "hi"}, {"role": " Assistant ", "content": "hello"}]
        ).text
        == "User: hi\nAssistant: hello"
    )

    with pytest.raises(ChatFormatError) as caught:
        render_conversation([{"role": "user", "content": "hi"}, {"role": "tool", "content": "{}"}])
    assert "'tool'" in caught.value.problem
    assert "will not guess" in caught.value.hint


def test_a_conversation_with_no_reply_is_refused() -> None:
    """Not an empty span list: that would put unlearnable tokens in the dataset.

    Every token of such a record is masked out of every step, so it costs training
    time and contributes nothing, and the trained share would fall without anything
    upstream being wrong.
    """
    with pytest.raises(ChatFormatError) as caught:
        render_conversation([{"role": "user", "content": "hello?"}])

    assert "nothing in it to train on" in caught.value.problem


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ("User: hi", "not a list"),
        ([], "empty"),
        (["hi"], "not an object"),
        ([{"content": "hi"}], 'no string "role"'),
        ([{"role": "user"}], 'no string "content"'),
        ([{"role": "user", "content": 3}], 'no string "content"'),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "  "}], "empty"),
    ],
)
def test_every_shape_it_cannot_render_is_refused_with_a_reason(
    messages: object, expected: str
) -> None:
    with pytest.raises(ChatFormatError) as caught:
        render_conversation(messages)

    assert expected in caught.value.problem
    assert caught.value.hint


def test_the_template_is_describable_as_data() -> None:
    """It reaches the manifest, so a model can be told apart from one trained under
    a different layout instead of the mismatch showing up as mediocre replies."""
    described = describe_template()

    assert described["version"] == TEMPLATE_VERSION
    assert described["labels"] == dict(ROLE_LABELS)
    assert described["trained_roles"] == sorted(TRAINED_ROLES)
    assert json.loads(json.dumps(described)) == described


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
# What a generation harness writes before asking the model to continue. The risk here
# is not that it looks wrong -- it is that it looks right and differs from the shards
# by one character, and the character that matters is the space after the colon. The
# shards hold "Assistant: Hey", which a byte-level BPE tokenizes with the space merged
# into " Hey"; a prompt ending after that space is a token sequence the model has never
# seen once, and what comes back is not a slightly worse reply but noise. So the prompt
# stops at the colon and the model writes the space itself. The tests below pin it to
# the *rendered conversation* rather than to a second copy of the layout, one of them
# spells it out literally, and one takes a real tokenizer to the boundary.
@pytest.mark.parametrize(
    "history",
    [
        pytest.param([{"role": "user", "content": "Hi!"}], id="one-turn"),
        pytest.param(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi!"}],
            id="system",
        ),
        pytest.param(CONVERSATION[:3], id="multi-turn"),
    ],
)
def test_the_prompt_is_the_conversation_cut_where_the_reply_begins(
    history: list[dict[str, str]],
) -> None:
    """The invariant the function exists for: a prompt cannot drift from the shards.

    Compared against a conversation rendered with a *different* reply, so this passes
    only if the prompt is the shared prefix -- if ``render_prompt`` grew its own copy of
    the label, the colon or the separator rule, the two sides diverge here.
    """
    whole = render_conversation([*history, {"role": REPLY_ROLE, "content": "a reply"}])
    prompt = render_prompt(history)

    assert whole.text.startswith(prompt)
    # And what it cut off is the space and then the reply: the prompt stops one character
    # before the model's first character, because that space belongs to the model's first
    # token. Asserted as one string so an off-by-one either way fails here.
    assert whole.text[len(prompt) :] == " a reply"


def test_the_prompt_ends_at_the_colon_the_shards_hold() -> None:
    """The literal form, in case the invariant above is ever satisfied by two wrongs."""
    assert render_prompt([{"role": "user", "content": "Hi!"}]) == "User: Hi!\nAssistant:"
    assert render_prompt(CONVERSATION[:3]) == (
        "User: Hi!\n"
        "Assistant: Hey! What would you like help with?\n"
        "\n"
        "User: what is 2 + 2?\n"
        "Assistant:"
    )
    # Still a prefix of the corpus text, one character shorter than it used to be.
    assert FLATTENED.startswith(render_prompt([{"role": "user", "content": "Hi!"}]))
    assert not render_prompt(CONVERSATION[:3]).endswith(" ")


def test_the_prompt_is_a_token_prefix_and_not_only_a_text_prefix() -> None:
    """The bug this pins produced no error, no warning and no readable text.

    A text prefix is not enough, and the difference is invisible in every assertion
    above: the pre-tokenizer keeps a space with the word after it, so the corpus's
    ``"Assistant: Hey"`` holds one ``" Hey"`` token and no lone space token anywhere. A
    prompt ending after the space therefore asks the model to continue a token sequence
    that occurs nowhere in its training data. Measured on the 4,000-conversation corpus
    in ``data/chat-typed`` with the tokenizer trained on it: 0 of 4,000 prompts were a
    token prefix before this, and 4,000 of 4,000 after. Here it is one conversation and a
    tokenizer trained on the shipped text, which is enough to fail if the space returns.
    """
    from trainai.data.tokenizer import train_tokenizer

    tokenizer = train_tokenizer([FLATTENED * 40], vocab_size=300)
    whole = tokenizer.encode(FLATTENED)
    prompt = tokenizer.encode(render_prompt(CONVERSATION[:1]))

    assert whole[: len(prompt)] == prompt
    # And the token the model produces first carries the space, which is why the prompt
    # must not: encoding it as its own token is what never happens in the shards.
    assert tokenizer.decode([whole[len(prompt)]]).startswith(" ")
    assert tokenizer.encode(render_prompt(CONVERSATION[:1]) + " ") != whole[: len(prompt) + 1]


def test_a_prompt_needs_no_reply_in_it_although_a_corpus_record_does() -> None:
    """The one place the two callers of the parser legitimately differ.

    A record with no reply is refused during ingestion because every one of its tokens
    is masked out of every step. A *prompt* is exactly that record -- it is the normal
    case, and the whole point is that the reply does not exist yet.
    """
    history = [{"role": "user", "content": "hello?"}]

    assert render_prompt(history) == "User: hello?\nAssistant:"
    with pytest.raises(ChatFormatError) as caught:
        render_conversation(history)
    assert "nothing in it to train on" in caught.value.problem


def test_a_conversation_that_already_ends_with_a_reply_is_refused() -> None:
    """Rendering the label twice would ask the model to answer its own answer."""
    with pytest.raises(ChatFormatError) as caught:
        render_prompt(CONVERSATION)

    assert "already ends with Assistant" in caught.value.problem
    assert caught.value.fields["last_role"] == REPLY_ROLE
    # The way to continue a reply the model started is named, because it exists.
    assert caught.value.hint and "render the whole conversation" in caught.value.hint


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ("User: hi", "not a list"),
        ([], "empty"),
        ([{"role": "user", "content": "hi"}, {"role": "tool", "content": "{}"}], "'tool'"),
        ([{"role": "user", "content": ""}], "empty content"),
    ],
)
def test_the_prompt_refuses_what_the_renderer_refuses(messages: object, expected: str) -> None:
    """One format, so one set of refusals: they come from the shared parser."""
    with pytest.raises(ChatFormatError) as caught:
        render_prompt(messages)

    assert expected in caught.value.problem
    assert caught.value.hint


def test_the_throwaway_reply_used_to_find_the_cut_never_reaches_the_prompt() -> None:
    """The prompt is built by rendering a placeholder reply and cutting at its span.

    The placeholder is read from the module rather than copied here, and passed as a
    *user's* message, so this fails whether the cut is off by one or the placeholder is
    removed by searching the text for it afterwards -- the second of which would also
    eat the same words out of somebody's question.
    """
    from trainai.data.chat import _PROBE

    for content in ("Hi!", _PROBE, f"is {_PROBE} in {_PROBE}?"):
        assert render_prompt([{"role": "user", "content": content}]) == (
            f"User: {content}\nAssistant:"
        )


def test_every_label_has_a_turn_boundary() -> None:
    """A label without one is a label a generation harness would not stop at."""
    assert len(TURN_BOUNDARIES) == len(ROLE_LABELS)
    assert set(TURN_BOUNDARIES) == {f"\n{label}:" for label in ROLE_LABELS.values()}


def test_a_turn_boundary_finds_where_a_reply_stops_in_the_shipped_corpus() -> None:
    """Checked against the text a model is actually trained on, not a hand-built string.

    A model shown whole conversations continues past its own reply into the next
    ``User:``, because that is what the corpus taught it. This is the cut that keeps
    the invented follow-up question out of what the user sees.
    """
    continuation = FLATTENED[len("User: Hi!\nAssistant: ") :]

    cut = min(
        continuation.index(boundary) for boundary in TURN_BOUNDARIES if boundary in continuation
    )

    # One trailing newline is left for the caller to strip: the boundary is the shorter
    # form, so against the corpus's blank line it matches at the second newline.
    assert continuation[:cut] == "Hey! What would you like help with?\n"
    assert continuation[:cut].rstrip("\n") == CONVERSATION[1]["content"]


def test_a_boundary_matches_a_label_after_one_newline_and_after_two() -> None:
    """Why the constant holds the single-newline form rather than the blank line.

    The corpus separates turns with a blank line, but a sampling model writes the label
    after one newline too -- within-turn messages are joined with one -- and a harness
    that matched only the longer form would miss it and keep generating.
    """
    for continuation in ("Hey!\nUser: hi again", "Hey!\n\nUser: hi again"):
        cut = min(
            continuation.index(boundary) for boundary in TURN_BOUNDARIES if boundary in continuation
        )
        assert continuation[:cut].rstrip("\n") == "Hey!"


# --------------------------------------------------------------------------- #
# Turning a generation back into a message
# --------------------------------------------------------------------------- #
def test_a_reply_loses_the_gap_the_model_wrote_and_the_separator_it_stopped_on() -> None:
    """The inverse of rendering, checked against a generation the corpus can produce.

    The prompt ends at the colon, so the model writes the gap as part of its first token,
    and generation stops at the newline that begins the next label. Both are the
    renderer's, and both are invisible until the conversation is rendered again.
    """
    assert reply_content(" Hey! What would you like help with?\n") == CONVERSATION[1]["content"]


def test_only_one_gap_comes_off_a_reply() -> None:
    """A second space is the model's, not the label's.

    ``lstrip`` here would silently reformat a reply that begins with deliberate
    whitespace -- indented code, most obviously -- and reformatting is not this function's
    job.
    """
    assert reply_content("  indented") == " indented"
    assert reply_content("no gap at all") == "no gap at all"


def test_a_continuation_keeps_its_leading_space() -> None:
    """``/more`` generates from inside the content, where there is no label gap to remove.

    Removing one anyway would join two words the model meant to separate, once per
    continuation, and the only place it would show is the next prompt.
    """
    assert reply_content(" more words\n", continuing=True) == " more words"


def test_a_remembered_reply_renders_back_to_what_the_model_was_given() -> None:
    """The round trip, which is the only assertion that catches a doubled separator.

    Render a prompt, generate a reply the way a model does -- the gap in front, the next
    label's newline behind -- store it, and render the conversation again: the result has
    to be the text a corpus holds, character for character. This is the whole reason the
    inverse lives beside the renderer.

    Spelled out rather than sliced out of ``FLATTENED``, for the reason the module
    docstring gives: an expectation computed from the thing under test passes whatever
    that thing does.
    """
    generated = " Hey! What would you like help with?\n"

    stored = [CONVERSATION[0], {"role": REPLY_ROLE, "content": reply_content(generated)}]

    assert render_conversation(stored).text == (
        "User: Hi!\nAssistant: Hey! What would you like help with?"
    )
    assert render_prompt([*stored, CONVERSATION[2]]) == (
        "User: Hi!\n"
        "Assistant: Hey! What would you like help with?\n"
        "\n"
        "User: what is 2 + 2?\n"
        "Assistant:"
    )


# --------------------------------------------------------------------------- #
# Chat mode in the reader
# --------------------------------------------------------------------------- #
def test_a_chat_corpus_yields_documents_carrying_their_spans(tmp_path: Path) -> None:
    chat_corpus(tmp_path, [CONVERSATION, CONVERSATION])

    documents, stats = ingest(tmp_path, jsonl_messages_field="messages")

    assert [document.text for document in documents] == [FLATTENED, FLATTENED]
    assert stats.chat_documents == 2
    assert stats.chat_chars == 2 * len(FLATTENED)
    assert stats.chat_trained_chars == 2 * sum(
        end - start for start, end in documents[0].trained_spans
    )
    assert 0 < stats.chat_trained_chars < stats.chat_chars


def test_without_the_option_the_same_file_is_read_as_text_and_has_no_spans(
    tmp_path: Path,
) -> None:
    """The negative control. Chat mode is reached only by naming the field.

    Also the reason the two flags cannot both be given: this file has a ``messages``
    field and no text field, so the ordinary reader fails on it, which is the correct
    outcome -- a conversation read as text would train on the prompts too.
    """
    chat_corpus(tmp_path, [CONVERSATION])

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "no obvious text field" in str(caught.value)


def test_a_flat_text_corpus_gains_no_chat_counters(tmp_path: Path) -> None:
    """The other half of the control: a guard accidentally inverted fails here."""
    path = tmp_path / "plain.jsonl"
    path.write_text(json.dumps({"text": "just prose"}) + "\n", encoding="utf-8", newline="\n")

    documents, stats = ingest(tmp_path)

    assert [document.trained_spans for document in documents] == [()]
    assert (stats.chat_documents, stats.chat_chars, stats.chat_trained_chars) == (0, 0, 0)


def test_the_two_field_options_cannot_both_be_given() -> None:
    """One says "train on all of this field", the other "train on the replies"."""
    with pytest.raises(UsageError) as caught:
        IngestOptions(jsonl_field="text", jsonl_messages_field="messages")

    assert "cannot both be given" in str(caught.value)
    assert caught.value.hint and "--jsonl-messages-field" in caught.value.hint


def test_the_option_reaches_the_manifest(tmp_path: Path) -> None:
    """It changes what the model is trained on, so it is part of what makes a
    prepared dataset reproducible."""
    recorded = IngestOptions(jsonl_messages_field="messages").to_dict()

    assert recorded["jsonl_messages_field"] == "messages"
    assert IngestOptions().to_dict()["jsonl_messages_field"] is None


def test_a_bad_conversation_names_the_file_the_line_and_the_message(tmp_path: Path) -> None:
    chat_corpus(
        tmp_path,
        [CONVERSATION, [{"role": "user", "content": "hi"}, {"role": "tool", "content": "{}"}]],
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, jsonl_messages_field="messages")

    assert "chat.jsonl line 2" in str(caught.value)
    assert caught.value.details["role"] == "tool"
    assert caught.value.details["line"] == 2


def test_a_missing_messages_field_lists_what_the_record_has(tmp_path: Path) -> None:
    chat_corpus(tmp_path, [CONVERSATION], field="turns")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, jsonl_messages_field="messages")

    assert "'messages' is missing" in str(caught.value)
    assert caught.value.details["available_fields"] == ["turns"]
    # The likeliest cause is a corpus that is flattened already, so say so.
    assert caught.value.hint and "--jsonl-field" in caught.value.hint


def test_on_error_skip_drops_a_bad_conversation_and_counts_it(tmp_path: Path) -> None:
    chat_corpus(tmp_path, [CONVERSATION, [{"role": "user", "content": "nobody replied"}]])

    documents, stats = ingest(tmp_path, jsonl_messages_field="messages", on_error="skip")

    assert [document.text for document in documents] == [FLATTENED]
    assert stats.records_skipped_malformed == 1
    assert stats.chat_documents == 1


def test_a_json_array_of_conversations_works_too(tmp_path: Path) -> None:
    """``.json`` and ``.jsonl`` hold the same records, so they get the same reader."""
    path = tmp_path / "chats.json"
    path.write_text(
        json.dumps([{"messages": CONVERSATION}, {"messages": CONVERSATION}]),
        encoding="utf-8",
        newline="\n",
    )

    documents, stats = ingest(tmp_path, jsonl_messages_field="messages")

    assert [document.text for document in documents] == [FLATTENED, FLATTENED]
    assert stats.chat_documents == 2


def test_a_bad_conversation_in_a_json_array_says_element_not_line(tmp_path: Path) -> None:
    path = tmp_path / "chats.json"
    path.write_text(json.dumps([{"messages": "User: hi"}]), encoding="utf-8", newline="\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, jsonl_messages_field="messages")

    assert "chats.json element 0" in str(caught.value)


def test_a_record_that_is_not_an_object_is_blamed_as_a_record_and_not_as_a_field(
    tmp_path: Path,
) -> None:
    """A file of bare conversations: the message list *is* the line.

    ``--jsonl-messages-field messages`` says the conversation lives in a field of each
    record, so a line that is already the message list has nowhere for that field to
    be. It is the obvious mistake to make -- the flag names a field, and a corpus
    written as one array per line has none -- and the message has to be about the shape
    of the record rather than about ``'messages'``.

    `test_a_missing_messages_field_lists_what_the_record_has` is the neighbouring case,
    and its hint points at ``--jsonl-field`` on the theory that the text is already
    flattened into a string. That advice is wrong for this file: the conversation is
    structured, it is just not wrapped, and flattening it with ``--jsonl-field`` would
    train on the prompts. So the two branches must not share a message, and
    ``available_fields`` -- present in the missing-field error, absent here because
    there are no fields to list -- is what tells them apart in a structured log.
    """
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps(CONVERSATION) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, jsonl_messages_field="messages")

    assert "chat.jsonl line 1: the record is a list, not an object" in str(caught.value)
    assert "'messages' field" in str(caught.value)
    assert caught.value.details["field"] == "messages"
    assert "available_fields" not in caught.value.details
    assert caught.value.hint and "--jsonl-field" not in caught.value.hint


def test_a_conversation_below_the_minimum_length_leaves_no_trace_in_the_chat_counters(
    tmp_path: Path,
) -> None:
    """``--min-doc-chars`` measures the rendered conversation, and it measures it first.

    The counters are the whole assertion. ``chat_documents``, ``chat_chars`` and
    ``chat_trained_chars`` are all incremented after the length check, so a dropped
    conversation must leave nothing behind in any of them -- and the CLI divides the
    last by the second to print the share of the corpus that is actually trained on.
    Counting a conversation that was never emitted inflates the denominator with text
    the trainer never saw, and the ratio stays plausible while being wrong, which is the
    kind of number nobody checks.

    It is counted once, as ``documents_skipped_short``, the same counter a short plain
    document lands in (`test_min_doc_chars_drops_short_documents_and_counts_them` in the
    reader's own tests): a conversation is a document with spans, and the bounds apply
    to it identically. ``records_skipped_malformed`` stays at zero because this record
    was fine -- the two ways ``_chat_document`` returns nothing must not be confused,
    since one of them means the corpus lost something and the other means the corpus was
    filtered as asked.
    """
    short = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    chat_corpus(tmp_path, [CONVERSATION, short])

    documents, stats = ingest(tmp_path, jsonl_messages_field="messages", min_doc_chars=40)

    assert [document.text for document in documents] == [FLATTENED]
    assert stats.documents_skipped_short == 1
    assert stats.records_skipped_malformed == 0
    assert stats.chat_documents == 1
    assert stats.chat_chars == len(FLATTENED)
    assert stats.chat_trained_chars == sum(end - start for start, end in documents[0].trained_spans)


def test_a_dropped_conversation_in_a_json_array_costs_no_ordinal(tmp_path: Path) -> None:
    """The array reader holds its own copy of "this record produced no document".

    ``.json`` and ``.jsonl`` share ``_chat_document`` and not the loop around it, so the
    branch that skips a record it returned nothing for exists twice.
    `test_on_error_skip_drops_a_bad_conversation_and_counts_it` reaches the ``.jsonl``
    copy through the other reason a record is dropped; this reaches the ``.json`` copy
    through this one, and between them each copy has run for each reason.

    Ordinals are what make it worth asserting rather than assuming. They number the
    documents that were emitted, not the records that were read, so the conversation
    after the dropped one is 1: it becomes ``chats.json:1`` in the shard and in every
    error message that cites a document later in the pipeline. A skip that let the
    counter advance would leave ``0, 2`` -- two documents whose locations claim there is
    a third somewhere between them.
    """
    short = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    path = tmp_path / "chats.json"
    path.write_text(
        json.dumps([{"messages": CONVERSATION}, {"messages": short}, {"messages": CONVERSATION}]),
        encoding="utf-8",
        newline="\n",
    )

    documents, stats = ingest(tmp_path, jsonl_messages_field="messages", min_doc_chars=40)

    assert [document.text for document in documents] == [FLATTENED, FLATTENED]
    assert [document.ordinal for document in documents] == [0, 1]
    assert [document.location.split(" ")[0] for document in documents] == [
        "chats.json:0",
        "chats.json:1",
    ]
    assert stats.documents_skipped_short == 1
    assert stats.chat_documents == 2
