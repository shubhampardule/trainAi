"""Tests for :mod:`trainai.eval.perplexity`.

Two of these carry the module.

``test_the_mean_is_weighted_by_tokens_not_by_batches`` is the one that found a real
bug. ``sequential_batches`` keeps the ragged final batch rather than dropping the tail
of the split, so averaging per-batch losses gives that short batch a full batch's
weight. The trainer did exactly that: on the 1.1 MB Shakespeare validation split, six
sequences out of thirty-eight carried half of the reported number, moving it 0.012 nats
-- and the best checkpoint is chosen on that number.

``test_coverage_is_counted_in_windows_not_tokens`` is the one that found a reporting
error. Scored *target positions* and split *tokens* are different units: a window of
``seq_len + 1`` tokens yields only ``seq_len`` predictions, because the first token is
context and is never predicted. Dividing one by the other reported 95% coverage on a
split that had been scored completely, and the note blamed a partial window for a gap
that arithmetic had invented.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from trainai.data.loader import open_split
from trainai.errors import DatasetError, UsageError
from trainai.eval import evaluate, evaluate_split
from trainai.infer import InferenceSession
from trainai.model.config import ModelConfig
from trainai.train.config import TrainConfig
from trainai.train.loop import Trainer

SEQ_LEN = 64


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory, prepared_dataset: Any) -> Path:
    run_dir = tmp_path_factory.mktemp("eval") / "run"
    Trainer(
        dataset=prepared_dataset,
        model_config=ModelConfig(
            vocab_size=prepared_dataset.vocab_size,
            n_layer=2,
            n_head=4,
            d_model=64,
            seq_len=SEQ_LEN,
        ),
        train_config=TrainConfig(
            steps=4,
            batch_size=4,
            seq_len=SEQ_LEN,
            lr=1e-3,
            warmup_steps=1,
            eval_every=2,
            eval_batches=2,
            checkpoint_every=2,
            log_every=4,
            seed=7,
            device="cpu",
        ),
        run_dir=run_dir,
        quiet=True,
    ).run()
    return run_dir


@pytest.fixture(scope="module")
def session(trained_run: Path) -> InferenceSession:
    return InferenceSession.open(trained_run, device="cpu", precision="fp32")


# The run every "which context?" test needs: built for one length, trained at another.
# `trainai train`'s --seq-len defaults to min(context, 256), so this is not a contrived
# shape -- three of the four presets produce it, and --preset large builds for 1,024 and
# trains at 256. The module's other fixture has the two equal, which is why the defect
# these tests pin survived it.
GAPPED_CONTEXT = SEQ_LEN
GAPPED_TRAINED = SEQ_LEN // 2


@pytest.fixture(scope="module")
def gapped_run(tmp_path_factory: pytest.TempPathFactory, prepared_dataset: Any) -> Path:
    run_dir = tmp_path_factory.mktemp("eval_gap") / "run"
    Trainer(
        dataset=prepared_dataset,
        model_config=ModelConfig(
            vocab_size=prepared_dataset.vocab_size,
            n_layer=2,
            n_head=4,
            d_model=64,
            seq_len=GAPPED_CONTEXT,
        ),
        train_config=TrainConfig(
            steps=4,
            batch_size=4,
            seq_len=GAPPED_TRAINED,
            lr=1e-3,
            warmup_steps=1,
            eval_every=2,
            eval_batches=2,
            checkpoint_every=2,
            log_every=4,
            seed=7,
            device="cpu",
        ),
        run_dir=run_dir,
        quiet=True,
    ).run()
    return run_dir


@pytest.fixture(scope="module")
def gapped_session(gapped_run: Path) -> InferenceSession:
    return InferenceSession.open(gapped_run, device="cpu", precision="fp32")


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def test_a_split_gets_a_loss_and_the_three_ways_of_saying_it(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    result = evaluate_split(session, prepared_dataset, "val")

    assert result.loss > 0
    assert result.perplexity == pytest.approx(math.exp(result.loss))
    assert result.bits_per_token == pytest.approx(result.loss / math.log(2))
    assert result.tokens_scored > 0
    assert result.batches > 0


def test_the_batch_size_does_not_change_the_answer(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """A number that depends on how it was chunked is not a property of the model."""
    one = evaluate_split(session, prepared_dataset, "val", batch_size=1)
    four = evaluate_split(session, prepared_dataset, "val", batch_size=4)

    assert one.tokens_scored == four.tokens_scored
    assert one.loss == pytest.approx(four.loss, abs=1e-5)


def test_the_mean_is_weighted_by_tokens_not_by_batches(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The bug this module was written to avoid, asserted directly.

    The final batch of a split is ragged, and an unweighted mean gives it a full
    batch's weight. The test also asserts the two means *differ* on this fixture, so it
    cannot pass by coincidence.
    """
    batch_size = 16
    batcher, stream = open_split(prepared_dataset, "val", seq_len=SEQ_LEN, batch_size=batch_size)
    try:
        assert batcher.windows % batch_size != 0, (
            f"no ragged final batch at batch_size {batch_size} ({batcher.windows} windows); "
            "this test would be vacuous"
        )
        per_batch: list[tuple[int, float]] = []
        with torch.no_grad():
            for batch in batcher.sequential_batches():
                inputs = torch.from_numpy(batch.inputs).to(session.device).long()
                targets = torch.from_numpy(batch.targets).to(session.device).long()
                _, loss, _ = session.model(inputs, targets)
                assert loss is not None
                per_batch.append((int(targets.numel()), float(loss)))
    finally:
        stream.close()

    result = evaluate_split(session, prepared_dataset, "val", batch_size=batch_size)
    weighted = sum(n * loss for n, loss in per_batch) / sum(n for n, _ in per_batch)
    unweighted = sum(loss for _, loss in per_batch) / len(per_batch)

    assert result.loss == pytest.approx(weighted, abs=1e-9)
    assert weighted != pytest.approx(unweighted, abs=1e-9)


def test_a_batch_size_larger_than_the_split_is_reduced_not_refused(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The batcher refuses this; evaluation must not inherit the refusal.

    A batch holding the same window twice would make a training gradient smaller than
    it looks, which is why ``TokenBatcher`` rejects a batch size above the window
    count. Scoring builds no gradient and walks the split in order, so the only effect
    of the default batch size exceeding a small validation split would be
    ``trainai eval`` failing with no flags at all. The size used is reported, and the
    loss is the same one a fitting batch size gives.
    """
    huge = evaluate_split(session, prepared_dataset, "val", batch_size=100_000)
    fitting = evaluate_split(session, prepared_dataset, "val", batch_size=4)

    assert huge.batch_size == huge.windows_in_split
    assert huge.batch_size < 100_000, "the point of the test is that it was reduced"
    assert huge.batches == 1
    assert huge.loss == pytest.approx(fitting.loss, abs=1e-5)
    assert huge.tokens_scored == fitting.tokens_scored


def test_it_agrees_with_the_trainers_own_validation_number(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The training curve and ``trainai eval`` must be the same measurement.

    Recomputed at the settings the trainer used -- its batch size, its ``eval_batches``
    cap, its precision. If these two ever disagree, one of them is wrong and neither
    says which.
    """
    config = session.train_config
    same = InferenceSession.open(
        session.layout.run_dir or session.layout.checkpoint_path,
        device="cpu",
        precision=config.precision,
    )
    recomputed = evaluate_split(
        same,
        prepared_dataset,
        "val",
        batch_size=config.batch_size,
        max_batches=config.eval_batches,
    )

    assert same.best_val_loss is not None
    assert recomputed.loss == pytest.approx(same.best_val_loss, abs=1e-9)


# --------------------------------------------------------------------------- #
# What it says about its own coverage
# --------------------------------------------------------------------------- #
def test_coverage_is_counted_in_windows_not_tokens(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """Scored positions and split tokens are different units.

    Each window of ``seq_len + 1`` tokens yields ``seq_len`` predictions, so
    ``tokens_scored / tokens_in_split`` can never reach 1.0 even on a complete pass.
    Coverage counts windows, and the tokens genuinely out of reach are reported
    separately.
    """
    result = evaluate_split(session, prepared_dataset, "val")

    assert result.coverage == 1.0
    assert not result.stopped_early
    assert result.tokens_scored == result.windows_scored * result.seq_len
    assert result.tokens_scored < result.tokens_in_split, (
        "if these were equal the units would be the same and this test would be moot"
    )


def test_the_token_accounting_adds_up_exactly(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    for split in ("train", "val"):
        result = evaluate_split(session, prepared_dataset, split)
        inside = result.windows_in_split * (result.seq_len + 1)

        assert inside + result.tokens_outside_windows == result.tokens_in_split
        assert result.tokens_outside_windows >= 0


def test_stopping_early_is_reported_as_such(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """ "Perplexity on the validation split" having covered a fifth of it is the claim
    this field exists to prevent."""
    full = evaluate_split(session, prepared_dataset, "val", batch_size=4)
    short = evaluate_split(session, prepared_dataset, "val", batch_size=4, max_batches=1)

    assert short.stopped_early
    assert short.batches == 1
    assert short.coverage < full.coverage
    assert full.coverage == 1.0


def test_a_full_pass_says_which_tail_it_could_not_reach(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    report = evaluate(session, prepared_dataset, splits=("val",))
    notes = " ".join(report.notes)

    assert "not comparable to a model with a different tokenizer" in notes
    result = report.result_for("val")
    assert result is not None
    if result.tokens_outside_windows:
        assert "do not fill a sequence" in notes
        assert f"{result.tokens_outside_windows:,}" in notes


def test_an_early_stop_note_does_not_blame_the_tail(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    report = evaluate(session, prepared_dataset, splits=("val",), batch_size=4, max_batches=1)
    notes = " ".join(report.notes)

    assert "--max-batches stopped it early" in notes
    assert "do not fill a sequence" not in notes


def test_the_same_corpus_resplit_is_flagged_as_not_held_out(
    session: InferenceSession, prepared_dataset: Any, tmp_path: Path
) -> None:
    """The dangerous near-miss, and the reason the content hash is checked at all.

    Re-preparing the same corpus with a different ``--seed`` moves the train/val
    boundary and leaves the tokenizer fingerprint identical. Every other check here
    passes, and the new ``val`` split holds documents the model was trained on -- so
    the perplexity comes out lower and reads as a better model. Legitimate uses exist
    (scoring on another corpus), so it is flagged rather than refused, and flagged
    first because it changes what every other number means.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents
    from trainai.data.binarize import TOKENIZER_NAME
    from trainai.data.tokenizer import ByteLevelBPE

    corpus = Path(prepared_dataset.root).parent / "corpus"
    ingestor = Ingestor(IngestOptions())
    documents = list(ingestor.documents(ingestor.discover(corpus)))
    resplit = binarize_documents(
        documents,
        tmp_path / "resplit",
        ByteLevelBPE.load(Path(prepared_dataset.root) / TOKENIZER_NAME),
        seed=prepared_dataset.seed + 1,
        val_fraction=prepared_dataset.val_fraction,
        ingest_options=ingestor.options,
    )
    assert resplit.tokenizer_fingerprint == prepared_dataset.tokenizer_fingerprint, (
        "if the tokenizers differed the earlier check would catch it and this "
        "test would not be about the case it claims to be about"
    )
    assert resplit.content_hash != prepared_dataset.content_hash

    report = evaluate(session, resplit, splits=("val",))

    assert not report.dataset_matches_run
    assert report.to_dict()["dataset_matches_run"] is False
    assert report.dataset_check == "mismatch"
    assert "not necessarily held out" in report.notes[0]


def test_the_run_s_own_dataset_is_not_flagged(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The other half: the flag must not fire on the ordinary case."""
    report = evaluate(session, prepared_dataset, splits=("val",))

    assert report.dataset_matches_run
    assert report.dataset_check == "match"
    assert not any("not necessarily held out" in note for note in report.notes)
    assert not any("Could not check" in note for note in report.notes)


def test_a_missing_content_hash_is_reported_as_unknown_not_as_a_match(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The check needs a hash on both sides, and "could not check" is its own answer.

    This used to return the equivalent of ``"match"``: stripping ``content_hash`` from a
    manifest produced output indistinguishable from a verified match, including
    ``"dataset_matches_run": true`` in the JSON a script reads, with the resplit warning
    absent. The comment justifying it cited runs "trained before the checkpoint recorded
    one" -- and no such run has ever existed, since the manifest has written a hash since
    the commit that introduced manifests and the checkpoint since the commit that
    introduced checkpoints. What the branch actually covered was a damaged manifest.

    Note what is *not* asserted: that this is a mismatch. It is not known to be one. The
    dataset here really is the run's own, only unverifiable -- so the numbers are still
    produced and the note says which claim could not be made.
    """
    damaged = dataclasses.replace(prepared_dataset, content_hash="")

    report = evaluate(session, damaged, splits=("val",))

    assert report.dataset_check == "unknown"
    assert report.dataset_matches_run is False, "an unmade check is not a verified match"
    assert report.to_dict()["dataset_check"] == "unknown"
    assert "Could not check" in report.notes[0], "it changes what every number below means"
    assert not any("not necessarily held out" in note for note in report.notes), (
        "asserting a mismatch is as unfounded as asserting a match"
    )
    assert report.result_for("val") is not None, "the numbers are still worth having"


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def test_the_report_carries_what_makes_the_number_comparable(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """A perplexity without its vocabulary size cannot be compared to another one."""
    report = evaluate(session, prepared_dataset, splits=("train", "val"))
    data = report.to_dict()

    assert [s["split"] for s in data["splits"]] == ["train", "val"]
    assert data["model"]["vocab_size"] == prepared_dataset.vocab_size
    assert data["model"]["parameters"] > 0
    assert data["precision"] == session.precision_note
    assert data["step"] == session.step
    assert data["which"] == session.layout.which
    json.dumps(data)  # it has to survive --json


def test_progress_is_reported_per_batch(session: InferenceSession, prepared_dataset: Any) -> None:
    seen: list[tuple[str, int, int]] = []

    evaluate(
        session,
        prepared_dataset,
        splits=("val",),
        batch_size=4,
        on_batch=lambda split, done, tokens: seen.append((split, done, tokens)),
    )

    assert seen
    assert [entry[1] for entry in seen] == list(range(1, len(seen) + 1))
    assert all(entry[0] == "val" for entry in seen)
    assert seen[-1][2] > seen[0][2], "the token count has to climb, not restart"


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_a_context_beyond_what_the_model_was_built_for_is_refused(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """RoPE extrapolates somewhat, but the number stops being comparable.

    This test used to assert the message said "was trained at", which is what the code
    said and was not true: the number it compared against is the context the model was
    *built* for. On this fixture the two happen to be equal, so the assertion passed
    while documenting the wrong quantity -- see the tests below for the run where they
    differ.
    """
    with pytest.raises(UsageError) as caught:
        evaluate_split(session, prepared_dataset, "val", seq_len=SEQ_LEN * 4)

    assert "was built for" in str(caught.value)
    assert caught.value.details["built_for"] == SEQ_LEN
    assert caught.value.details["requested"] == SEQ_LEN * 4


# --------------------------------------------------------------------------- #
# Which context gets scored
#
# Two lengths, and eval used to conflate them. A model is *built* for
# ``model_config.seq_len`` -- the most context it can be handed. It was *trained* at
# ``train_config.seq_len``. Only the second is the quantity the training curve reports,
# so only the second can be compared to it.
# --------------------------------------------------------------------------- #
def test_the_default_context_is_the_one_the_run_trained_at(
    gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """Not the one the model was built for, which is what it used to be."""
    assert gapped_session.model_config.seq_len == GAPPED_CONTEXT
    assert gapped_session.train_config.seq_len == GAPPED_TRAINED

    result = evaluate_split(gapped_session, prepared_dataset, "val")

    assert result.seq_len == GAPPED_TRAINED


def test_the_default_reproduces_the_trainers_own_number_when_the_two_differ(
    gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """The payoff, and the reason the default matters rather than merely being tidier.

    Scoring at the built context instead put a gap between ``trainai eval`` and the
    training curve for every run whose context exceeds its ``--seq-len`` -- which is
    three of the four presets. On the 512/128 run this was measured on, the curve's
    best val loss was 4.8157 and the old default reported 4.8295.
    """
    config = gapped_session.train_config
    recomputed = evaluate_split(
        gapped_session,
        prepared_dataset,
        "val",
        batch_size=config.batch_size,
        max_batches=config.eval_batches,
    )

    assert gapped_session.best_val_loss is not None
    assert recomputed.loss == pytest.approx(gapped_session.best_val_loss, abs=1e-9)


def test_scoring_past_the_trained_context_still_works_and_says_so(
    gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """It is a real thing to want -- it is just not the training curve's number.

    Reachable only by asking for it explicitly now. The note is what stops the result
    being read as comparable, and it has to name both lengths: a reader who sees only
    the one that was scored cannot tell which case they are in.
    """
    report = evaluate(gapped_session, prepared_dataset, splits=("val",), seq_len=GAPPED_CONTEXT)

    scored = report.result_for("val")
    assert scored is not None and scored.seq_len == GAPPED_CONTEXT
    note = next((n for n in report.notes if "extrapolation" in n), None)
    assert note is not None, f"no extrapolation note in {report.notes}"
    assert str(GAPPED_TRAINED) in note
    assert str(GAPPED_CONTEXT) in note


def test_the_refusal_offers_the_comparable_context_not_just_the_largest_legal_one(
    gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """Advising ``--seq-len <built>`` alone sends the user back to the incomparable number."""
    with pytest.raises(UsageError) as caught:
        evaluate_split(gapped_session, prepared_dataset, "val", seq_len=GAPPED_CONTEXT * 4)

    hint = caught.value.hint or ""
    assert f"--seq-len {GAPPED_CONTEXT}" in hint
    assert f"--seq-len {GAPPED_TRAINED}" in hint, "the comparable setting has to be offered"
    assert caught.value.details == {
        "requested": GAPPED_CONTEXT * 4,
        "built_for": GAPPED_CONTEXT,
        "trained_at": GAPPED_TRAINED,
    }


def test_the_report_carries_the_trained_context_separately(
    gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """So a reader can tell "scored at the trained context" from "scored past it"."""
    report = evaluate(gapped_session, prepared_dataset, splits=("val",))

    assert report.trained_seq_len == GAPPED_TRAINED
    assert report.to_dict()["model"]["trained_seq_len"] == GAPPED_TRAINED


def test_a_run_whose_context_matches_its_seq_len_gains_no_note(
    session: InferenceSession, prepared_dataset: Any
) -> None:
    """The negative control. A guard inverted here would annotate every ordinary run."""
    report = evaluate(session, prepared_dataset, splits=("val",))

    assert report.trained_seq_len == SEQ_LEN
    assert not [n for n in report.notes if "extrapolation" in n]


def test_a_seq_len_recorded_above_the_built_context_does_not_become_the_default(
    monkeypatch: pytest.MonkeyPatch, gapped_session: InferenceSession, prepared_dataset: Any
) -> None:
    """A checkpoint can carry a ``train.seq_len`` larger than the model's context.

    Not from ``trainai train``, which refuses that pairing -- but ``from_dict`` accepts
    whatever is on disk, so a default that trusted it would ask for windows the model
    cannot be handed and turn a strange checkpoint into a shape error from inside the
    attention. The clamp is one ``min``; without it this raises instead of scoring.
    """
    overlong = dataclasses.replace(
        gapped_session.train_config, seq_len=gapped_session.model_config.seq_len * 2
    )
    monkeypatch.setattr(gapped_session.checkpoint, "train_config", overlong)
    assert gapped_session.train_config.seq_len > gapped_session.model_config.seq_len

    result = evaluate_split(gapped_session, prepared_dataset, "val")

    assert result.seq_len == gapped_session.model_config.seq_len


def test_a_split_too_small_for_one_window_suggests_a_context_that_works(
    session: InferenceSession, prepared_dataset: Any, tmp_path: Path
) -> None:
    """A split shorter than one window is refused, and the advice can be followed.

    Built from a single short document, because that is the only way to reach this now:
    a batch size above the window count is reduced rather than refused, so the failure
    has to be genuine -- there is no ``seq_len + 1`` tokens to score at all.

    This test used to assert the hint said the split needed ``seq_len + 1`` tokens,
    which pinned a bug rather than a behaviour: the real requirement is
    ``2 * seq_len + 1``, because a window is ``seq_len + 1`` tokens and another
    ``seq_len`` is reserved for the alignment offset. On a 22-token split the panel
    said "needs 129" and then advised, two lines later, that 65 would do. The
    assertion now takes the suggested context and scores with it.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents
    from trainai.data.binarize import TOKENIZER_NAME
    from trainai.data.tokenizer import ByteLevelBPE

    corpus = tmp_path / "one_document"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "A document too short to fill one window.", encoding="utf-8", newline="\n"
    )
    ingestor = Ingestor(IngestOptions())
    tiny = binarize_documents(
        list(ingestor.documents(ingestor.discover(corpus))),
        tmp_path / "prepared",
        ByteLevelBPE.load(Path(prepared_dataset.root) / TOKENIZER_NAME),
        seed=1,
        val_fraction=0.0,
        ingest_options=ingestor.options,
    )

    with pytest.raises(UsageError) as caught:
        evaluate_split(session, tiny, "train", seq_len=SEQ_LEN)

    assert "Cannot evaluate the train split" in str(caught.value)
    assert caught.value.details["split"] == "train"
    # The requirement is stated once, by the loader, and it is the real one.
    assert f"needs {2 * SEQ_LEN + 1}" in str(caught.value)

    suggested = int(re.search(r"--seq-len (\d+)", str(caught.value.hint)).group(1))
    scored = evaluate_split(session, tiny, "train", seq_len=suggested)

    assert scored.windows_scored >= 1
    assert scored.seq_len == suggested
    with pytest.raises(UsageError):
        evaluate_split(session, tiny, "train", seq_len=suggested + 1)


def test_the_hint_is_the_loaders_and_not_a_second_opinion(
    session: InferenceSession, prepared_dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the loader says about a split it refuses is what the user is told.

    The wrapper used to compose its own hint, which is how it came to state the window
    arithmetic differently from the code that enforces it. Pinning the pass-through is
    what stops a second opinion growing back.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise DatasetError("nope.", hint="a very specific instruction", details={})

    monkeypatch.setattr("trainai.eval.perplexity.open_split", refuse)

    with pytest.raises(UsageError) as caught:
        evaluate_split(session, prepared_dataset, "val")

    assert caught.value.hint == "a very specific instruction"


def test_an_unexpected_error_is_not_relabelled_as_a_usage_problem(
    session: InferenceSession, prepared_dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler caught ``Exception``, so any bug came out as advice about split size.

    A ``RuntimeError`` from the loader is not something the user can fix by re-preparing
    at a larger ``--val-fraction``, and saying so would send them to rebuild a dataset
    that is fine.
    """

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("an internal problem")

    monkeypatch.setattr("trainai.eval.perplexity.open_split", explode)

    with pytest.raises(RuntimeError, match="an internal problem"):
        evaluate_split(session, prepared_dataset, "val")


def test_a_batcher_that_yields_nothing_is_reported_not_divided_by(
    session: InferenceSession, prepared_dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero scored positions must not reach the mean, which divides by them.

    Nothing reachable gets here -- ``open_split`` refuses a split with no windows -- so
    the guard is for a future change to the batcher. It is driven with an empty one
    because a guard no test exercises is decoration, and this one carried the same
    wrong token requirement as the message beside it until now.
    """
    from trainai.data.loader import TokenBatcher

    monkeypatch.setattr(TokenBatcher, "sequential_batches", lambda self, **kwargs: iter(()))

    with pytest.raises(UsageError) as caught:
        evaluate_split(session, prepared_dataset, "val")

    assert "produced no batches" in str(caught.value)
    assert f"at least {2 * SEQ_LEN + 1} tokens" in str(caught.value.hint)


def test_a_dataset_the_model_was_not_trained_on_is_refused(
    session: InferenceSession, tmp_path: Path
) -> None:
    """Perplexity over a differently-tokenized corpus is a plausible meaningless float.

    A different check from the tokenizer one in ``infer.session``: that one is about
    whether ids can be decoded at all, this one is about whether the number means
    anything.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents, train_tokenizer

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "text": f"Document {i} about kettles and gantries. Lanterns and "
                    f"harbours recur so the tokenizer has pairs to merge, {i}."
                }
            )
            for i in range(80)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ingestor = Ingestor(IngestOptions())
    documents = list(ingestor.documents(ingestor.discover(corpus)))
    other = binarize_documents(
        documents,
        tmp_path / "prepared",
        train_tokenizer((d.text for d in documents), vocab_size=512),
        seed=1,
        val_fraction=0.2,
        ingest_options=ingestor.options,
    )

    with pytest.raises(UsageError) as caught:
        evaluate(session, other, splits=("val",))

    assert "tokenized differently" in str(caught.value)
    assert (
        caught.value.details["dataset_tokenizer_fingerprint"]
        != caught.value.details["model_tokenizer_fingerprint"]
    )
