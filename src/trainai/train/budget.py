"""How much data a run has, relative to how much the model needs.

The measured curve that motivates this module, from a real run on this project's
development machine: a 13.8M-parameter model on a 307,726-token training split,
1,200 steps at 8,192 tokens per step. Validation loss fell until step 300 and then
rose monotonically for the remaining 900 steps.

    step   train      val    epochs
     100  4.8923   4.7679       2.7
     300  3.5191   4.2479       8.0   <- best
     600  1.7555   5.0792      16.0
    1200  0.4058   5.9805      31.9

The training loss reached 0.41, which looks like success and is not: the model had
memorised the corpus. Nothing in the run failed, no error was raised, and the
final checkpoint was worse than the one from step 300.

Two ratios predict this before a single step runs, and both are simple arithmetic
over numbers TrainAI already has:

* **Epochs.** Tokens the run will process divided by tokens in the training split.
  Above a handful, the model is seeing the same text repeatedly and will start
  reproducing it.
* **Tokens per parameter.** Training tokens divided by parameters. The
  compute-optimal figure from the Chinchilla scaling work is about 20; well below
  that, the model has more capacity than the data can constrain.

Neither is a hard rule and neither is used to refuse anything here. They are
reported before the run starts, so the user finds out in advance rather than from
a checkpoint that turned out to be worse than an earlier one. M3's planner uses the
same arithmetic to choose a shape instead of only describing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["EPOCHS_BEFORE_MEMORISING", "TOKENS_PER_PARAMETER_TARGET", "DataBudget"]

#: Roughly the compute-optimal tokens-per-parameter ratio for from-scratch
#: training, from Hoffmann et al. 2022 ("Training Compute-Optimal Large Language
#: Models"). Used as a reference point to describe how data-starved a run is, not
#: as a target to enforce -- the paper's regime is far larger than anything that
#: fits on a consumer GPU, and a small model on little data is still a legitimate
#: thing to train, as long as the user knows what they will get.
TOKENS_PER_PARAMETER_TARGET = 20.0

#: Above this many passes over the training split, expect the model to reproduce
#: training text rather than generalise from it. Chosen from the measurement in the
#: module docstring, where validation loss turned at epoch 8 and was clearly worse
#: by epoch 16 -- so the warning fires with room to spare.
EPOCHS_BEFORE_MEMORISING = 4.0

#: Below this, the run is data-starved enough to say so plainly.
LOW_TOKENS_PER_PARAMETER = 5.0


@dataclass(frozen=True)
class DataBudget:
    """The relationship between a dataset, a model size, and a planned run.

    Every field is arithmetic over measured inputs: the token counts come from the
    dataset manifest, the parameter count from the model's own shapes.
    """

    train_tokens: int
    val_tokens: int
    parameters: int
    non_embedding_parameters: int
    tokens_per_step: int
    steps: int

    @property
    def tokens_processed(self) -> int:
        return self.tokens_per_step * self.steps

    @property
    def epochs(self) -> float:
        """Passes over the training split. Fractional below one."""
        if self.train_tokens <= 0:
            return 0.0
        return self.tokens_processed / self.train_tokens

    @property
    def tokens_per_parameter(self) -> float:
        """Training tokens available per parameter, counting each token once."""
        if self.parameters <= 0:
            return 0.0
        return self.train_tokens / self.parameters

    @property
    def steps_per_epoch(self) -> float:
        if self.tokens_per_step <= 0:
            return 0.0
        return self.train_tokens / self.tokens_per_step

    @property
    def compute_optimal_tokens(self) -> int:
        """Tokens this model size would want at the reference ratio."""
        return int(self.parameters * TOKENS_PER_PARAMETER_TARGET)

    @property
    def is_data_limited(self) -> bool:
        return self.tokens_per_parameter < LOW_TOKENS_PER_PARAMETER

    @property
    def will_memorise(self) -> bool:
        return self.epochs > EPOCHS_BEFORE_MEMORISING

    def warnings(self) -> list[str]:
        """Plain-language notes about what this run will and will not produce.

        Empty when there is nothing to say. Each entry quotes the number it is
        based on, so the user can disagree with the judgement while still having
        the measurement.
        """
        notes: list[str] = []
        if self.will_memorise:
            notes.append(
                f"This run makes {self.epochs:.1f} passes over the training split "
                f"({self.train_tokens:,} tokens). Past about "
                f"{EPOCHS_BEFORE_MEMORISING:.0f}, validation loss usually turns "
                "around and starts rising while training loss keeps falling: the "
                "model is reproducing the text rather than learning from it. The "
                f"best checkpoint will probably be from around step "
                f"{max(1, int(EPOCHS_BEFORE_MEMORISING * 2 * self.steps_per_epoch))}, "
                "not the last one. Use fewer --steps, or more text."
            )
        if self.is_data_limited:
            notes.append(
                f"{self.tokens_per_parameter:.2f} training tokens per parameter "
                f"({self.train_tokens:,} tokens, "
                f"{self.parameters:,} parameters). From-scratch training usually "
                f"wants roughly {TOKENS_PER_PARAMETER_TARGET:.0f}, which for this "
                f"model size would be about {self.compute_optimal_tokens:,} tokens. "
                "Expect fluent-looking text in the shape of your corpus, and facts "
                "that are wrong. A smaller model would generalise better on this "
                "much data."
            )
        if self.val_tokens == 0:
            notes.append(
                "There is no validation split, so this run cannot tell you whether "
                "the model is generalising or memorising. Re-prepare the dataset "
                "with a non-zero --val-fraction."
            )
        return notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
            "parameters": self.parameters,
            "non_embedding_parameters": self.non_embedding_parameters,
            "tokens_processed": self.tokens_processed,
            "epochs": round(self.epochs, 3),
            "steps_per_epoch": round(self.steps_per_epoch, 2),
            "tokens_per_parameter": round(self.tokens_per_parameter, 4),
            "compute_optimal_tokens": self.compute_optimal_tokens,
            "data_limited": self.is_data_limited,
            "will_memorise": self.will_memorise,
        }

    def describe(self) -> str:
        return (
            f"{self.train_tokens:,} train tokens, {self.epochs:.1f} epochs, "
            f"{self.tokens_per_parameter:.2f} tokens/parameter"
        )
