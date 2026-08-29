"""Reading a configuration dataclass back out of the JSON a checkpoint or plan holds.

The rule this module exists to enforce: **a key that is present is read, a key that is
absent is a refusal, and a key of the wrong type is a refusal that quotes it.** Filling
a missing key from the dataclass default rebuilds a *different object* than the file
describes, and how that surfaces depends entirely on which key it was. A missing
``n_layer`` eventually fails the weight load -- blaming the weights. A missing
``rope_theta`` changes no tensor shape, so it loads clean and the model emits noise. A
missing ``lr`` or ``seed`` resumes at a different learning rate or a different batch
order, and nothing in the output says so.

What each field accepts is read off the dataclass annotations rather than a table kept
beside them, because a table beside them is a second copy that rots. A field is optional
exactly when its annotation admits ``None`` -- which is the same thing as saying
``__post_init__`` derives it from the others -- so nothing has to list the exemptions.

Two annotations are deliberately *not* type-checked here:

``Literal[...]``
    The values are already validated in ``__post_init__``, whose message names the flag
    to change and lists what is allowed. Checking the JSON type as well would add a
    second, worse refusal for the same mistake.
Anything else
    A field annotated with a type this module does not know is not merely unchecked --
    it is also not *required*, so it would go back to defaulting silently. That is the
    one gap in the design, and ``tests/test_conventions.py`` fails on the day a field
    like that is added rather than leaving it to be discovered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, get_args, get_type_hints

from trainai.errors import ConfigError, json_literal, json_type_name

__all__ = ["checked_fields", "serialised_field_types"]

#: Annotation -> the JSON types that satisfy it. A ``float`` field accepts an integer
#: because ``rope_theta: 10000`` is what a person types where ``to_dict`` writes
#: ``10000.0``; a ``bool`` field does *not* accept ``1``, because JSON spells booleans
#: and these files are JSON. Deliberately not exhaustive over Python: see the module
#: docstring on what an unknown annotation costs, and what asserts it cannot happen.
_ANNOTATION_TYPES: dict[type, tuple[type, ...]] = {
    int: (int,),
    float: (int, float),
    bool: (bool,),
    str: (str,),
}

#: What a field's accepted types are called in a refusal.
_TYPE_NAMES: dict[tuple[type, ...], str] = {
    (int,): "an integer",
    (int, float): "a number",
    (bool,): "a boolean",
    (str,): "a string",
}

#: How to spell an acceptable value, where naming the type does not say enough. The
#: message names what the value is not; the hint has to say what to write instead, and
#: "set it to a boolean" leaves a reader who then types ``"true"`` no better off.
_TYPE_VALUES: dict[tuple[type, ...], str] = {
    (bool,): "true or false",
}


@dataclass(frozen=True)
class SerialisedField:
    """What one field accepts when read from a file.

    ``types`` is ``None`` for a ``Literal`` field: required to be present, with its
    value left to ``__post_init__``, which already refuses it by name.
    """

    types: tuple[type, ...] | None
    optional: bool


@cache
def serialised_field_types(cls: type) -> dict[str, SerialisedField]:
    """Every field of ``cls`` that :func:`checked_fields` knows how to check.

    A field missing from this mapping is one whose annotation is not understood, and is
    therefore neither required nor checked. Cached: it runs on every checkpoint load and
    the answer is fixed at import.
    """
    hints = get_type_hints(cls)
    result: dict[str, SerialisedField] = {}
    for name in cls.__dataclass_fields__:
        parts = get_args(hints[name]) or (hints[name],)
        optional = type(None) in parts
        if any(not isinstance(part, type) for part in parts):
            # A Literal: its arguments are values, not types.
            result[name] = SerialisedField(types=None, optional=optional)
            continue
        types: tuple[type, ...] = ()
        for part in parts:
            for accepted in _ANNOTATION_TYPES.get(part, ()):
                if accepted not in types:
                    types += (accepted,)
        if types:
            result[name] = SerialisedField(types=types, optional=optional)
    return result


def checked_fields(
    cls: type,
    raw: Any,
    *,
    subject: str,
    incomplete_hint: str,
) -> dict[str, Any]:
    """The fields of ``raw`` that ``cls`` declares, or a :class:`ConfigError` saying why not.

    Args:
        cls: The dataclass being rebuilt.
        raw: What was read from the file. Not assumed to be a mapping -- a file holding
            valid JSON that is not an object has no ``.get`` and no ``.items``.
        subject: What to call the thing in a message, e.g. ``"model configuration"``.
            Used as ``A {subject} has to be...`` and ``The {subject}'s lr is...``.
        incomplete_hint: What to tell someone whose file is missing a key. This differs
            per caller because the *consequence* differs: a model rebuilt at the wrong
            shape and a run resumed at the wrong learning rate need different advice.

    Returns:
        A dict safe to splat into ``cls(...)``. Keys ``cls`` does not declare are
        dropped rather than refused, which is what lets a file written by an older
        version load, and is promised in ``docs/plan-format.md``.
    """
    if not isinstance(raw, dict):
        raise ConfigError(
            f"A {subject} has to be an object, and this is {json_type_name(raw)}.",
            hint=(
                f"This is not a {subject}. Re-create it with `trainai train`, which "
                "records the full configuration."
            ),
            details={"found": json_type_name(raw)},
        )

    expected = serialised_field_types(cls)
    missing = sorted(
        name for name, spec in expected.items() if not spec.optional and name not in raw
    )
    if missing:
        raise ConfigError(
            f"The {subject} is missing {and_list(missing)}.",
            hint=incomplete_hint,
            details={"missing": missing},
        )

    for name, spec in expected.items():
        if spec.types is None or name not in raw:
            continue
        value = raw[name]
        if spec.optional and value is None:
            continue
        # bool is an int subclass, so `isinstance` alone would accept true as a width.
        ok = bool in spec.types if isinstance(value, bool) else isinstance(value, spec.types)
        if ok:
            continue
        raise ConfigError(
            f"The {subject}'s {name} is {json_literal(value)}, which is not "
            f"{_TYPE_NAMES[spec.types]}.",
            hint=(
                f"Set {name} to {_TYPE_VALUES.get(spec.types, _TYPE_NAMES[spec.types])}, "
                "or re-create this file with `trainai train`."
            ),
            details={"field": name, "found": json_type_name(value), "value": repr(value)},
        )

    return {key: value for key, value in raw.items() if key in expected}


def and_list(names: Sequence[str]) -> str:
    """``"a"`` / ``"a and b"`` / ``"a, b and c"`` -- for naming what a file lacks."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"
