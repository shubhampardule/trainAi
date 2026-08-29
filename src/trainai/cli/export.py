"""``trainai export`` -- write a run out as a model directory other tools can load.

Three things shape the output of this command:

**The verification is part of the result, not a debug detail.** Exporting weights is
the one operation where a mistake cannot be noticed later: a wrong tensor name or a
wrong config field produces a directory that loads cleanly and generates plausible
nonsense. So the report lists every check by name, and says which ones did not run.

**A skipped check is printed as skipped.** ``transformers`` is not a dependency of
this project, so on most machines the logits comparison cannot happen. Printing
"verified" with nothing about that would be the overstatement this project exists to
avoid; the yellow "not checked" line is the honest version.

**The export says what it is.** A ``README.md`` goes into the directory stating that
this is a base model that completes text, what it was trained on, and what its loss
was -- because that is the only place the message survives the directory being zipped
and sent to someone else.
"""

from __future__ import annotations

from pathlib import Path

from trainai.console import (
    DASH,
    SPINNER,
    console,
    emit_json,
    fmt_bytes,
    fmt_count,
    fmt_int,
    print_command,
    print_kv,
    rule,
)
from trainai.export import DEFAULT_DTYPE, DEFAULT_FORMAT, ExportResult, export_run
from trainai.infer import DEFAULT_WHICH

__all__ = ["run_export"]


def run_export(
    target: str,
    out: str,
    *,
    export_format: str = DEFAULT_FORMAT,
    dtype: str = DEFAULT_DTYPE,
    which: str = DEFAULT_WHICH,
    tokenizer: str | None = None,
    verify: bool = True,
    force: bool = False,
    json_output: bool = False,
) -> ExportResult:
    """Export a run and print what was written and what was checked."""
    quiet = json_output

    if not quiet:
        rule("Exporting")
        console.print(
            f"[dim]Reading {target}, writing {Path(out).as_posix()} as {export_format}[/]"
        )

    with console.status("exporting and verifying", spinner=SPINNER) if not quiet else _nothing():
        result = export_run(
            target,
            out,
            export_format=export_format,
            dtype=dtype,
            which=which,
            tokenizer=tokenizer,
            verify=verify,
            force=force,
        )

    if quiet:
        emit_json(result.to_dict())
        return result

    _report(result)
    return result


class _nothing:
    """A context manager that does nothing, so ``--json`` shares one code path."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report(result: ExportResult) -> None:
    rule("Written")
    print_kv("Export", _export_rows(result))
    print_kv("Files", _file_rows(result))
    _print_checks(result)
    _print_next_steps(result)


def _export_rows(result: ExportResult) -> list[tuple[str, str]]:
    source = result.source
    loss = source.get("best_val_loss")
    trained = f"step {fmt_int(int(source.get('step', 0)))}"
    if isinstance(loss, (int, float)):
        trained += f"  {DASH}  validation loss {loss:.4f}"
    return [
        ("Directory", f"[bold]{result.out_dir.as_posix()}[/]"),
        ("Format", f"[bold]{result.format}[/]  {DASH}  weights in {result.dtype}"),
        ("Size", f"[bold]{fmt_bytes(result.total_bytes)}[/]  [dim]({len(result.files)} files)[/]"),
        ("From", f"{Path(str(source.get('checkpoint', ''))).name}  [dim]({trained})[/]"),
        ("Model", f"[bold]{fmt_count(result.parameters)}[/] parameters  {DASH}  {_shape(result)}"),
    ]


def _shape(result: ExportResult) -> str:
    model = result.model
    return (
        f"{model.get('n_layer')} layers  {DASH}  {model.get('n_head')} heads  {DASH}  "
        f"d_model {model.get('d_model')}  {DASH}  context "
        f"{fmt_int(int(model.get('seq_len') or 0))}"
    )


def _file_rows(result: ExportResult) -> list[tuple[str, str]]:
    return [(item.name, f"[dim]{fmt_bytes(item.bytes)}[/]") for item in result.files]


def _print_checks(result: ExportResult) -> None:
    """Every check, passed and skipped alike.

    The skipped ones are printed in yellow rather than omitted. "Verified" that
    quietly means "two of three checks ran" is how a broken export ships.
    """
    rows: list[tuple[str, str]] = []
    for check in result.checks:
        if check.ran and check.passed:
            rows.append((check.name, f"[green]ok[/]  [dim]{check.detail}[/]"))
        elif check.ran:
            rows.append((check.name, f"[red]failed[/]  {check.detail}"))
        else:
            rows.append((check.name, f"[yellow]not checked[/]  [dim]{check.detail}[/]"))
    print_kv("Checks", rows, key_width=20)


def _print_next_steps(result: ExportResult) -> None:
    where = result.out_dir.as_posix()
    if result.format == "hf":
        console.print("\n[dim]Load it with:[/]")
        print_command(
            f'transformers.AutoModelForCausalLM.from_pretrained("{where}")',
            style="dim bold",
        )
    else:
        console.print(f"\n[dim]Weights and config are in[/] {where}")
    console.print(
        "[dim]It is a base model: it continues text rather than answering questions. "
        f"See {where}/README.md.[/]"
    )
