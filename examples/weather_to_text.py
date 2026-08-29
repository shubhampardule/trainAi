"""Turn an hourly weather CSV into text TrainAI can train on.

``trainai data prepare`` reads a CSV one named column at a time and will not join
columns into sentences, because doing that means deciding what every field means,
what unit it is in and what a blank cell stands for. This script makes those
decisions explicitly for the columns of the widely circulated "weather history"
hourly dataset: a formatted timestamp, a short summary, a precipitation type,
temperature and apparent temperature in Celsius, humidity, wind speed and
bearing, visibility, cloud cover, pressure, and a per-day summary.

Read this before trusting the output, because the rendering is not neutral --
five decisions here change what the model can learn.

**A constant column is dropped.** ``Loud Cover`` (a typo for Cloud Cover in the
source data) holds ``0.0`` in every row. A column with one value carries no
information, so emitting it would spend tokens teaching the model to copy a
constant.

**Impossible zeros are rendered as unknown, not as values.** Pressure is
``0.0`` in about 1.3% of rows and humidity in a handful more. Sea-level pressure
of zero millibars does not occur on Earth; that is missing data coded as zero. A
model trained on "pressure 0.0 mb" learns it as a real weather condition, and
nothing downstream can tell that it is wrong. Zero wind speed and zero
visibility are kept, because calm air and dense fog are both real.

**Floats are rounded.** The source records ``9.472222222222221`` degrees. Those
digits are artifacts of a Fahrenheit-to-Celsius conversion, not measurement, and
the tokenizer would spend roughly four times the tokens per reading to preserve
noise. One decimal place is more precision than the instrument had.

**Wind bearing becomes a compass point.** ``251.0`` and ``252.0`` are unrelated
strings to a language model. ``WSW`` recurs often enough to be learnable and is
what a person would write.

**No location is added.** This dataset is commonly attributed to one city, but
the file itself contains no location column, so naming one would be putting a
fact into the training data that is not in the source.

Days become documents: one paragraph per calendar day, blank line between them.
:mod:`trainai.data.ingest` cuts long text at the last paragraph break, so this
gives it natural seams, and it removes the 24-fold repetition that emitting the
daily summary on every hourly row would produce.

Usage::

    python examples/weather_to_text.py "examples/weatherHistory-(1).csv" \
        --out data/weather/corpus.txt

``--style rows`` emits cleaned comma-separated rows instead of prose, for
comparing what the model learns from each. Prose is the default because it is
what the tokenizer and the model are built for, and because its output can be
read and judged without a decoder.

The input is the "Weather in Szeged 2006-2016" hourly CSV that circulates on
Kaggle, with the twelve columns listed in ``REQUIRED_COLUMNS``. It is not
included in this repository -- it is 16 MB of third-party data -- so download it
yourself and point the script at it. Any hourly CSV with those column names will
work; anything else will be refused by name rather than silently mis-rendered.

Do read the warning this produces. ``trainai data prepare`` reports
``rows_not_prose`` on the output of this script, and it is right to: one
paragraph per day of comma-separated readings is still structured data wearing
prose clothing. A model trained on it learns the format perfectly and the
numbers not at all. The script exists to make that concrete, not to recommend it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import OrderedDict
from pathlib import Path

#: Sixteen-point compass, starting at north and going clockwise.
COMPASS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

#: Columns whose zero means "not recorded" rather than a measurement of zero.
#: Pressure of 0 mb and humidity of 0 do not occur; wind speed and visibility of
#: 0 do, so they are deliberately absent from this set.
ZERO_MEANS_MISSING = ("Humidity", "Pressure (millibars)")

#: Dropped because it holds a single value in every row of the source file.
CONSTANT_COLUMNS = ("Loud Cover",)

REQUIRED_COLUMNS = (
    "Formatted Date",
    "Summary",
    "Precip Type",
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Daily Summary",
)


def compass_point(degrees: float) -> str:
    """Bearing in degrees to one of sixteen compass points."""
    index = int((degrees % 360.0) / 22.5 + 0.5) % 16
    return COMPASS[index]


def parse_timestamp(raw: str) -> dt.datetime:
    """Parse ``2006-04-01 00:00:00.000 +0200``.

    The offset shifts with daylight saving in this data, which is why a handful
    of days hold 23 or 25 hours. That is preserved rather than corrected: the
    reading happened when it happened.
    """
    return dt.datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S.%f %z")


def optional_number(row: dict[str, str], column: str) -> float | None:
    """A numeric field, or ``None`` where a zero means the value is missing."""
    text = (row.get(column) or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value == 0.0 and column in ZERO_MEANS_MISSING:
        return None
    return value


def number(row: dict[str, str], column: str) -> float | None:
    text = (row.get(column) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def precipitation(row: dict[str, str]) -> str:
    """The precipitation type, as a labelled field rather than as a claim.

    The source value is ``rain``, ``snow`` or the string ``null``. It describes
    what precipitation falls in that period, not that it is falling in that
    hour, so it is written as ``precip rain`` rather than as "it is raining" --
    over 85,000 rows say ``rain``, including plenty of clear ones.
    """
    value = (row.get("Precip Type") or "").strip().lower()
    if not value or value == "null":
        return "precip none recorded"
    return f"precip {value}"


def hourly_prose(row: dict[str, str], stamp: dt.datetime) -> str:
    parts = [f"{stamp:%H:%M}"]
    summary = (row.get("Summary") or "").strip().lower()
    parts.append(summary if summary else "no summary")

    temperature = number(row, "Temperature (C)")
    if temperature is not None:
        parts.append(f"{temperature:.1f}C")
    apparent = number(row, "Apparent Temperature (C)")
    if apparent is not None:
        parts.append(f"feels {apparent:.1f}C")

    humidity = optional_number(row, "Humidity")
    parts.append("humidity unknown" if humidity is None else f"humidity {humidity * 100:.0f}%")

    speed = number(row, "Wind Speed (km/h)")
    bearing = number(row, "Wind Bearing (degrees)")
    if speed is not None and speed == 0.0:
        parts.append("wind calm")
    elif speed is not None and bearing is not None:
        parts.append(f"wind {speed:.1f} km/h {compass_point(bearing)}")
    elif speed is not None:
        parts.append(f"wind {speed:.1f} km/h")

    visibility = number(row, "Visibility (km)")
    if visibility is not None:
        parts.append(f"visibility {visibility:.1f} km")

    pressure = optional_number(row, "Pressure (millibars)")
    parts.append("pressure unknown" if pressure is None else f"pressure {pressure:.1f} mb")

    parts.append(precipitation(row))
    return f"{parts[0]}  " + ", ".join(parts[1:])


def hourly_row(row: dict[str, str], stamp: dt.datetime) -> str:
    """Cleaned comma-separated form, for ``--style rows``."""
    humidity = optional_number(row, "Humidity")
    pressure = optional_number(row, "Pressure (millibars)")
    bearing = number(row, "Wind Bearing (degrees)")
    speed = number(row, "Wind Speed (km/h)")
    temperature = number(row, "Temperature (C)")
    apparent = number(row, "Apparent Temperature (C)")
    visibility = number(row, "Visibility (km)")
    fields = [
        f"{stamp:%H:%M}",
        (row.get("Summary") or "").strip(),
        (row.get("Precip Type") or "").strip().lower().replace("null", ""),
        "" if temperature is None else f"{temperature:.1f}",
        "" if apparent is None else f"{apparent:.1f}",
        "" if humidity is None else f"{humidity:.2f}",
        "" if speed is None else f"{speed:.1f}",
        "" if bearing is None else compass_point(bearing),
        "" if visibility is None else f"{visibility:.1f}",
        "" if pressure is None else f"{pressure:.1f}",
    ]
    return ",".join(fields)


def render(source: Path, destination: Path, style: str) -> dict[str, int]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{source} is missing expected column(s): {', '.join(missing)}\n"
                f"Found: {', '.join(reader.fieldnames or ['<no header>'])}"
            )
        rows = list(reader)

    # Group by calendar day, keeping the first reading for a repeated timestamp.
    # Daylight-saving transitions produce genuine duplicates in this data.
    days: OrderedDict[str, OrderedDict[str, tuple[dt.datetime, dict[str, str]]]] = OrderedDict()
    unparseable = 0
    duplicates = 0
    for row in rows:
        try:
            stamp = parse_timestamp(row["Formatted Date"])
        except (ValueError, KeyError):
            unparseable += 1
            continue
        day = f"{stamp:%Y-%m-%d}"
        key = row["Formatted Date"].strip()
        bucket = days.setdefault(day, OrderedDict())
        if key in bucket:
            duplicates += 1
            continue
        bucket[key] = (stamp, row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    hours_written = 0
    with destination.open("w", encoding="utf-8", newline="\n") as out:
        for day, bucket in days.items():
            entries = sorted(bucket.values(), key=lambda pair: pair[0])
            stamp, first = entries[0]
            weekday = f"{stamp:%A}"
            daily = (first.get("Daily Summary") or "").strip()
            if style == "rows":
                out.write(f"{day},{weekday},{daily}\n")
            else:
                header = f"{day}, {weekday}."
                out.write(f"{header} {daily}\n" if daily else f"{header}\n")
            for stamp, row in entries:
                line = hourly_row(row, stamp) if style == "rows" else hourly_prose(row, stamp)
                out.write(line + "\n")
                hours_written += 1
            out.write("\n")

    return {
        "rows_read": len(rows),
        "hours_written": hours_written,
        "days": len(days),
        "duplicate_timestamps": duplicates,
        "unparseable_timestamps": unparseable,
        "bytes_written": destination.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="the hourly weather CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/weather/corpus.txt"),
        help="where to write the text (default: data/weather/corpus.txt)",
    )
    parser.add_argument(
        "--style",
        choices=("sentences", "rows"),
        default="sentences",
        help="prose (default) or cleaned comma-separated rows",
    )
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        parser.error(f"no such file: {args.csv_path}")

    stats = render(args.csv_path, args.out, args.style)

    print(f"Wrote {args.out}  ({stats['bytes_written'] / (1 << 20):.2f} MiB, style={args.style})")
    print(f"  {stats['rows_read']:,} rows read")
    print(f"  {stats['hours_written']:,} hourly readings written")
    print(f"  {stats['days']:,} days, one paragraph each")
    if stats["duplicate_timestamps"]:
        print(
            f"  {stats['duplicate_timestamps']:,} duplicate timestamp(s) dropped "
            "(daylight-saving transitions)"
        )
    if stats["unparseable_timestamps"]:
        print(f"  {stats['unparseable_timestamps']:,} row(s) skipped: unparseable timestamp")
    print(f"  dropped constant column(s): {', '.join(CONSTANT_COLUMNS)}")
    print("\nNext:")
    print(f"  trainai data prepare {args.out.parent} --out data/weather-prepared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
