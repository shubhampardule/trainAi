"""Download the Tiny Shakespeare corpus, so there is something to prepare.

About 1.1 MB of text: the smallest thing that produces a model which visibly
learned *something* rather than nothing. It is the standard first corpus for
from-scratch language modelling, which makes it useful for comparing notes.

    python examples/get_tinyshakespeare.py
    trainai data inspect data/corpus
    trainai data prepare data/corpus --out data/shakespeare

Needs network access; nothing else in TrainAI does. Stdlib only, no dependencies.

On the size check: the file's exact bytes are not pinned to a checksum. It lives
in a third-party repository that is free to change it, and a hard-coded digest
would turn "upstream edited a file" into "TrainAI is broken". The size is checked
loosely so that an HTML error page saved as .txt does not sail through, and the
digest of what actually arrived is printed so you can compare it with someone
else's run. Whether it matches is a fact about your download, not a verdict.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DESTINATION = Path("data/corpus/tinyshakespeare.txt")

#: Roughly 1.1 MB. Anything far outside this is not the file we asked for.
EXPECTED_BYTES = 1_115_394
TOLERANCE = 0.25

#: What this file hashed to when the example was written, for comparison only.
#: Not enforced -- see the module docstring.
OBSERVED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

TIMEOUT_SECONDS = 60


def download(url: str, destination: Path) -> bytes:
    print(f"Fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload: bytes = response.read()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not download the corpus: {exc}\n"
            "Check your network connection, or download the file by hand and save it "
            f"as {destination.as_posix()}."
        ) from exc
    return payload


def check(payload: bytes) -> None:
    low = int(EXPECTED_BYTES * (1 - TOLERANCE))
    high = int(EXPECTED_BYTES * (1 + TOLERANCE))
    if not low <= len(payload) <= high:
        raise SystemExit(
            f"Downloaded {len(payload):,} bytes, expected roughly {EXPECTED_BYTES:,}. "
            "That usually means the URL served an error page rather than the corpus. "
            "Nothing was written."
        )
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"The download is not valid UTF-8 ({exc}). Nothing was written.") from exc


def main() -> int:
    if DESTINATION.exists():
        print(f"{DESTINATION.as_posix()} already exists ({DESTINATION.stat().st_size:,} bytes).")
        print("Delete it first if you want to re-download.")
        return 0

    payload = download(URL, DESTINATION)
    check(payload)

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    # write_bytes, not write_text: writing text on Windows would turn every \n into
    # \r\n, changing the size, the checksum, and every token count downstream.
    DESTINATION.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    print(f"Wrote {DESTINATION.as_posix()} ({len(payload):,} bytes)")
    print(f"sha256 {digest}")
    if digest != OBSERVED_SHA256:
        print(
            "Note: this differs from the copy the example was written against "
            f"({OBSERVED_SHA256[:16]}...). Upstream may have edited the file; the "
            "corpus is still perfectly usable, but your token counts will not match "
            "the numbers in the README exactly."
        )
    print()
    print("Next:")
    print("  trainai data inspect data/corpus")
    print("  trainai data prepare data/corpus --out data/shakespeare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
