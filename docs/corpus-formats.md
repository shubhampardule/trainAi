# What TrainAI accepts as a corpus

This document is the input side. For the directory `data prepare` *writes*, see
[dataset-format.md](dataset-format.md).

Point `trainai data prepare` at a single file, at a directory, which is walked
recursively, or at a `.zip`/`.tar.gz` of either. Hidden files and directories
(anything whose name starts with `.`) are skipped, so a `.git` directory inside a
corpus costs nothing.

| Extension | Read as | One document is |
|---|---|---|
| `.txt`, `.text`, `.md`, `.markdown`, `.rst`, `.org`, `.log` | plain text | the whole file, split if long (below) |
| `.jsonl`, `.ndjson` | JSON Lines | one record's text field |
| `.json` | one JSON value | one element of a top-level array |
| `.csv`, `.tsv` | a table | one row, from **one named column** |
| `.docx` | a Word document | the whole document, split if long (below) |
| `.sqlite`, `.sqlite3`, `.db` | a database | one row, from **one named column of one named table** |

Anything else is ignored during a directory walk — but counted and named in a note,
never dropped in silence — and named in the error if it is the only thing you
pointed at. A file with **no extension at all** is a special case: see
[Files with no extension](#files-with-no-extension).

A `.zip` or `.tar` of any of these also works — see [Archives](#archives). The one
exception both ways is a **database**, which has to sit on disk as a plain file:
neither compressed nor inside an archive. [Why](#the-bounds).

## Compressed files

Every extension above except `.sqlite`/`.db` is also accepted with a codec suffix on
the end (`corpus.jsonl.bz2`), decompressed while streaming so a compressed corpus
never has to fit on disk twice:

| Suffix | Codec |
|---|---|
| `.gz` | gzip |
| `.bz2` | bzip2 |
| `.xz`, `.lzma` | LZMA |

The compression is invisible to everything downstream: the same corpus as `.gz`,
`.bz2` and `.xz` yields a byte-identical document stream, so the codec you happen
to have it in cannot change a `content_hash`. `data inspect` names which codec it
found rather than calling everything "gzipped".

`.zst` is **not** supported. Zstandard reached the standard library in Python 3.14
and TrainAI supports 3.10, so reading it would mean a dependency for a suffix — see
the [dependency policy](design/dependencies.md).

### If your Python was built without a codec

`bz2`, `lzma` and `sqlite3` are standard-library modules, but CPython only builds
each one when the matching system library — libbz2, liblzma, libsqlite3 — is present
at compile time. A Python built without one still ships the module's Python half, so
the import fails on the missing C accelerator rather than being absent outright.
This is common enough to have a well-known error message: `No module named '_lzma'`
is a frequent pyenv install failure, and slim container images often lack all three.

TrainAI does not need any of them to start. A plain `.txt` or `.jsonl` corpus
prepares normally on such an interpreter. Only a corpus that actually needs the
missing codec fails, and the error names the library to install:

```
this Python has no 'lzma' module, so TrainAI cannot read that file.
hint: 'lzma' is part of the standard library but is built only when liblzma is
available. Reinstall or rebuild Python with liblzma present, or convert the corpus
to a format that does not need it.
```

`gzip` is not in that list. It needs zlib, which is also required to build a working
pip — so an interpreter without it could not have installed TrainAI in the first
place.

### A corrupt or half-downloaded file

An interrupted download is an ordinary thing to have on disk, so it gets an ordinary
error naming the file:

```
corpus.txt.gz ended before the compressed data did: Compressed file ended before the
end-of-stream marker was reached.
hint: The file is corrupt or was not downloaded completely. Check its size against
the source and fetch it again. TrainAI stops rather than training on the part that
did arrive.
```

Truncation is distinguished from corruption, because they call for different checks:
a truncated file is a download to retry, a scrambled one is a file to re-fetch from
somewhere else.

**This is fatal even under `--on-error skip`, deliberately.** A decompressing stream
does not fail when it is opened — it fails part-way through being read, once it
reaches the bytes that do not decode. By then the documents from earlier in the file
have already been yielded and counted, so skipping the remainder would train on
however much of the corpus happened to arrive and report success. `--on-error skip`
exists for a bad record among good ones; half a file is not that. Every other
corrupt container here — a bad tar, a bad `.docx`, a shredded database — is already
fatal, so this matches them.

Discovery order is sorted and therefore identical on every machine and every run.
That matters more than it looks: document order decides which merges the tokenizer
learns, which decides shard contents, which decides every checksum in the
manifest.

## Archives

A Kaggle download arrives as `.zip`; a research corpus arrives as `.tar.gz`. Point
TrainAI at either one and the files inside it are the corpus — nothing is unpacked
to disk first:

```bash
trainai data prepare papers.tar.gz --out data/papers
```

| Suffix | Read as |
|---|---|
| `.zip` | zip |
| `.tar` | tar |
| `.tar.gz`, `.tgz` | gzipped tar |
| `.tar.bz2`, `.tbz`, `.tbz2` | bzip2 tar |
| `.tar.xz`, `.txz` | LZMA tar |

Members are named `archive::member` everywhere they appear — errors, the report,
the manifest — so a bad line inside an archive points somewhere you can find:

```
papers.zip::2023/notes.jsonl line 41 is not valid JSON.
```

An archive found during a directory walk is expanded where it sat in the sorted
order, so a directory holding `a.txt`, `b.zip` and `c.txt` reads all three.

**Members are read in sorted order, not the order they were stored.** This is not
cosmetic. `namelist()` returns insertion order, which is a property of the tool
that built the archive rather than of what is in it, so without sorting the same
files zipped by two different programs would produce different tokenizer merges and
different shard checksums. Verified: two zips of the same three files added in
opposite orders produce an identical `content_hash` and identical per-shard
`sha256`, as does the same content as a `.tar.gz`.

Sorting has one measurable cost, on a *compressed* tar whose stored order differs
from sorted order — a gzip stream cannot seek backwards, so `tarfile` restarts the
decompression. Measured on a 384-member 1 MiB `.tar.gz`: **1.34 s stored in sorted
order, 1.75 s stored in reverse**. A `.zip` has no such penalty, since a zip is
random-access by design. (An earlier implementation reopened the archive for every
member and took **28 s** on that file; the container is now held open across its
members, which is what the second number is measured against.)

### What is skipped, and named

| Skipped | Why |
|---|---|
| directory entries, symlinks, devices | not files. A zip symlink's *stored bytes are its target path*, so reading it would put a file name into the corpus as a document |
| `__MACOSX/`, and any member whose path has a `.`-prefixed part | the same dotfile rule as a directory walk. A zip made on a macOS desktop is full of AppleDouble resource forks |
| a member TrainAI cannot read | counted and **named** as an `archive_members_ignored` note |
| a `.sqlite`/`.db` member | counted and named the same way. SQLite needs to seek around a real file, so it cannot be read from inside a container — [why](#the-bounds) |
| an archive inside the archive | counted and named as a `nested_archives_ignored` note |

The last three are notes rather than silence on purpose: an archive is opaque, so a
member dropped without a name on screen is a corpus quietly smaller than the one
you pointed at. Pointing at an archive holding *nothing* readable is an error
listing what it does contain.

**One level only.** A zip inside a zip is reported, never opened. That is also the
real defence against the classic 42.zip, whose 4.5 PB exists only through five
levels of nesting.

### Bounds, and one guard that is deliberately absent

Two limits on a single archive, both read from its table of contents so nothing is
decompressed to check them: at most **20,000 members**, and at most **32 GiB**
declared once unpacked. Both protect discovery rather than memory — a million-entry
table of contents becomes a million objects before a single document is read, and a
declared total in the petabytes turns the reported size into nonsense.

There is deliberately **no per-member compression-ratio ceiling**, which is the
usual advice. Measured, gzip reaches **269:1** on legitimately repetitive JSON, so
any threshold low enough to catch a bomb also refuses real corpora. Nothing is ever
extracted to disk here, and every reader is already bounded in memory — text
streams in blocks and emits bounded documents, `.json` has [its own ceiling
measured on the bytes that arrive](#whole-file-json) — so a member that does expand
hugely is read exactly as a large plain file is read. A guard that fires on real
data while protecting against nothing is worse than none, because it advertises
safety it does not provide.

A member's reported size is what it **unpacks to**, not its share of the archive on
disk, because that is the number the reader will see. `data inspect` says so rather
than labelling the total "size on disk".

Members may themselves be compressed (`papers.zip` holding `notes.jsonl.gz` reads
fine) and may be any format in the table above.

## Files with no extension

```bash
trainai data prepare corpus --out data/mine        # read as plain text
```

A corpus in a file called `corpus` used to be a dead end. Point TrainAI at one and
it is read as text; a codec suffix is still honoured, so `corpus.gz` works the same
way. What does *not* qualify is a file with an extension TrainAI does not read —
`scan.pdf` and `release.2` are still refused, because they do carry an extension
and reading past it is how a binary file ends up in a corpus.

**Inside a directory walk they are still not selected**, and this asymmetry is
deliberate. Naming a file is an instruction; finding one in a tree is a guess, and
a stray `README` or `LICENSE` joining the training data is an invisible change to
what the model learns. So the walk's contents are exactly what they were before
this existed, and no prepared dataset shifted.

What did change is that a walk is no longer *silent* about it. Everything it passed
over is counted and named, as a `files_unrecognized` note and a "Passed over" row
in the report, so `Files read 1` out of a directory of three accounts for the other
two:

```
Read
Files read        1  (1.06 MiB)
Documents kept    69
Passed over       2  NOTICE, scan.pdf
```

Reading one is also recorded rather than assumed quietly: an `assumed_text` note
naming the file is written into the dataset's `manifest.json`, because "TrainAI
decided this was text" is not something a dataset should be silent about.

### The binary guard

**UTF-8 decodes a NUL byte happily.** So reading a `.png` as text does not fail —
it succeeds, the tokenizer learns merges from the garbage, and the only trace is a
control-character ratio reported as a neutral statistic. That is the same
looks-like-success failure the [tabular detector](#tabular-data-that-is-not-a-table-on-the-outside)
exists to prevent, so the first block is sniffed and a NUL refuses the file:

```
corpus looks like a binary file: byte 7 is NUL, which text does not contain.
```

It is checked on the bytes that arrive, so a `corpus.gz` of binary is caught too,
and it fires at discovery — before any tokenizer work — rather than part-way
through a run. `--on-error skip` does not apply: there is nothing to fall back to
when the one file you named is not text.

A wide encoding is full of NULs by design — every ASCII character in UTF-16 carries
one — so a UTF-16 byte-order mark, or an explicit `--encoding utf-16-le`, is
recognised first and let through.

The guard is scoped to extensionless files only. Naming a file `.txt` is your
assertion that it is text, and TrainAI does not re-judge it; widening this to every
text file would change what existing corpora do, which is a decision of its own.

## Text files

The file is one document, unless it is longer than `--max-doc-chars` (default
16,384). Long files are cut at the last paragraph break in the window, falling back
to the last line break and then the last space, so cuts land between paragraphs
rather than mid-sentence. A boundary in the first half of the window is ignored —
turning a 16 KiB window into a 200-character document plus a remainder is worse
than an arbitrary cut. No character is dropped either way: concatenating the
documents back together reproduces the file exactly.

`data prepare` reports how many files were split as a `documents_were_split` note.

No markup is stripped from any of these. A `.md` file trains on its `#` and `*`
characters, because deciding what a heading or a code fence is "worth" is a
judgement TrainAI would be making on your behalf, and the model sees the characters
either way.

`.log` is on the list because a log file genuinely is text. That it usually makes a
*narrow* corpus — thousands of near-identical lines — is not guessed at from the
extension: it is measured after tokenization and reported as
`vocabulary_saturated` (below), which is the same check that catches templated text
with no `.log` in sight.

## JSON Lines

One JSON object per line. The text is taken from `--jsonl-field NAME`; without it,
the first of these field names present in the first record wins:

```
text  content  body  document  raw_content
```

If none of them is there, preparation stops and the error lists the fields the
record actually has, because the fix is to pass `--jsonl-field` and you need to
know what to pass it. A line that is not valid JSON is reported with its line
number; `--on-error skip` counts and skips such lines instead, and the count is
surfaced as a `records_skipped` warning rather than swallowed.

## Whole-file JSON

A `.json` file is **one array of records**, and each element is one document. This
is how API dumps and hand-made exports usually arrive:

```json
[{"text": "the first document"}, {"text": "the second"}]
```

Elements may be objects — the text field is resolved exactly as for JSON Lines,
including `--jsonl-field` — or bare strings, in which case the string is the
document. An element with no usable text is reported by its **position in the
array**, not by a line number: a `.json` file has no meaningful lines, and "line
3" would send you hunting through a single 400 KB line. `--on-error skip` works
the same as everywhere else.

A top-level **object** is unwrapped only when the reading is unambiguous:

| Shape | Result |
|---|---|
| `{"data": [...], "count": 2}` | the one array is the documents |
| `{"train": [...], "test": [...]}` | refused, listing both keys |
| `{"text": "...", "tags": [...]}` | refused as ambiguous |
| `{"title": "a", "n": 1}` | refused: no array to read |

The third row is the one worth explaining, because a naive "unwrap the single
array" rule reads it as two documents, `news` and `politics`, and throws away the
actual text. It is equally defensible as one record *or* as a wrapper around many,
so TrainAI names both readings and stops — the same reason `--csv-text-column`
exists rather than a column guesser.

**The whole file is parsed at once**, because JSON has no record separator: there
is no first document until the last byte has been read. That is a real constraint
rather than an implementation shortcut, so it is bounded — a `.json` over **128
MiB** is refused, with the one-liner that converts it to JSON Lines, which streams
with no ceiling at all. The limit is set from measured cost: `json.loads` peaks at
roughly 1.1× the file size for an array of bare strings, 1.3× for records with one
long text field, and 3.9× for wide records of many short fields.

The ceiling is checked against the bytes that actually arrive, not only the size on
disk, because for a compressed file the size on disk bounds nothing: 269:1 on
repetitive JSON is easy to reach, which would put a 128 MiB `.json.gz` at 36 GB
once expanded. Reading stops one block past the limit and the refusal says
"expands to more than", since how much was behind it is unknown.

Unlike JSON Lines and CSV, `.json` **may be UTF-16 or UTF-32**. Those are refused
for the line-oriented formats because records are found by splitting on the newline
byte, which is ambiguous in a wide codec; here nothing is split, so the encoding is
just an encoding.

## CSV and TSV

**One column, named, and nothing is joined.**

```bash
trainai data prepare reviews.csv --out data/reviews --csv-text-column review_text
```

Without `--csv-text-column`, TrainAI looks for a header whose name means "this
column is the text", matched case-insensitively and ignoring surrounding spaces:

```
text  content  body  document  review  review_text  message  comment  abstract  sentence
```

A file with exactly one column uses it. Anything else is an error listing the
columns the file has.

That list is deliberately short, and deliberately excludes plausible-looking names
like `summary`, `description` and `title`. The weather CSV used as this project's
worked example has a `Summary` column; auto-selecting it would silently pick one
column out of twelve and train on 96,000 repetitions of "Partly cloudy". Guessing
wrong here is worse than asking.

**Whole rows are never concatenated into sentences.** Doing that means deciding
what every field means, what unit it is in, and what a blank cell stands for.
Those are domain judgements, they change what the model can learn, and TrainAI has
no business making them invisibly. [`examples/weather_to_text.py`](../examples/weather_to_text.py)
makes them explicitly for one dataset and documents each one; its docstring is the
short version of why this is not a built-in.

**The separator comes from the extension** — `.csv` is a comma, `.tsv` is a tab —
and is never sniffed. A sniffer that guesses wrong yields a single giant column and
a dataset that looks perfectly fine; taking it from the name fails loudly instead,
with an error naming the one absurd column it found. If a `.csv` turns out to hold
semicolons, the error says so specifically and tells you to re-export.

Rows are numbered from 0 as the document ordinal. Quoted fields spanning several
lines are read correctly, and the reported byte offsets still point at the row's
first byte. A row too short to contain the chosen column is reported with its line
number, or counted and skipped under `--on-error skip`.

A row holding **more** fields than the header names is a quieter hazard, so it is
counted rather than dropped. A trailing comma on every line produces this and is
harmless; a comma inside a field that was left unquoted also produces it, and there
the extracted text is only the part before the comma — a truncated document with
nothing else to signal it. TrainAI reads the value it can, and reports the count as
a `csv_rows_with_extra_fields` warning so the truncation is never silent. Quote the
fields that contain the delimiter, or confirm the row count is what you expected.

## Word documents

**`.docx` only, and the text comes out exactly.** A `.docx` is a zip of XML in
which the paragraphs are real elements and the characters inside them are the
characters Word displays. Nothing is reconstructed from a layout, which is the line
that separates this from [`.pdf`](#deliberately-not-supported).

```bash
trainai data prepare thesis.docx --out data/thesis
```

One paragraph becomes one line — the same as Word's own "Save as plain text" — so
an empty paragraph becomes a blank line and paragraph structure survives into the
corpus. Within a paragraph, a line break is a newline, a tab is a tab, and a
non-breaking hyphen is `-`.

**Table cells are read, one per line.** A table's contents are prose the author
typed, and losing it would quietly drop whole sections of a technical document.
This is more than `python-docx` returns from `.paragraphs`, which skips tables
entirely.

What is deliberately left out:

| Left out | Why |
|---|---|
| headers and footers | page furniture. A running header repeats on every page and would enter the corpus dozens of times |
| footnotes, endnotes, comments | out of the reading flow, and each lives in its own part of the package |
| field codes (`w:instrText`) | the *instruction*, not the text: `HYPERLINK "http://example.com"` is markup that happens to be stored as characters |
| text struck out under tracked changes (`w:delText`) | text in the file that is not text in the document. An **insertion** is the mirror case and *is* kept |
| the `mc:Fallback` half of a text box | the same content twice. Word writes a text box as a modern `mc:Choice` and an older fallback saying the same thing; reading both would teach the model to repeat itself |
| pictures, charts, equations | not text. A document of nothing but pictures yields no documents, which is not an error |

Only `word/document.xml` is read, not the whole package — a `.docx` is mostly
images, and on a real 3.2 MB document that part was 189 KB of it. Two bounds apply:
that part must be under **32 MiB** (XML has to be parsed whole, and the parsed form
costs 8–9× the bytes; 32 MiB of Word XML is a book), and a `.docx` that does *not*
sit on disk as a plain file — one inside an archive, or a `.docx.gz` — must be under
**64 MiB**, because reading it means seeking around inside it. Both refusals point
at saving the document as `.txt`, which streams with no limit at all.

A **document type declaration** in any part is refused outright. A DTD is where XML
parsing stops reading a file and starts running a program: the classic "billion
laughs" is 600 bytes that expand to gigabytes, and it would arrive through exactly
this path. The format forbids a DTD anyway, so nothing valid is lost, and the check
cannot misfire on prose — a raw `<` cannot appear in XML character data, so
`&lt;!DOCTYPE` written in a sentence reads back as text.

Two failures get their own message rather than a generic one. A legacy **`.doc`**
renamed to `.docx` is an OLE compound file, not a zip, and so is a
**password-protected** `.docx`; both are named as such instead of being reported as
a corrupt download. And a plain `.zip` renamed to `.docx` lists what it actually
contains and suggests renaming it back, since TrainAI reads zips.

`--encoding` does not apply: the XML declares its own encoding and the parser
honours it.

Finally, `.docx` is classified as a Word document **before** the archive rule, even
though it is technically a zip. Otherwise pointing at one — or having one inside a
`.zip` — would walk it as a container and hand back its XML as documents.

## Databases

**`.sqlite`, `.sqlite3`, `.db` — one row of one column of one table.** The same rule
as a CSV, because a database table is a table.

```bash
trainai data prepare posts.db --out data/posts --db-table posts --csv-text-column body
```

Two names are needed, and both are resolved the same way a CSV column is:

- **The table.** `--db-table` names it, matched ignoring case. A database with
  exactly one table uses it. Anything else is an error listing the tables it has.
- **The column.** `--csv-text-column` names it — the same flag, because it means the
  same thing: *the text column of a table*. The same
  [candidate list](#csv-and-tsv) applies, and a table with exactly one column uses
  it. Anything else is an error listing the columns.

Both choices are printed in the report and recorded in the manifest, so a run whose
table was picked for it says which one.

### Row order is pinned, because SQL has none

**A query has no inherent row order.** A plain `SELECT` comes back in rowid order
today and is free to come back in index order tomorrow, after someone adds an index
— nothing in the file changes, and every checksum in the manifest does. So the order
is written into the query rather than inherited from the engine:

| Table | Ordered by |
|---|---|
| an ordinary table, and a virtual one | its rowid — insertion order |
| a table with a column *called* `rowid` | the next unshadowed spelling, `_rowid_` or `oid` |
| a `WITHOUT ROWID` table | its primary key columns, in declared order |
| neither of those | **refused**, rather than read in whatever order arrives |

A declared column shadows the name it shares, so `ORDER BY rowid` on a table with a
`rowid` column sorts by that column — deterministic, but not the row order. Each
spelling is probed with a `LIMIT 0` query rather than reasoned about.

### Tables, views, and the ones that are not really tables

**A view is never chosen automatically, and naming one is refused.** A view is a
stored query, and a query has no row order to pin — the guarantee above cannot be
made for it. The error says so and suggests the underlying table. A database of
nothing but views names them rather than reporting "no tables".

**A full-text index counts as one table, not six.** An FTS5 table is one virtual
table plus five shadow tables (`_config`, `_content`, `_data`, `_docsize`, `_idx`)
holding its internals. Those are filtered by name prefix, so a database with one
searchable table and nothing else reads without a flag instead of demanding a choice
between six names that mean one thing. SQLite's own `sqlite_%` tables are filtered
the same way.

### Reading does not write

The database is opened **read-only** (`mode=ro`). A training tool has no business
writing to a file it was asked to read, and the default mode would *create* one if
the path were slightly wrong — silently, and empty.

This is measured, not assumed. On a database with an unmerged write-ahead log, a
read-write open reads all 200 rows and **changes the main file's checksum** on close,
by checkpointing the log into it. `mode=ro` reads the same 200 rows and leaves the
file byte-identical. Its one cost: a read-only connection cannot clean up the
`-shm` and `-wal` sidecars, so those are **left behind** next to the database. That
is the right trade — leftover sidecars are inert, a rewritten corpus file is not.

`immutable=1` would be faster still and is deliberately not used: it tells SQLite to
skip recovery, which on that same database means quietly reading stale rows and
reporting success.

One consequence worth knowing about, and it is SQLite's, not TrainAI's: **a `.db`
copied without its `-wal` file can look empty or short.** If a database has fewer
rows than you expect, check whether a `<name>.db-wal` was left behind where it came
from.

### The bounds

- **`.db` is a generic extension**, so the `SQLite format 3` header is verified
  before the file is claimed as a database. A Windows `Thumbs.db` is refused as what
  it is rather than reported as corrupt. The check is a better message, not a
  validation: a valid header over a damaged body passes SQLite's own words on.
- **No archives, and no compression.** SQLite opens a file and seeks around inside
  it — header, then catalogue, then pages scattered through the file — which a
  decompressing stream cannot do. A `.db` inside a `.zip`, or a `.db.gz`, is refused
  with that reason; the fix is to unpack it first. Inside an archive walk the
  database is counted and **named** as an ignored member rather than failing the
  whole archive.
- **A directory walk passes over a database it was not told to read.** Pointing at
  `posts.db` reads it; finding one in a tree does not, because reading it would need
  a table and column named for it and one stray `.db` would otherwise fail the whole
  run. It is counted and named in a note, not dropped in silence. A `.db.gz` in a
  walk is passed over the same way.
- **Memory is not a concern here.** The cursor is streamed, not fetched: walking
  40 MB of text out of a database peaked at 4.5 KB of traced memory, so unlike
  `.json` and `.docx` there is no size ceiling.
- **A Python built without libsqlite3 cannot read any database**, and says so with
  the library named rather than raising `ImportError` on `_sqlite3`. Everything else
  in this document still works on such an interpreter — see
  [if your Python was built without a codec](#if-your-python-was-built-without-a-codec).

A cell that is not text — a number, a blob — is reported with its row number, or
counted and skipped under `--on-error skip`. A `NULL` cell is an empty document, not
a broken one, and falls out through the minimum-length check like an empty CSV field.

`--encoding` does not apply: SQLite stores its text encoding in the file header and
decodes accordingly.

## Encoding

UTF-8 by default. A byte order mark is honoured over that default, so a file saved
by Excel as "UTF-8 with BOM" reads correctly without a flag. `--encoding` takes any
codec Python knows (`--encoding latin-1` covers most Western European text).

A decoding failure names the file and the byte offset of the first bad sequence,
not just the file:

```
broken.txt is not valid utf-8: byte 30 begins the invalid sequence ff.
```

**UTF-16 and UTF-32 are refused for JSON Lines, CSV and TSV.** Records in those
formats are found by splitting on the newline *byte*, which is ambiguous in a wide
codec — every ASCII character contains a zero byte, and the newline appears inside
other characters. TrainAI could produce plausible-looking rubbish here; instead it
stops and asks for the file as UTF-8. Plain text and whole-file `.json` have no
such problem and are read in any encoding.

**`--encoding` does not apply to `.docx`.** An XML part declares its own encoding
and the parser honours the declaration, so there is nothing to override.

## Tabular data that is not a table on the outside

A CSV renamed to `.txt` is not a rare accident. It is the obvious next move for
someone whose CSV was refused, and until this check existed TrainAI prepared it
without a single complaint: the corpus measured 66% digits, which was filed under
"Writing systems" as a neutral statistic, and training produced a model at
perplexity 1.6 that generates flawlessly formatted weather rows with **July
temperatures for a January date**. Fluent form, wrong facts, and nothing in the
pipeline said so until a human read the samples.

So the shape is measured rather than trusted to the extension. What separates a
table from prose is **delimiter regularity**: prose never puts the same number of
commas on every line, and a table always does. The strongest regular shape found,
and what share of the text carries it, is recorded in the prepared dataset's
`manifest.json` under `corpus.rows` whether or not it trips anything:

```json
"rows": {
  "delimiter": ",", "modal_fields": 3,
  "row_share": 0.1453, "sampled_lines": 4028, "sampled_chars": 135103
}
```

(That is the complete works of Shakespeare.) When it does trip, the finding states
the measurement in the message, so the verdict is checkable rather than something
to take on faith.

The share is weighted by **characters, not lines**, because lines are not
equal-sized units. A file of long unwrapped paragraphs with a 22-line option table
pasted into the middle is 96% table by line count and under 1% by character. The
second number is the true one, and it is also the one that matters — characters
become the tokens the model trains on.

Measured through the real pipeline on real corpora:

| Corpus | Share of text on regular rows | Digits |
|---|---|---|
| Weather CSV renamed to `.txt` | **1.00** | 66% |
| A TSV of numbers | **1.00** | 100% |
| A CSV of reviews | **1.00** | 11% |
| A markdown file that is only a table | **1.00** | 14% |
| The same weather data rendered as sentences | **0.97** | 24% |
| Complete works of Shakespeare | 0.15 | 0% |
| This project's Python source | 0.10 | 1% |
| This project's markdown, tables included | 0.08 | 2% |
| Unwrapped paragraphs with an option table in them | 0.00 | 0% |

Nothing legitimate above 0.15, nothing tabular below 0.97. The threshold sits at
**0.80**, and at least 20 lines must have been sampled — a five-line file whose
every line happens to hold two commas is a coincidence, not a measurement.

Crossing that with the digit share tells the two situations apart, because they
need different advice:

**Mostly digits (≥ 40%) — these are measurements.** `data prepare` stops with
`looks_like_a_table` and exit code 3. A language model predicts the next token, and
to a tokenizer `9.47` and `9.48` are unrelated strings; it cannot learn that they
are close numbers. Trained on this it produces rows in exactly the right shape with
values that are confidently wrong. If you want to predict one column from the
others that is regression, and scikit-learn's `HistGradientBoostingRegressor` does
it in seconds on a CPU while reporting its own error.

**Few digits — some column holds real prose.** A warning, `rows_not_prose`, and
preparation continues. The fix is to train on that column alone, which is what
`--csv-text-column` is for; if the file is a renamed spreadsheet export, renaming it
back to `.csv` is the missing step and the message says so.

### Overriding it

```bash
trainai data prepare weather.csv --out data/w --allow-tabular
```

The error becomes a `tabular_override` warning and preparation continues. The
finding is **downgraded, never deleted**: it is written into the dataset's
`manifest.json` under `validation`, so a dataset prepared this way says so
permanently and `data inspect` will show it. Same principle as the export's
"a skipped check is reported as skipped".

If you do this, judge the model by reading its samples rather than by its
validation loss. A table is highly predictable, so the loss will look excellent
while the numbers the model generates are wrong.

### A second, independent signal

Tabular and templated corpora are large but *narrow*, and that shows up again after
tokenization: a byte-level BPE runs out of distinct text to learn merges from and
cannot fill the vocabulary it was asked for. The weather data rendered as sentences
built 698 of 8,192 requested tokens — 9%. Below 25%, `data prepare` reports
`vocabulary_saturated`.

This catches corpora the row measurement lets through, including logs and
templated text that are not delimiter-regular at all. Nothing is broken when it
fires — the model is sized for the tokens that exist — but expect it to reproduce
the corpus rather than generalise from it.

## What a finding does

| Level | Effect |
|---|---|
| `error` | preparation stops, exit code 3, nothing is written |
| `warning` | preparation continues; recorded in the manifest |
| `note` | worth knowing; recorded in the manifest |

Every finding carries a stable `code` you can match on in a script, the
measurement that produced it, and a hint naming a concrete next action. They are
all written to `manifest.json`, so the verdict travels with the dataset instead of
scrolling past in a terminal.

## Deliberately not supported

The rule for what gets a reader is whether extraction is **exact**: either the bytes
already are text, or the format is a typed container with a real text field. Where
text has to be *reconstructed* from a visual layout, extraction becomes a guess —
and a guess that fails by producing plausible-looking wrong text, which is the same
silent failure the [tabular detector](#tabular-data-that-is-not-a-table-on-the-outside)
exists to prevent.

**`.pdf`** is the clearest case. A PDF stores glyphs at positions, not paragraphs,
so reading one means inferring the reading order:

- a two-column paper interleaves into nonsense, line by line
- running heads, page numbers and footers repeat on every page
- a word hyphenated across a line break comes out as two words
- ligatures arrive as single characters (`ﬁ`, `ﬂ`) unless mapped back
- a scanned PDF has no text layer at all and yields nothing

None of that raises an error. It produces a corpus that looks fine and teaches the
model to write scrambled prose, which is worse than a refusal. Supporting `.pdf`
needs an extraction-quality gate designed with measurements behind it, the way the
tabular detector was — that is its own piece of work, not a reader bolted on.

**`.html`** has the same shape of problem: naive tag-stripping yields navigation
menus, cookie banners and inline JavaScript. It needs real boilerplate removal
before it is worth having.

**`.parquet` and `.xlsx`** are different — extraction from both is exact, and both
are worth doing. They need `pyarrow` and `openpyxl`, so they need an optional extra
and a [dependency-policy](design/dependencies.md) entry. That is a decision on its
own, not an omission.

**`.doc` and `.rtf`** are legacy binary formats; `.doc` is recognised well enough to
[say so by name](#word-documents) rather than to read. **`.zst`** is stdlib only from
Python 3.14, and TrainAI supports 3.10 upward.

Nothing here is ever auto-converted, and nothing is sniffed. The extension decides
what a file is, and a table still needs its column named.
