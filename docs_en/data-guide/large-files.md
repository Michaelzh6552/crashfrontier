# Large files and the byte-offset date index

`open_option_source` chooses a backend from the **sum of option-file
bytes** and the suffixes. Default threshold
`DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024` (536870912). Override with
`Dataset(..., memory_limit=...)`. Tests use `memory_limit=0` to force the
index even on tiny files.

The index is a local cache the reader writes next to **your** extract. It
is not a vendor file.

## How the backend is chosen

Let `kinds = [file_kind(p) for p in paths]`,
`total_size = sum(st_size)`.

| Condition | `describe()["options"]["kind"]` | Class |
|---|---|---|
| argument is already a DataFrame | `frame` | `FrameSource` |
| `total_size <= memory_limit` (any suffix) | `frame` | whole-table `read_table` then `FrameSource` |
| all uncompressed `text` and `total_size > memory_limit` | `indexed_text` | `IndexedTextSource` |
| all `parquet` and above the threshold | `parquet` | `ParquetSource` (filter on the source date column) |
| all in `{text_gz, sas, stata, excel}` and above the threshold | `chunk_scan` | `ChunkScanSource` |
| mixed suffixes on large files | `SchemaError: mixed large option file types` | — |

So an uncompressed options CSV above the threshold → `indexed_text`. The
same content via `pd.read_csv` then `from_frames` tries to load the whole
table. **Large CSVs should stay on a path / `from_directory`.**

`.csv.gz`, even when large, does not get a byte-offset date index
(compressed streams cannot be sliced by offset). It uses `chunk_scan`; each
day may scan the whole file. Opening logs a warning that chunk scans reread
the entire file and that an uncompressed CSV is preferable
(`test_chunk_scan_source_for_gzip`).

## Default cache directory

When `cache_dir is None`: `paths[0].resolve().parent.parent / "CACHE"`.

| Option path | Default cache |
|---|---|
| `/path/to/market-data/RAW_DATA/options.csv` | `/path/to/market-data/CACHE` |
| `/path/to/market-data/options.csv` (options sit directly under the folder) | **one level above** `/path/to/CACHE` |

A flat directory should pass `cache_dir=` so the index stays next to the
files. CLI `--cache` has the same meaning.

`describe()["cache_dir"]`: if the backend has `_cache_dir`, that value is
used; otherwise the value stored on Dataset. A small-file `FrameSource` has
no `_cache_dir`, so the result equals the `cache_dir` you passed (which
may be `None`).

## Uncompressed text: byte-range index

`IndexedTextSource` does, for each file:

1. `peek_columns` to find the date column’s **index** in the header (0-based; a typical OptionMetrics export has `date` second, `date_column=1`).
2. `sniff_delimiter`.
3. `load_date_index(csv, cache/f"{stem}.index.json", date_column=..., delimiter=...)`.
4. Index keys are **raw date strings** (not parsed). Those keys are then run through `parse_dates(..., dayfirst=)` to build `Timestamp → [(path, raw_key), ...]`.

For `options.csv`, `{stem}.index.json` is `options.index.json`.
If several input files have the same stem, a digest of the resolved path is
added to each cache name so one file cannot reuse another file's ranges.

`DateIndex` fields: `source`, `size`, `mtime`, `delimiter`, `header`
(bytes), `date_column`,
`ranges: dict[str, list[tuple[start, end]]]`.

**Rebuild** (any mismatch rescans the whole file):

- current `st_size` ≠ recorded `size`
- `mtime` differs by more than `1e-6` seconds (`math.isclose`, `rel_tol=0`)
- resolved source path differs
- `date_column` changed (for example you changed `column_map` and the date column moved)
- `delimiter` changed
- JSON is unreadable

If the resolved source, `size`, `mtime`, `date_column`, and `delimiter` all
match, the JSON is **read as-is; the body is not scanned**
(`test_mtime_change_rebuilds_index`).
`os.utime` advanced by 10 seconds forces a rebuild.

Rows for one day that are not contiguous produce **several** byte ranges
for that key (`test_noncontiguous_date_ranges`). `read_rows_by_ranges`
prepends the header to those fragments and `read_csv`s the small CSV.

Index builds show a `tqdm` bar in bytes. The first construction scans the
entire source; runtime depends on disk and file size. When `size` /
`mtime` / `date_column` / `delimiter` already match, the JSON cache is
reused and the body is not scanned. Fetching one day’s cross-section is a
random read of the recorded byte ranges.

The JSON records `size`, `mtime`, `delimiter`, `date_column`, and `ranges`
keyed by the raw date strings in the file. One observation date that is
not contiguous yields several ranges.

Changing `dayfirst` **does not** invalidate the index (keys stay raw
strings) but does change how keys parse. After flipping `dayfirst` on
ambiguous dates, check that `available_dates()` is still the calendar you
want.

## Chunk-scan date cache

`ChunkScanSource` writes each file’s unique dates to `{stem}.dates.json`.
For `options.csv.gz`, `Path.stem` is `options.csv`, so the
filename is `options.csv.dates.json`. Several files opened together that
share a stem get `{stem}-{hash}.dates.json`, where `hash` is 12 hex
characters of the resolved path. A hit requires the stored `source`,
`size`, `mtime` (same `1e-6` tolerance), and `dayfirst`. Fetching a day
still `iter_table`s the whole file.

Excel above the threshold also takes this path, and `iter_table` reads the
whole sheet once — convert to CSV first.

## Parquet

Above the threshold, every raw timestamp that normalizes to the requested
calendar day is kept, and the file is read with an `in` filter on those
values. A small parquet still loads whole into `FrameSource`. Requires
`pyarrow`.

## Force the index on a tiny CSV

`memory_limit=0` shows automatic index build, cache hit, and rebuild after
mtime change. The file is a few rows; the backend is the same as for a
multi-gigabyte uncompressed CSV.

```python
import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from crashfrontier import Dataset
from crashfrontier.data.source import IndexedTextSource, open_option_source
from crashfrontier.data.index import load_date_index

body = pd.DataFrame(
    {
        "date": ["2021-07-21", "2021-07-22", "2021-07-21"],
        "exdate": ["2021-08-04", "2021-08-04", "2021-08-04"],
        "cp_flag": ["P", "C", "C"],
        "strike_price": [3900.0, 4100.0, 4000.0],
        "best_bid": [10.0, 9.0, 8.0],
        "best_offer": [10.2, 9.2, 8.2],
        "volume": [4, 5, 6],
    }
)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    csv_path = tmp / "opt.csv"
    cache = tmp / "CACHE"
    body.to_csv(csv_path, index=False)

    source, mapping = open_option_source(
        csv_path, cache_dir=cache, memory_limit=0
    )
    assert isinstance(source, IndexedTextSource)
    assert source.dates() == [pd.Timestamp("2021-07-21"), pd.Timestamp("2021-07-22")]
    day = source.rows("2021-07-21")
    assert len(day) == 2
    assert list(day.columns) == list(body.columns)

    index_path = cache / "opt.index.json"
    assert index_path.is_file()
    payload = json.loads(index_path.read_text())
    assert payload["size"] == csv_path.stat().st_size
    assert payload["delimiter"] == ","
    assert payload["date_column"] == 0
    assert set(payload["ranges"]) == {"2021-07-21", "2021-07-22"}
    assert len(payload["ranges"]["2021-07-21"]) == 2  # noncontiguous

    # second open should not rewrite (mtime unchanged). Dataset uses the same cache.
    mtime_before = index_path.stat().st_mtime
    ds = Dataset(csv_path, rates=0.0, strike_scale=1.0, cache_dir=cache, memory_limit=0)
    assert ds.describe()["options"]["kind"] == "indexed_text"
    assert index_path.stat().st_mtime == mtime_before

    index = load_date_index(csv_path, index_path, date_column=0, delimiter=",")
    assert index.date_column == 0

    stat = csv_path.stat()
    os.utime(csv_path, (stat.st_mtime + 10, stat.st_mtime + 10))
    open_option_source(csv_path, cache_dir=cache, memory_limit=0)
    rebuilt = json.loads(index_path.read_text())
    assert rebuilt["mtime"] != payload["mtime"]
    assert rebuilt["size"] == payload["size"]
    print("index OK", ds.describe()["options"]["kind"], list(payload["ranges"]))
```

`kind=="indexed_text"`; two ranges on the 21st; second open leaves `mtime`
unchanged; advancing the file time changes the `mtime` field in the
payload.

## Your extract (`CRASHFRONTIER_DATA`)

```python
import os
from pathlib import Path
from crashfrontier import Dataset
from crashfrontier.config import DEFAULT_MEMORY_LIMIT

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
info = ds.describe()
print(info["options"]["kind"], info["cache_dir"])
print(info["options"]["n_dates"], info["options"]["first_date"], info["options"]["last_date"])
print("threshold bytes", DEFAULT_MEMORY_LIMIT)
```

If the uncompressed options CSV is larger than `DEFAULT_MEMORY_LIMIT`,
`kind` is `indexed_text` and `{stem}.index.json` sits under the resolved
`cache_dir`. Do not raise `memory_limit` above the file size “for
simplicity” — that takes the below-threshold branch and loads the whole
table into `FrameSource`.

`crashfrontier index` writes `{stem}.index.json` under the same rules;
`--force` rebuilds. There is no separate `Dataset.build_index` on the
Python side: opening `IndexedTextSource` calls `load_date_index`.

## Several files concatenated

Several uncompressed CSVs with matching canonical column names: date union,
`rows(date)` `concat`s in file order. Variant `date9_split_dollar_novolume`
covers this with `opt_2021a.csv` + `opt_2021b.csv`. Mismatched names:
`SchemaError: option file columns do not match`.
