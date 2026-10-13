# Install

`crashfrontier` requires **Python 3.10** or later.

```bash
python --version
```

Prefer a virtual environment so the install is isolated from the system interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell: `.\.venv\Scripts\Activate.ps1`.

## pip

```bash
python -m pip install crashfrontier
```

CSV files and the default solver stack need nothing further. Add an extra only when you need that format or a fallback solver:

| Extra | What it adds |
|---|---|
| `parquet` | Parquet readers (`pyarrow`) |
| `excel` | `openpyxl` for `.xlsx` and `.xlsm`. `.xls` needs `xlrd`; `.xlsb` needs `pyxlsb`. A missing engine raises `ImportError` for that suffix |
| `solvers` | ECOS and SCS, tried after CLARABEL on physical-measure cone programs |

```bash
python -m pip install 'crashfrontier[parquet,excel,solvers]'
```

Any subset works: `'crashfrontier[parquet]'`, `'crashfrontier[excel]'`, `'crashfrontier[solvers]'`.

## Check the install

```bash
python -c "import crashfrontier; print(crashfrontier.__version__)"
```

A printed version string is enough. Continue with [Market data](data.md) or
run the [self-contained first frontier](first-frontier.md).

If `import crashfrontier` raises `ModuleNotFoundError`, the current `python` is not the environment where the package was installed. Check with `which python` (Windows: `where python`) and run `python -m pip install crashfrontier` in that interpreter.

## Install from a source tree

Use this section only if you are changing the library itself. An editable install does **not** require a market-data folder.

From the directory that contains `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The same extras apply. `dev` adds `pytest` and is only useful when you run the test suite:

```bash
python -m pip install -e ".[parquet,excel,solvers,dev]"
```

Repeat the import check above. Then pass **your** data path to `Dataset.from_directory`; see [Market data](data.md).
