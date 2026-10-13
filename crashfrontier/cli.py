from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from crashfrontier.config import (
    DEFAULT_LM_RANGE,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_P0,
    DEFAULT_SERIES_TARGET,
    DEFAULT_SERIES_WINDOW,
    DEFAULT_TENORS,
    DEFAULT_THRESHOLD,
)
from crashfrontier.data.dataset import Dataset
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.index import build_date_index
from crashfrontier.data.parse import parse_dates
from crashfrontier.data.readers import file_kind, peek_columns, sniff_delimiter
from crashfrontier.data.schema import ColumnMap, normalize_name
from crashfrontier.data import source as source_mod
from crashfrontier.data.source import classify_table
from crashfrontier.errors import CrashFrontierError
from crashfrontier.export import write_frontier
from crashfrontier.data.spot import SpotTable
from crashfrontier.frontier import (
    binding_quotes,
    compute_frontier,
    nested_frontiers,
    scenario_check,
)
from crashfrontier.series import frontier_series, term_structure

__all__ = ["main"]

logger = logging.getLogger("crashfrontier")

_MTIME_ABS_TOL = 1e-6
_DEFAULT_DATA_DIR = Path("DATA")
_INDEX_HINT = "date index does not exist; run crashfrontier index"
_INDEX_STALE_HINT = "date index is stale; run crashfrontier index --force"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a crashfrontier subcommand."""
    standalone = argv is None
    args_list = sys.argv[1:] if argv is None else list(argv)
    try:
        code = _run(args_list)
    except SystemExit as exc:
        code = _exit_code(exc.code)
    if standalone:
        raise SystemExit(code)
    return int(code)


def _run(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    debug = _configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except (CrashFrontierError, ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        return 1


def _build_parser() -> argparse.ArgumentParser:
    shared = _ArgumentParser(add_help=False)
    _add_global_options(shared)
    parser = _ArgumentParser(
        prog="crashfrontier",
        description="Crash frontiers from option bid-ask quotes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_p = subparsers.add_parser("inspect", parents=[shared], help="print dataset summary")
    inspect_p.set_defaults(handler=_cmd_inspect)

    index_p = subparsers.add_parser("index", parents=[shared], help="build option date indexes")
    index_p.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    index_p.set_defaults(handler=_cmd_index)

    expiries_p = subparsers.add_parser("expiries", parents=[shared], help="list expiries for a date")
    expiries_p.add_argument("date")
    expiries_p.add_argument("--lm-range", nargs=2, type=float, metavar=("LO", "HI"))
    expiries_p.set_defaults(handler=_cmd_expiries)

    select = _ArgumentParser(add_help=False)
    _add_selection_options(select)

    frontier_p = subparsers.add_parser(
        "frontier",
        parents=[shared, select],
        help="compute and plot a crash frontier",
    )
    _add_frontier_options(frontier_p)
    frontier_p.set_defaults(handler=_cmd_frontier)

    scenario_p = subparsers.add_parser(
        "scenario",
        parents=[shared, select],
        help="check a (prob, loss) scenario",
    )
    scenario_p.add_argument("--K", type=float, default=DEFAULT_THRESHOLD)
    scenario_p.add_argument("--gamma", type=int, choices=(0, 1, 2), default=0)
    scenario_p.add_argument("--prob", type=float, required=True)
    scenario_p.add_argument("--loss", type=float, required=True)
    scenario_p.add_argument("--quote-set")
    scenario_p.set_defaults(handler=_cmd_scenario)

    binding_p = subparsers.add_parser(
        "binding",
        parents=[shared, select],
        help="plot binding quotes for a support direction",
    )
    binding_p.add_argument("--K", type=float, default=DEFAULT_THRESHOLD)
    binding_p.add_argument("--direction", default="1,0")
    binding_p.add_argument("--quote-set")
    binding_p.add_argument("--out")
    binding_p.add_argument("--formats", default="pdf,png")
    binding_p.set_defaults(handler=_cmd_binding)

    term_p = subparsers.add_parser(
        "term",
        parents=[shared],
        help="term structure of crash frontiers on one date",
    )
    term_p.add_argument("--date", required=True)
    term_p.add_argument("--tenors", default=",".join(str(item) for item in DEFAULT_TENORS))
    term_p.add_argument("--K", type=float, default=DEFAULT_THRESHOLD)
    term_p.add_argument("--gamma", type=int, choices=(0, 1, 2), default=0)
    term_p.add_argument("--am-settlement", type=int, choices=(0, 1))
    term_p.add_argument("--quote-set")
    term_p.add_argument(
        "--support-cap",
        default=argparse.SUPPRESS,
        type=_parse_support_cap,
        metavar="CAP",
    )
    term_p.add_argument("--out")
    term_p.add_argument("--formats", default="pdf,png")
    term_p.add_argument("--export")
    term_p.set_defaults(handler=_cmd_term)

    series_p = subparsers.add_parser(
        "series",
        parents=[shared],
        help="crash-frontier time series over a date range",
    )
    series_p.add_argument("--start")
    series_p.add_argument("--end")
    series_p.add_argument("--dates")
    series_p.add_argument("--freq", choices=("D", "W", "M", "Q"), default="W")
    series_p.add_argument("--target", type=int, default=DEFAULT_SERIES_TARGET)
    series_p.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"))
    series_p.add_argument("--days", type=int)
    series_p.add_argument("--K", type=float, default=DEFAULT_THRESHOLD)
    series_p.add_argument("--gamma", type=int, choices=(0, 1, 2), default=0)
    series_p.add_argument("--quote-set")
    series_p.add_argument(
        "--support-cap",
        default=argparse.SUPPRESS,
        type=_parse_support_cap,
        metavar="CAP",
    )
    series_p.add_argument("--am-settlement", type=int, choices=(0, 1))
    series_p.add_argument("--workers", type=int, default=1)
    series_p.add_argument(
        "--resume",
        help="checkpoint CSV for frontier_series; not the index --cache directory",
    )
    series_p.add_argument("--out", required=True)
    series_p.add_argument("--overlay")
    series_p.add_argument("--formats", default="pdf,png")
    series_p.set_defaults(handler=_cmd_series)
    return parser


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", help="data directory (default DATA if it exists)")
    parser.add_argument("--options", nargs="+", metavar="PATH", help="option table paths")
    parser.add_argument("--forwards", help="forward table path")
    parser.add_argument("--rates", help="rate table path or constant")
    parser.add_argument("--spot", help="spot table path")
    parser.add_argument("--strike-scale", default="auto", help="auto or a positive scale")
    parser.add_argument("--dayfirst", action="store_true", help="prefer day-first dates")
    parser.add_argument(
        "--rate-unit",
        choices=("percent", "decimal"),
        default="percent",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        metavar="NAME=RAW",
        help="canonical-to-raw column override; repeatable",
    )
    parser.add_argument("--cache", help="index cache directory")
    parser.add_argument("--no-volume-filter", action="store_true")
    parser.add_argument("--min-volume", type=float)
    parser.add_argument("--log-level", default="INFO")


def _add_selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", required=True)
    parser.add_argument("--expiry")
    parser.add_argument("--days", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"))
    parser.add_argument("--am-settlement", type=int, choices=(0, 1))


def _add_frontier_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--K", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--gamma", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--support-cap",
        default=argparse.SUPPRESS,
        type=_parse_support_cap,
        metavar="CAP",
        help="support upper bound (default: 5 for gamma>0, none for gamma=0; 'none' for unbounded)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quote-set")
    group.add_argument("--nested", action="store_true")
    parser.add_argument("--sets", help="comma-separated nested quote-set names")
    parser.add_argument("--method", choices=("auto", "hull", "grid"), default="auto")
    parser.add_argument("--mode", choices=("auto", "hard", "relaxed"), default="auto")
    parser.add_argument("--p0", type=float, default=DEFAULT_P0)
    parser.add_argument("--layout", choices=("panels", "report"), default="panels")
    parser.add_argument("--out")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--export")


def _cmd_inspect(args: argparse.Namespace) -> int:
    option_paths = _option_paths(args)
    state = _index_state(args, option_paths)
    if state != "ok" and _needs_byte_index(option_paths):
        _print_partial_inspect(args, option_paths)
        hint = _INDEX_HINT if state == "missing" else _INDEX_STALE_HINT
        print(hint)
        return 0
    dataset = _open_dataset(args)
    _print_mapping(dataset.describe())
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    option_paths = _option_paths(args)
    cache_dir = _cache_dir(args, option_paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    column_map = _column_map(args, option_paths)
    dayfirst = _dayfirst(args)
    cache_names = _index_cache_names(option_paths)
    for path in option_paths:
        if file_kind(path) != "text":
            logger.info("skipping non-text option file %s", path)
            continue
        date_column = _date_column_index(path, column_map)
        delimiter = sniff_delimiter(path)
        index_path = cache_dir / cache_names[path]
        current = (
            _index_file_state(
                path, index_path, date_column, delimiter, dayfirst=dayfirst
            )
            == "ok"
        )
        if args.force or not current:
            index = build_date_index(
                path, index_path, date_column=date_column, delimiter=delimiter
            )
            action = "wrote"
        else:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            n_dates = len(payload.get("ranges", {}))
            print(f"index already current: {index_path} ({n_dates} dates)")
            continue
        print(f"{action} {index_path} ({len(index.ranges)} dates)")
    return 0


def _cmd_expiries(args: argparse.Namespace) -> int:
    dataset = _open_dataset(args)
    stamp = _parse_date(args.date, dayfirst=_dayfirst(args))
    lm_range = _lm_range(args)
    table = dataset.expiries(stamp, lm_range=lm_range)
    display = table.copy()
    if not display.empty:
        display["expiry"] = pd.to_datetime(display["expiry"]).dt.strftime("%Y-%m-%d")
    print(display.to_string(index=False))
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    if args.sets and not args.nested:
        raise ValueError("--sets requires --nested")
    if args.export and not args.out:
        raise ValueError("--export requires --out")
    dataset = _open_dataset(args)
    section = _cross_section(dataset, args)
    threshold = float(args.K)
    gamma = int(args.gamma)
    options: dict[str, object] = {
        "gamma": gamma,
        "method": args.method,
        "mode": args.mode,
        "p0": float(args.p0),
    }
    if hasattr(args, "support_cap"):
        options["support_cap"] = args.support_cap
    certificate = None
    if args.nested:
        sets = _csv_list(args.sets) if args.sets else None
        fronts = nested_frontiers(section, threshold, sets=sets, **options)
        named = fronts
    else:
        if args.quote_set is not None:
            options["quote_set"] = args.quote_set
        frontier = compute_frontier(section, threshold, **options)
        fronts = frontier
        named = {frontier.quote_set: frontier}
    for frontier in named.values():
        print(frontier.describe())
        print()
    if args.layout == "report" and gamma == 0:
        binding_options = dict(options)
        binding_options.pop("method", None)
        binding_options.pop("mode", None)
        binding_options.pop("p0", None)
        binding_options.pop("gamma", None)
        if args.quote_set is not None:
            binding_options["quote_set"] = args.quote_set
        certificate = binding_quotes(
            section, threshold, direction=(1.0, 0.0), **binding_options
        )
    if args.out:
        from crashfrontier.plotting import plot_frontier_panels, plot_report, save_figure

        if args.layout == "report":
            figure = plot_report(section, fronts, certificate=certificate)
        else:
            figure = plot_frontier_panels(fronts)
        save_figure(figure, args.out, formats=_csv_list(args.formats))
        import matplotlib.pyplot as plt

        plt.close(figure)
        if args.export:
            write_frontier(
                fronts,
                args.out,
                formats=_csv_list(args.export),
                certificate=certificate,
            )
    return 0


def _cmd_scenario(args: argparse.Namespace) -> int:
    dataset = _open_dataset(args)
    section = _cross_section(dataset, args)
    options = {"gamma": int(args.gamma)}
    if args.quote_set is not None:
        options["quote_set"] = args.quote_set
    result = scenario_check(
        section,
        float(args.K),
        float(args.prob),
        float(args.loss),
        **options,
    )
    print(f"feasible: {bool(result['feasible'])}")
    print(f"geometry: {result['geometry']}")
    print(f"method: {result['method']}")
    return 0


def _cmd_binding(args: argparse.Namespace) -> int:
    dataset = _open_dataset(args)
    section = _cross_section(dataset, args)
    options = {}
    if args.quote_set is not None:
        options["quote_set"] = args.quote_set
    certificate = binding_quotes(
        section,
        float(args.K),
        direction=_parse_direction(args.direction),
        **options,
    )
    print(certificate.to_frame().to_string(index=False))
    if args.out:
        from crashfrontier.plotting import plot_binding_quotes, save_figure
        import matplotlib.pyplot as plt

        figure = plot_binding_quotes(certificate)
        save_figure(figure, args.out, formats=_csv_list(args.formats))
        plt.close(figure)
    return 0


def _cmd_term(args: argparse.Namespace) -> int:
    if args.export and not args.out:
        raise ValueError("--export requires --out")
    dataset = _open_dataset(args)
    stamp = _parse_date(args.date, dayfirst=_dayfirst(args))
    options = _compute_options(args)
    table, fronts = term_structure(
        dataset,
        stamp,
        tenors=_int_csv_list(args.tenors),
        K=float(args.K),
        gamma=int(args.gamma),
        am_settlement=args.am_settlement,
        **options,
    )
    print(_display_table(table).to_string(index=False))
    if args.out:
        from crashfrontier.plotting import plot_term_structure, save_figure
        import matplotlib.pyplot as plt

        figure = plot_term_structure(table)
        save_figure(figure, args.out, formats=_csv_list(args.formats))
        plt.close(figure)
        if args.export and "csv" in _csv_list(args.export):
            Path(f"{args.out}.csv").parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(f"{args.out}.csv", index=False)
        if fronts and args.export and "json" in _csv_list(args.export):
            write_frontier(fronts, args.out, formats=("json",))
    return 0


def _cmd_series(args: argparse.Namespace) -> int:
    if args.days is not None and args.window is not None:
        raise ValueError("--window cannot be combined with --days")
    if args.dates is not None and (args.start is not None or args.end is not None):
        raise ValueError("pass either --dates or --start/--end, not both")
    if args.dates is None and (args.start is None or args.end is None):
        raise ValueError("--start and --end are required unless --dates is given")
    dataset = _open_dataset(args)
    options = _compute_options(args)
    window = DEFAULT_SERIES_WINDOW if args.window is None else (int(args.window[0]), int(args.window[1]))
    series_kwargs: dict[str, object] = {
        "K": float(args.K),
        "gamma": int(args.gamma),
        "quote_set": args.quote_set,
        "am_settlement": args.am_settlement,
        "workers": int(args.workers),
        "cache_path": None if args.resume is None else Path(args.resume),
        **options,
    }
    if args.days is not None:
        series_kwargs["days"] = int(args.days)
        series_kwargs["target_days"] = None
        series_kwargs["window"] = None
    else:
        series_kwargs["target_days"] = int(args.target)
        series_kwargs["window"] = window
    if args.dates is not None:
        table = frontier_series(
            dataset,
            _read_dates_file(Path(args.dates), dayfirst=_dayfirst(args)),
            **series_kwargs,
        )
    else:
        table = frontier_series(
            dataset,
            start=_parse_date(args.start, dayfirst=_dayfirst(args)),
            end=_parse_date(args.end, dayfirst=_dayfirst(args)),
            freq=args.freq,
            **series_kwargs,
        )
    print(_display_table(table).to_string(index=False))
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(f"{stem}.csv"), index=False)
    from crashfrontier.plotting import plot_time_series, save_figure
    import matplotlib.pyplot as plt

    overlay = None if args.overlay is None else _overlay_series(args.overlay, table)
    overlay_label = None if args.overlay is None else Path(args.overlay).stem
    for value in ("prob", "loss", "depth"):
        figure = plot_time_series(
            table, value=value, overlay=overlay, overlay_label=overlay_label
        )
        save_figure(figure, Path(f"{stem}_{value}"), formats=_csv_list(args.formats))
        plt.close(figure)
    return 0


def _compute_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    if getattr(args, "quote_set", None) is not None:
        options["quote_set"] = args.quote_set
    if hasattr(args, "support_cap"):
        options["support_cap"] = args.support_cap
    return options


def _int_csv_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv_list(text))


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for name in ("date", "expiry", "requested_date"):
        if name not in display.columns:
            continue
        stamps = pd.to_datetime(display[name], errors="coerce")
        display[name] = stamps.dt.strftime("%Y-%m-%d").where(stamps.notna(), "")
    return display


def _read_dates_file(path: Path, *, dayfirst: bool | None) -> list[pd.Timestamp]:
    path = Path(path)
    raw = _read_dates_table(path, header=None)
    if raw is not None and not raw.empty and _token_might_be_date(raw.iloc[0, 0]):
        parsed = _parse_date_column(raw.iloc[:, 0], dayfirst=dayfirst)
        if parsed is not None and not parsed.empty and bool(parsed.notna().all()):
            return _normalized_dates(parsed)
    table = _read_dates_table(path, header=0)
    if table is None or table.empty:
        raise ValueError(f"could not parse dates in {path}")
    parsed = _parse_date_column(table.iloc[:, 0], dayfirst=dayfirst)
    if parsed is None or parsed.empty or bool(parsed.isna().any()):
        raise ValueError(f"could not parse dates in {path}")
    return _normalized_dates(parsed)


def _read_dates_table(path: Path, *, header: int | None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, header=header)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return None


def _token_might_be_date(value) -> bool:
    text = str(value).strip()
    return any(character.isdigit() for character in text)


def _parse_date_column(values: pd.Series, *, dayfirst: bool | None) -> pd.Series | None:
    try:
        return parse_dates(values, dayfirst=dayfirst)
    except (CrashFrontierError, ValueError, TypeError):
        return None


def _normalized_dates(parsed: pd.Series) -> list[pd.Timestamp]:
    stamps: list[pd.Timestamp] = []
    for item in parsed:
        stamp = pd.Timestamp(item)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        stamps.append(stamp.normalize())
    return stamps


def _overlay_series(path: str, table: pd.DataFrame) -> pd.Series:
    spot = SpotTable.from_path(path)
    if "date" in table.columns:
        stamps = pd.to_datetime(table["date"], errors="coerce")
    else:
        stamps = pd.Series(pd.NaT, index=table.index)
    if "requested_date" in table.columns:
        stamps = stamps.fillna(pd.to_datetime(table["requested_date"], errors="coerce"))
    values = [
        float("nan") if pd.isna(stamp) else float(spot.spot(stamp)) for stamp in stamps
    ]
    return pd.Series(values, index=pd.DatetimeIndex(stamps))


def _open_dataset(args: argparse.Namespace) -> Dataset:
    option_paths = _option_paths(args)
    if _needs_byte_index(option_paths):
        state = _index_state(args, option_paths)
        if state != "ok":
            hint = _INDEX_HINT if state == "missing" else _INDEX_STALE_HINT
            raise ValueError(hint)
    kwargs = _dataset_kwargs(args)
    if args.options:
        return Dataset(option_paths, **kwargs)
    return Dataset.from_directory(_data_root(args), **kwargs)


def _dataset_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict[str, object] = {
        "column_map": _column_overrides(args.column),
        "strike_scale": _strike_scale(args.strike_scale),
        "dayfirst": _dayfirst(args),
        "rate_unit": args.rate_unit,
        "quote_filter": _quote_filter(args),
        "cache_dir": None if args.cache is None else Path(args.cache),
    }
    if args.forwards is not None:
        kwargs["forwards"] = args.forwards
    if args.rates is not None:
        kwargs["rates"] = _parse_rates(args.rates)
    if args.spot is not None:
        kwargs["spot"] = args.spot
    return kwargs


def _cross_section(dataset: Dataset, args: argparse.Namespace):
    stamp = _parse_date(args.date, dayfirst=_dayfirst(args))
    expiry = None if args.expiry is None else _parse_date(args.expiry, dayfirst=_dayfirst(args))
    days = args.days
    target = args.target
    chosen = sum(value is not None for value in (expiry, days, target))
    if chosen != 1:
        raise ValueError("exactly one of --expiry, --days, or --target is required")
    if args.window is not None and target is None:
        raise ValueError("--window requires --target")
    window = None if args.window is None else (int(args.window[0]), int(args.window[1]))
    return dataset.cross_section(
        stamp,
        expiry=expiry,
        days=days,
        target_days=target,
        window=window,
        am_settlement=args.am_settlement,
    )


def _option_paths(args: argparse.Namespace) -> list[Path]:
    if args.data is not None and args.options:
        raise ValueError("pass either --data or --options, not both")
    if args.options:
        paths: list[Path] = []
        for raw in args.options:
            path = Path(raw)
            if path.is_dir():
                paths.extend(_option_files_in_root(path))
            else:
                paths.append(path)
        if not paths:
            raise ValueError("no option files given")
        return paths
    return _option_files_in_root(_data_root(args))


def _data_root(args: argparse.Namespace) -> Path:
    if args.data is not None:
        return Path(args.data)
    if _DEFAULT_DATA_DIR.is_dir():
        return _DEFAULT_DATA_DIR
    raise ValueError("no data directory; pass --data DIR or --options PATH")


def _option_files_in_root(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in _directory_tables(root):
        if file_kind(path) == "unknown":
            continue
        if classify_table(path) == "options":
            files.append(path)
    if not files:
        raise ValueError(f"no option tables in directory {root}")
    return files


def _directory_tables(root: Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    search = [directory]
    nested = directory / "RAW_DATA"
    if nested.is_dir():
        search.append(nested)
    files: list[Path] = []
    for folder in search:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                files.append(path)
    return files


def _print_partial_inspect(args: argparse.Namespace, option_paths: list[Path]) -> None:
    root = None if args.options else _data_root(args)
    buckets: dict[str, list[str]] = {}
    if root is not None:
        for path in _directory_tables(root):
            if file_kind(path) == "unknown":
                continue
            kind = classify_table(path)
            if kind == "unknown":
                continue
            buckets.setdefault(kind, []).append(str(path))
    payload = {
        "options": {
            "files": [str(path) for path in option_paths],
            "kind": "indexed_text",
        },
        "forwards": buckets.get("forwards", [None])[0] if root is not None else args.forwards,
        "rates": _first_rate(buckets) if root is not None else args.rates,
        "spot": bool(buckets.get("spot")) if root is not None else args.spot is not None,
        "cache_dir": str(_cache_dir(args, option_paths)),
    }
    _print_mapping(payload)


def _first_rate(buckets: dict[str, list[str]]):
    for key in ("rate_series", "zero_curve"):
        files = buckets.get(key)
        if files:
            return files[0]
    return None


def _needs_byte_index(paths: list[Path]) -> bool:
    if not paths:
        return False
    kinds = [file_kind(path) for path in paths]
    total = sum(path.stat().st_size for path in paths)
    return all(kind == "text" for kind in kinds) and total > DEFAULT_MEMORY_LIMIT


def _index_state(args: argparse.Namespace, option_paths: list[Path]) -> str:
    if not _needs_byte_index(option_paths):
        return "ok"
    cache_dir = _cache_dir(args, option_paths)
    column_map = _column_map(args, option_paths)
    dayfirst = _dayfirst(args)
    cache_names = _index_cache_names(option_paths)
    states = []
    for path in option_paths:
        if file_kind(path) != "text":
            continue
        date_column = _date_column_index(path, column_map)
        delimiter = sniff_delimiter(path)
        index_path = cache_dir / cache_names[path]
        states.append(
            _index_file_state(path, index_path, date_column, delimiter, dayfirst=dayfirst)
        )
    if not states:
        return "ok"
    if any(state == "missing" for state in states):
        return "missing"
    if any(state == "stale" for state in states):
        return "stale"
    return "ok"


def _index_file_state(
    csv_path: Path,
    index_path: Path,
    date_column: int,
    delimiter: str,
    *,
    dayfirst: bool | None,
) -> str:
    if not index_path.is_file():
        return "missing"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        size = int(payload["size"])
        mtime = float(payload["mtime"])
        stored_column = int(payload["date_column"])
        stored_delimiter = str(payload["delimiter"])
        stored_source = payload["source"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "missing"
    if not _same_resolved_path(stored_source, csv_path):
        return "stale"
    if "dayfirst" in payload and payload["dayfirst"] != dayfirst:
        return "stale"
    info = csv_path.stat()
    if (
        size == int(info.st_size)
        and math.isclose(mtime, float(info.st_mtime), rel_tol=0.0, abs_tol=_MTIME_ABS_TOL)
        and stored_column == int(date_column)
        and stored_delimiter == str(delimiter)
    ):
        return "ok"
    return "stale"


def _index_cache_names(paths: list[Path]) -> dict[Path, str]:
    """Cache filenames shared with the data layer, including path-hashed stems."""
    ordered = [Path(path) for path in paths]
    return source_mod._cache_filenames(ordered, "index.json")


def _same_resolved_path(stored, path: Path) -> bool:
    try:
        return Path(str(stored)).resolve() == Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _cache_dir(args: argparse.Namespace, option_paths: list[Path]) -> Path:
    if args.cache is not None:
        return Path(args.cache)
    return option_paths[0].resolve().parent.parent / "CACHE"


def _column_map(args: argparse.Namespace, option_paths: list[Path]) -> ColumnMap:
    overrides = _column_overrides(args.column)
    return ColumnMap.detect(peek_columns(option_paths[0]), overrides=overrides)


def _column_overrides(items: list[str]) -> dict[str, str] | None:
    if not items:
        return None
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=RAW in --column, got {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if not name or not raw:
            raise ValueError(f"expected NAME=RAW in --column, got {item!r}")
        overrides[name] = raw
    return overrides


def _date_column_index(path: Path, column_map: ColumnMap) -> int:
    columns = peek_columns(path)
    target = normalize_name(column_map.date)
    for position, name in enumerate(columns):
        if normalize_name(name) == target:
            return position
    raise ValueError(f"date column {column_map.date!r} is not in {path}")


def _quote_filter(args: argparse.Namespace) -> QuoteFilter:
    filt = QuoteFilter()
    if args.no_volume_filter:
        filt = replace(filt, min_volume=0.0)
    if args.min_volume is not None:
        filt = replace(filt, min_volume=float(args.min_volume))
    return filt


def _strike_scale(value: str) -> float | str:
    text = str(value).strip()
    if text == "auto":
        return "auto"
    scale = float(text)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"strike-scale must be auto or a positive number, got {value!r}")
    return scale


def _parse_rates(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def _dayfirst(args: argparse.Namespace) -> bool | None:
    return True if args.dayfirst else None


def _lm_range(args: argparse.Namespace) -> tuple[float, float]:
    if args.lm_range is None:
        return DEFAULT_LM_RANGE
    return float(args.lm_range[0]), float(args.lm_range[1])


def _parse_date(value, *, dayfirst: bool | None) -> pd.Timestamp:
    parsed = parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"could not parse date {value!r}")
    return pd.Timestamp(parsed).normalize()


def _parse_support_cap(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "unbounded"}:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _parse_direction(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"direction must be P,L got {text!r}")
    return float(parts[0]), float(parts[1])


def _csv_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _print_mapping(mapping: dict, prefix: str = "") -> None:
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            _print_mapping(value, name + ".")
        else:
            print(f"{name}: {_format_value(value)}")


def _format_value(value) -> str:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "None"
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        inner = ", ".join(_format_value(item) for item in value)
        return f"[{inner}]"
    return str(value)


def _configure_logging(level: str) -> bool:
    name = str(level).upper()
    numeric = getattr(logging, name, None)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")
    logging.basicConfig(level=numeric, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("crashfrontier").setLevel(numeric)
    return name == "DEBUG"


def _exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1
