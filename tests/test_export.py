from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crashfrontier.export import frontier_frames, write_frontier, write_json
from crashfrontier.frontier import binding_quotes, compute_frontier, nested_frontiers
from tests.test_frontier import THRESHOLD, _quotes, _section


def test_frontier_frames_and_write_frontier(tmp_path: Path) -> None:
    quotes = _quotes()
    section = _section(quotes)
    frontier = compute_frontier(section, THRESHOLD, quote_set=quotes)
    frames = frontier_frames(frontier)
    assert set(frames) == {"summary", "vertices", "halfplanes", "quotes"}
    assert len(frames["summary"]) == 1
    assert frames["summary"]["name"].iloc[0] == "custom"
    assert len(frames["vertices"]) == len(frontier.inner) + len(frontier.outer)
    assert list(frames["vertices"].columns[:6]) == [
        "name",
        "kind",
        "order",
        "prob",
        "loss",
        "depth",
    ]
    assert "a_prob" in frames["halfplanes"].columns
    assert "quote_id" in frames["quotes"].columns
    certificate = binding_quotes(section, THRESHOLD, quote_set=quotes)
    stem = tmp_path / "front"
    written = write_frontier(frontier, stem, certificate=certificate)
    paths = {path.name for path in written}
    assert paths == {
        "front_summary.csv",
        "front_vertices.csv",
        "front_halfplanes.csv",
        "front_quotes.csv",
        "front_binding.csv",
        "front.json",
    }
    for path in written:
        assert path.is_file()
    vertices = pd.read_csv(tmp_path / "front_vertices.csv")
    assert len(vertices) == len(frontier.inner) + len(frontier.outer)
    summary = pd.read_csv(tmp_path / "front_summary.csv")
    assert len(summary) == 1
    payload = json.loads((tmp_path / "front.json").read_text(encoding="utf-8"))
    assert "custom" in payload["frontiers"]
    assert payload["certificate"] is not None
    assert payload["frontiers"]["custom"]["date"] == "2021-07-21"


def test_write_frontier_nested_and_write_json(tmp_path: Path) -> None:
    quotes = _quotes()
    fronts = nested_frontiers(_section(quotes), THRESHOLD)
    stem = tmp_path / "nested"
    written = write_frontier(fronts, stem, formats=("csv", "json"))
    names = {path.name for path in written}
    assert "nested_summary.csv" in names
    assert "nested.json" in names
    assert "nested_binding.csv" not in names
    summary = pd.read_csv(tmp_path / "nested_summary.csv")
    assert list(summary["name"]) == ["put_pair", "local_eight", "put_wing"]
    vertices = pd.read_csv(tmp_path / "nested_vertices.csv")
    expected = sum(len(item.inner) + len(item.outer) for item in fronts.values())
    assert len(vertices) == expected
    payload = json.loads((tmp_path / "nested.json").read_text(encoding="utf-8"))
    assert payload["certificate"] is None
    assert set(payload["frontiers"]) == {"put_pair", "local_eight", "put_wing"}
    extra = tmp_path / "plain.json"
    write_json({"ok": True, "date": pd.Timestamp("2017-07-05")}, extra)
    assert extra.is_file()
    assert not Path(str(extra) + ".tmp").exists()
    assert json.loads(extra.read_text(encoding="utf-8")) == {"ok": True, "date": "2017-07-05"}


def test_write_json_strict_null_for_nonfinite_and_nat(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    write_json(
        {
            "x": float("nan"),
            "y": float("inf"),
            "when": pd.NaT,
            "ok": 1.0,
        },
        path,
    )
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"x": None, "y": None, "when": None, "ok": 1.0}
