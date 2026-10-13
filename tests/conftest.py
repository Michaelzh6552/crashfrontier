from __future__ import annotations

from pathlib import Path

import pytest

from tests.synthetic import (
    VARIANT_NAMES,
    FrameBundle,
    scenarios,
    synthetic_quotes,
    write_optionmetrics_dir,
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return write_optionmetrics_dir(tmp_path / "standard")


@pytest.fixture(params=list(VARIANT_NAMES))
def variant_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path | FrameBundle:
    if request.param == "parquet":
        pytest.importorskip("pyarrow")
    return write_optionmetrics_dir(tmp_path / "variant", name=request.param)


@pytest.fixture
def dataset(raw_dir: Path, tmp_path: Path):
    from crashfrontier.data.dataset import Dataset

    return Dataset.from_directory(raw_dir, cache_dir=tmp_path / "cache")


@pytest.fixture
def atomic_quotes():
    nodes, weights = scenarios(0.9)["bimodal"]
    return synthetic_quotes(nodes, weights, spread=1e-6)
