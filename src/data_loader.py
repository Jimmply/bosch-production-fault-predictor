"""Chunked loading + Parquet reencoding for the 14 GB Bosch dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


@dataclass
class DataPaths:
    raw_dir: Path
    parquet_dir: Path
    train_numeric: Path
    train_date: Path
    train_categorical: Path
    test_numeric: Path
    test_date: Path


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> DataPaths:
    raw = REPO_ROOT / cfg["data"]["raw_dir"]
    parquet = REPO_ROOT / cfg["data"]["parquet_dir"]
    files = cfg["data"]["files"]
    return DataPaths(
        raw_dir=raw,
        parquet_dir=parquet,
        train_numeric=raw / files["train_numeric"],
        train_date=raw / files["train_date"],
        train_categorical=raw / files["train_categorical"],
        test_numeric=raw / files["test_numeric"],
        test_date=raw / files["test_date"],
    )


def iter_chunks(csv_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        yield chunk


def reencode_to_parquet(csv_path: Path, parquet_path: Path, chunksize: int = 100_000) -> None:
    """Stream CSV -> Parquet in O(n) using a single pyarrow ParquetWriter handle."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("stream-reencoding %s -> %s (chunksize=%d)", csv_path.name, parquet_path.name, chunksize)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for i, chunk in enumerate(iter_chunks(csv_path, chunksize)):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(chunk)
            if i % 10 == 0:
                logger.info("  chunk %d cumulative %d rows", i, total_rows)
    finally:
        if writer is not None:
            writer.close()
    logger.info("done: %d rows written to %s", total_rows, parquet_path)


def load_train_features(cfg: dict, nrows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    paths = resolve_paths(cfg)
    numeric_pq = paths.parquet_dir / "train_numeric.parquet"
    if not numeric_pq.exists():
        raise FileNotFoundError(
            f"expected {numeric_pq}; run scripts/download_data.py then reencode via reencode_to_parquet()."
        )
    df = pd.read_parquet(numeric_pq) if nrows is None else pd.read_parquet(numeric_pq).head(nrows)
    y = df[cfg["loading"]["target_col"]].astype(int)
    X = df.drop(columns=[cfg["loading"]["target_col"], cfg["loading"]["id_col"]])
    return X, y
