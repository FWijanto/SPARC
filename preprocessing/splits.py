"""Split an MPP manifest into deterministic, balanced worker manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def prepare_splits(input_csv: Path, output_dir: Path, num_splits: int, overwrite: bool = False) -> list[Path]:
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")
    frame = pd.read_csv(input_csv)
    missing = {"wsi", "mpp"} - set(frame.columns)
    if missing:
        raise ValueError(f"MPP manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("MPP manifest contains no slides")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"split_{index:02d}.csv" for index in range(num_splits)]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Split outputs exist; pass --overwrite: {existing[0]}")
    if overwrite:
        for stale_path in output_dir.glob("split_*.csv"):
            if stale_path not in paths:
                stale_path.unlink()
    index_chunks = np.array_split(np.arange(len(frame)), num_splits)
    for path, indices in zip(paths, index_chunks):
        frame.iloc[indices].to_csv(path, index=False)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-splits", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = prepare_splits(args.input_csv, args.output_dir, args.num_splits, args.overwrite)
    print(f"Wrote {len(paths)} split manifests below {args.output_dir}")


if __name__ == "__main__":
    main()
