"""Discover whole-slide images and write their microns-per-pixel metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

SLIDE_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi"}


def discover_slides(wsi_dir: Path) -> list[Path]:
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.is_dir():
        raise NotADirectoryError(f"WSI directory does not exist: {wsi_dir}")
    return sorted(
        path for path in wsi_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SLIDE_EXTENSIONS
        and not path.name.startswith("._")
    )


def read_mpp(slide_path: Path, default_mpp: float) -> tuple[float, str]:
    """Read MPP using OpenSlide, then Aperio TIFF metadata, then the default."""
    try:
        import openslide

        slide = openslide.OpenSlide(str(slide_path))
        try:
            x = float(slide.properties.get("openslide.mpp-x", 0))
            y = float(slide.properties.get("openslide.mpp-y", 0))
        finally:
            slide.close()
        if x > 0 and y > 0:
            return (x + y) / 2.0, "openslide"
    except Exception:
        pass

    try:
        import tifffile

        with tifffile.TiffFile(slide_path) as tif:
            description = getattr(tif.pages[0], "description", "") or ""
        match = re.search(r"MPP\s*=\s*([\d.]+)", description)
        if match:
            return float(match.group(1)), "tifffile"
    except Exception:
        pass
    return float(default_mpp), "default"


def prepare_mpp(
    wsi_dir: Path,
    output_csv: Path,
    default_mpp: float = 0.25,
    overwrite: bool = False,
    reader: Callable[[Path, float], tuple[float, str]] = read_mpp,
) -> Path:
    if default_mpp <= 0:
        raise ValueError("default_mpp must be positive")
    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_csv}")
    slides = discover_slides(wsi_dir)
    if not slides:
        raise FileNotFoundError(f"No supported slides found below {wsi_dir}")
    rows = []
    for path in tqdm(slides, desc="Inspecting slide MPP"):
        mpp, source = reader(path, default_mpp)
        rows.append({"wsi": str(path.resolve()), "mpp": mpp, "mpp_source": source})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--default-mpp", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = prepare_mpp(args.wsi_dir, args.output_csv, args.default_mpp, args.overwrite)
    print(f"Wrote MPP manifest: {path}")


if __name__ == "__main__":
    main()
