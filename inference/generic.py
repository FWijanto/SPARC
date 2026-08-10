"""Run SPARC-Risk on arbitrary matching H-Optimus and SPARC-Map directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from inference.core import (
    fusion_needs_gep, load_checkpoints, load_fold_models, load_slide_features,
    run_model_on_slide, save_slide_prediction,
)
from sparc.data.dataset import CANCER_TYPE_TO_IDX


def build_cancer_map(slides: Iterable[str], cancer_type: str | None = None,
                     cancer_map_csv: Path | None = None) -> dict[str, int]:
    slides = list(slides)
    if (cancer_type is None) == (cancer_map_csv is None):
        raise ValueError("Provide exactly one of --cancer-type or --cancer-map-csv")
    if cancer_type is not None:
        if cancer_type not in CANCER_TYPE_TO_IDX:
            raise ValueError(f"Unknown cancer type: {cancer_type}")
        return {slide: CANCER_TYPE_TO_IDX[cancer_type] for slide in slides}
    frame = pd.read_csv(cancer_map_csv)
    required = {"slide_id", "cancer_type"}
    if not required.issubset(frame.columns):
        raise ValueError("Cancer map must contain slide_id,cancer_type columns")
    if frame["slide_id"].duplicated().any():
        raise ValueError("Cancer map contains duplicate slide_id values")
    unknown = sorted(set(frame["cancer_type"]) - set(CANCER_TYPE_TO_IDX))
    if unknown:
        raise ValueError(f"Unknown cancer types: {unknown}")
    raw = dict(zip(frame["slide_id"].astype(str), frame["cancer_type"]))
    missing = sorted(set(slides) - set(raw))
    if missing:
        raise ValueError(f"Cancer map is missing slides: {missing[:20]}")
    return {slide: CANCER_TYPE_TO_IDX[raw[slide]] for slide in slides}


def validate_inputs(embedding_dir: Path, gep_dir: Path, needs_gep: bool = True) -> list[str]:
    embedding_dir = Path(embedding_dir)
    if not embedding_dir.is_dir():
        raise NotADirectoryError(embedding_dir)
    slides = sorted(path.stem for path in embedding_dir.glob("*.h5") if path.is_file())
    if not slides:
        raise FileNotFoundError(f"No embedding HDF5 files in {embedding_dir}")
    if needs_gep:
        missing = [slide for slide in slides if not (Path(gep_dir) / f"{slide}.h5").exists()]
        if missing:
            raise FileNotFoundError(f"Missing SPARC-Map files: {missing[:20]}")
    return slides


def run_generic(embedding_dir: Path, gep_dir: Path, checkpoint_dir: Path, output_dir: Path,
                cancer_type: str | None = None, cancer_map_csv: Path | None = None,
                device_name: str = "cuda:0", per_fold: bool = False) -> list[Path]:
    checkpoints = load_checkpoints(checkpoint_dir)
    needs_gep = fusion_needs_gep(checkpoints)
    slides = validate_inputs(embedding_dir, gep_dir, needs_gep)
    cancer_map = build_cancer_map(slides, cancer_type, cancer_map_csv)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    fold_models = load_fold_models(checkpoints, device)
    outputs = []
    try:
        for slide in slides:
            destination = Path(output_dir) / f"{slide}.npz"
            if destination.exists():
                outputs.append(destination)
                continue
            embedding, gep, coords = load_slide_features(
                Path(embedding_dir) / f"{slide}.h5",
                Path(gep_dir) / f"{slide}.h5" if needs_gep else None,
            )
            if gep is not None and gep.shape[0] != embedding.shape[0]:
                raise ValueError(f"Patch count mismatch for {slide}: {embedding.shape[0]} != {gep.shape[0]}")
            if gep is not None and gep.shape[1] != 40:
                raise ValueError(f"Expected 40 GEP features for {slide}, got {gep.shape[1]}")
            emb_t = torch.from_numpy(embedding).to(device)
            gep_t = torch.from_numpy(gep).to(device) if gep is not None else None
            coords_t = torch.from_numpy(coords).to(device)
            risk, representation = run_model_on_slide(
                fold_models, emb_t, gep_t, coords_t, cancer_map[slide], device, per_fold,
            )
            save_slide_prediction(destination, risk, representation, cancer_map[slide], per_fold)
            outputs.append(destination)
    finally:
        for _, _, handle in fold_models:
            handle.remove()
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--gep-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cancer-type")
    group.add_argument("--cancer-map-csv", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-fold", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run_generic(
        args.embedding_dir, args.gep_dir, args.checkpoint_dir, args.output_dir,
        args.cancer_type, args.cancer_map_csv, args.device, args.per_fold,
    )
    print(f"Wrote {len(outputs)} risk predictions below {args.output_dir}")


if __name__ == "__main__":
    main()
