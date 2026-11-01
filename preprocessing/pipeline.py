"""Unified command-line interface for SPARC extraction, mapping, and risk inference."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from inference.generic import run_generic
from preprocessing.extract import parse_gpus, run_parallel
from preprocessing.mpp import prepare_mpp
from preprocessing.splits import prepare_splits


def feature_dir(work_dir: Path, magnification: int, patch_size: int, overlap: int) -> Path:
    return Path(work_dir) / f"{magnification}x_{patch_size}px_{overlap}px_overlap" / "features_hoptimus1"


def build_map_command(embedding_dir: Path, output_dir: Path, model_path: Path,
                      gpu: int, overwrite: bool = False) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable, "-m", "preprocessing.sparc_map",
        "--embedding-dir", str(embedding_dir), "--output-dir", str(output_dir),
        "--model-path", str(model_path), "--model-type", "auto", "--output-format", "h5",
    ]
    if overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, environment


def run_map(embedding_dir: Path, output_dir: Path, model_path: Path,
            gpu: int, overwrite: bool = False) -> None:
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"SPARC-Map checkpoint does not exist: {model_path}")
    command, environment = build_map_command(embedding_dir, output_dir, model_path, gpu, overwrite)
    subprocess.run(command, env=environment, check=True)


def _mpp_backend(reader_type: str) -> str:
    # MPP probing only understands the openslide-like
    # backends (openslide / tiffslide). trident's other reader types
    # (image / cucim) keep their own MPP handling, so fall back to openslide
    # for manifest generation.
    return reader_type if reader_type in ("openslide", "tiffslide") else "openslide"


def run_extract(args: argparse.Namespace) -> Path:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.work_dir / "slides_with_mpp.csv"
    split_dir = args.work_dir / "split_lists"
    if not manifest.exists() or args.force:
        prepare_mpp(args.wsi_dir, manifest, args.default_mpp, args.force,
                    backend=_mpp_backend(args.reader_type))
    # Split manifests are cheap to regenerate and must match the current worker count.
    prepare_splits(manifest, split_dir, len(args.gpu_ids), overwrite=True)
    if args.hf_cache:
        os.environ["HF_HOME"] = str(args.hf_cache)
    return run_parallel(
        args.wsi_dir, args.work_dir, split_dir, args.gpu_ids,
        patch_size=args.patch_size, magnification=args.magnification,
        overlap=args.overlap, batch_size=args.batch_size, reader_type=args.reader_type,
    )


def add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--default-mpp", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--magnification", type=int, default=20)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reader-type", choices=["openslide", "tiffslide", "image", "cucim"], default="openslide")
    parser.add_argument("--force", action="store_true")


def add_risk_mapping(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cancer-type")
    group.add_argument("--cancer-map-csv", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    extract = subparsers.add_parser("extract", help="Extract H-Optimus-1 features from raw slides")
    add_extract_arguments(extract)

    mapping = subparsers.add_parser("map", help="Predict per-patch SPARC-Map programs")
    mapping.add_argument("--embedding-dir", type=Path, required=True)
    mapping.add_argument("--output-dir", type=Path, required=True)
    mapping.add_argument("--model-path", type=Path, required=True)
    mapping.add_argument("--gpu", type=int, default=0)
    mapping.add_argument("--force", action="store_true")

    risk = subparsers.add_parser("risk", help="Predict per-slide SPARC-Risk scores")
    risk.add_argument("--embedding-dir", type=Path, required=True)
    risk.add_argument("--gep-dir", type=Path, required=True)
    risk.add_argument("--checkpoint-dir", type=Path, required=True)
    risk.add_argument("--output-dir", type=Path, required=True)
    risk.add_argument("--device", default="cuda:0")
    risk.add_argument("--per-fold", action="store_true")
    risk.add_argument("--force", action="store_true")
    add_risk_mapping(risk)

    all_stages = subparsers.add_parser("all", help="Run all three stages")
    add_extract_arguments(all_stages)
    all_stages.add_argument("--map-model-path", type=Path, required=True)
    all_stages.add_argument("--risk-checkpoint-dir", type=Path, required=True)
    all_stages.add_argument("--per-fold", action="store_true")
    add_risk_mapping(all_stages)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(args, "gpus"):
        args.gpu_ids = parse_gpus(args.gpus)
    if args.stage == "extract":
        print(f"H-Optimus-1 features: {run_extract(args)}")
        return
    if args.stage == "map":
        run_map(args.embedding_dir, args.output_dir, args.model_path, args.gpu, args.force)
        return
    if args.stage == "risk":
        run_generic(
            args.embedding_dir, args.gep_dir, args.checkpoint_dir, args.output_dir,
            args.cancer_type, args.cancer_map_csv, args.device, args.per_fold,
            overwrite=args.force,
        )
        return

    embeddings = run_extract(args)
    gep_dir = args.work_dir / "sparc_map"
    risk_dir = args.work_dir / "risk"
    run_map(embeddings, gep_dir, args.map_model_path, args.gpu_ids[0], args.force)
    run_generic(
        embeddings, gep_dir, args.risk_checkpoint_dir, risk_dir,
        args.cancer_type, args.cancer_map_csv, f"cuda:{args.gpu_ids[0]}", args.per_fold,
        overwrite=args.force,
    )
    print(f"SPARC pipeline complete. Risk scores: {risk_dir}")


if __name__ == "__main__":
    main()
