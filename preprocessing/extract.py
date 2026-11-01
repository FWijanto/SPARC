"""Run the vendored Trident GrandQC/H-Optimus-1 workflow over slide splits."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Sequence

from preprocessing.slide_backend import (
    ensure_tiffslide_registered,
)

def parse_gpus(value: str | Sequence[int]) -> list[int]:
    items = value if not isinstance(value, str) else value.split(",")
    gpus = [int(item) for item in items]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative indices")
    return gpus


def run_partition(wsi_dir: Path, work_dir: Path, split_csv: Path, gpu: int, patch_size: int = 224,
                  magnification: int = 20, overlap: int = 0, batch_size: int = 64,
                  reader_type: str = "openslide") -> None:
    try:
        from trident import Processor
        from trident.patch_encoder_models.load import encoder_factory
        from trident.segmentation_models.load import segmentation_model_factory
    except ImportError as exc:
        raise ImportError(
            "Raw-slide extraction requires TRIDENT; install SPARC with "
            "`pip install -e '.[wsi]'`."
        ) from exc

    ensure_tiffslide_registered()

    processor = Processor(
        job_dir=str(work_dir), wsi_source=str(wsi_dir), custom_list_of_wsis=str(split_csv),
        skip_errors=False, reader_type=reader_type,
    )
    segmenter = segmentation_model_factory("grandqc")
    processor.run_segmentation_job(
        segmenter, seg_mag=segmenter.target_mag, holes_are_tissue=True,
        artifact_remover_model=None, batch_size=batch_size, device=f"cuda:{gpu}",
    )
    coords_dir = f"{magnification}x_{patch_size}px_{overlap}px_overlap"
    processor.run_patching_job(
        target_magnification=magnification, patch_size=patch_size, overlap=overlap,
        saveto=coords_dir, min_tissue_proportion=0.0,
    )
    encoder = encoder_factory("hoptimus1")
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir, patch_encoder=encoder, device=f"cuda:{gpu}",
        saveas="h5", batch_limit=batch_size,
    )


def run_parallel(wsi_dir: Path, work_dir: Path, split_dir: Path, gpus: Sequence[int], **kwargs) -> Path:
    gpus = parse_gpus(gpus)
    jobs = []
    for index, gpu in enumerate(gpus):
        split = Path(split_dir) / f"split_{index:02d}.csv"
        if not split.exists():
            raise FileNotFoundError(f"Missing worker manifest: {split}")
        process = mp.Process(target=run_partition, args=(wsi_dir, work_dir, split, gpu), kwargs=kwargs)
        process.start()
        jobs.append(process)
    for process in jobs:
        process.join()
    failures = [process.exitcode for process in jobs if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"Trident worker failures: {failures}")
    patch_size = kwargs.get("patch_size", 224)
    magnification = kwargs.get("magnification", 20)
    overlap = kwargs.get("overlap", 0)
    return Path(work_dir) / f"{magnification}x_{patch_size}px_{overlap}px_overlap" / "features_hoptimus1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--magnification", type=int, default=20)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reader-type", choices=["openslide", "tiffslide", "image", "cucim"], default="openslide")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.hf_cache:
        os.environ["HF_HOME"] = str(args.hf_cache)
    output = run_parallel(
        args.wsi_dir, args.work_dir, args.split_dir, parse_gpus(args.gpus),
        patch_size=args.patch_size, magnification=args.magnification,
        overlap=args.overlap, batch_size=args.batch_size, reader_type=args.reader_type,
    )
    print(f"H-Optimus-1 features: {output}")


if __name__ == "__main__":
    main()
