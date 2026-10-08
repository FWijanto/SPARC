#!/usr/bin/env python3
"""Compute per-patch GEP scores from SPARC-Map KNN transformer checkpoints.

Supports both SPARC-Map checkpoint families:
  - meanresid: dual-head model where final prediction = mean_head + resid_head
  - original: single-head model with direct predictions
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocessing.map_models import (
    MeanResidSpatialTransformer, OriginalSpatialTransformer, build_knn_graph,
    denormalize_predictions_array,
)


PROGRAMS = [
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_COAGULATION",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_HYPOXIA",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION",
    "REACTOME_COLLAGEN_FORMATION",
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
    "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
    "REACTOME_MISMATCH_REPAIR",
    "REACTOME_NEUTROPHIL_DEGRANULATION",
    "REACTOME_TCR_SIGNALING",
    "REACTOME_TOLL_LIKE_RECEPTOR_CASCADES",
    "ANTIGEN_PRESENTATION_THOMPSON_2020_APM8",
    "B_CELL_CORE_BUDCZIES_2021",
    "CIN70_CHROMOSOMAL_INSTABILITY",
    "TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018",
    "TLS_CABRITA_9",
    "T_CELL_INFLAMED_GEP_18_AYERS_2017",
    "GOLDRATH_NAIVE_VS_MEMORY_CD8_TCELL_DN",
    "GSE13306_TREG_VS_TCONV_UP",
    "GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP",
    "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_DN",
    "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_UP",
    "GSE9650_EFFECTOR_VS_EXHAUSTED_CD8_TCELL_UP",
    "GSE9946_IMMATURE_VS_MATURE_STIMULATORY_DC_DN",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "REACTOME_YAP1_AND_WWTR1_TAZ_STIMULATED_GENE_EXPRESSION",
    "REVSC",
]


DEFAULT_CHECKPOINTS = {
    "meanresid": Path("checkpoints/meanresid_ddp/best_model.pth"),
    "original": Path("checkpoints/original_ddp/best_model.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-slide SPARC-Map GEP scores.")
    parser.add_argument("--embedding-dir", type=Path, required=True, help="Directory with embedding .h5 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store GEP score outputs.")
    parser.add_argument(
        "--model-type",
        choices=["auto", "meanresid", "original"],
        default="auto",
        help="Checkpoint architecture. 'auto' infers from checkpoint weights.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Torch checkpoint. Defaults to the bundled checkpoint for --model-type when possible.",
    )
    parser.add_argument("--embed-key", default="features", help="Embedding dataset name inside input H5 files.")
    parser.add_argument("--coord-key", default="coords", help="Coordinate dataset name inside input H5 files.")
    parser.add_argument("--output-key", default="features", help="Dataset name for saved HDF5 GEP scores.")
    parser.add_argument("--k-neighbors", type=int, default=None, help="Override KNN neighbors from checkpoint args.")
    parser.add_argument(
        "--coord-norm",
        choices=["auto", "median_knn", "none"],
        default="auto",
        help="Relative-coordinate normalization. 'auto' uses checkpoint args or architecture defaults.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    parser.add_argument(
        "--output-format",
        choices=["h5", "csv", "both"],
        default="h5",
        help="Output format.",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=None,
        help="Directory containing patches.h5 files for loading barcodes.",
    )
    parser.add_argument(
        "--csv-filename",
        default="knn_transformer_predictions.csv",
        help="Filename for per-slide CSV output.",
    )
    parser.add_argument("--ids", default="", help="Optional comma-separated slide IDs to process.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for .h5 files. By default, direct *.h5 and */embeddings.h5 are used.",
    )
    parser.add_argument(
        "--no-denorm",
        action="store_true",
        help="Skip de-normalization; save predictions in the normalized (model output) space.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def checkpoint_arg(ckpt_args: Any, key: str, default: Any = None) -> Any:
    if hasattr(ckpt_args, key):
        return getattr(ckpt_args, key)
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(key, default)
    return default


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    if args.model_type == "auto":
        raise ValueError("--model-path is required when --model-type auto is used.")
    return Path(__file__).resolve().parent / DEFAULT_CHECKPOINTS[args.model_type]


def infer_model_type(checkpoint: Dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    state_dict = checkpoint.get("model_state_dict", {})
    keys = set(state_dict.keys())
    if any(k.startswith("mean_head.") or k.startswith("resid_head.") for k in keys):
        return "meanresid"
    if any(k.startswith("head.") for k in keys):
        return "original"
    raise ValueError("Could not infer model type from checkpoint. Pass --model-type explicitly.")


def list_embedding_files(embedding_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        files = sorted(p for p in embedding_dir.rglob("*.h5") if p.is_file())
    else:
        direct_files = [p for p in embedding_dir.glob("*.h5") if p.is_file()]
        nested_embeddings = [p for p in embedding_dir.glob("*/embeddings.h5") if p.is_file()]
        files = sorted(set(direct_files + nested_embeddings))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {embedding_dir}")
    return files


def parse_ids_arg(ids: str) -> Optional[set[str]]:
    ids = ids.strip()
    if not ids:
        return None
    return {item.strip().strip("'\"") for item in ids.split(",") if item.strip().strip("'\"")}


def slide_id_for_embedding(h5_path: Path, embedding_dir: Path) -> str:
    if h5_path.name == "embeddings.h5" and h5_path.parent != embedding_dir:
        return h5_path.parent.name
    return h5_path.stem


class KNNInferenceDataset(Dataset):
    """Loads one embedding H5 and returns all center patches with k-NN context."""

    def __init__(
        self,
        h5_path: Path,
        embed_key: str,
        coord_key: str,
        k_neighbors: int = 16,
        coord_norm: str = "none",
    ):
        self.h5_path = h5_path
        self.k_neighbors = k_neighbors
        self.coord_norm = coord_norm

        with h5py.File(h5_path, "r") as f:
            if embed_key in f:
                self.embeddings = torch.from_numpy(f[embed_key][:]).float()
            elif "embeddings" in f:
                self.embeddings = torch.from_numpy(f["embeddings"][:]).float()
            elif "features" in f:
                self.embeddings = torch.from_numpy(f["features"][:]).float()
            else:
                raise KeyError(f"No embedding key '{embed_key}', 'embeddings', or 'features' in {h5_path}. Keys: {list(f.keys())}")

            if coord_key in f:
                self.coords = torch.from_numpy(f[coord_key][:]).float()
            elif "coords" in f:
                self.coords = torch.from_numpy(f["coords"][:]).float()
            elif "coordinates" in f:
                self.coords = torch.from_numpy(f["coordinates"][:]).float()
            else:
                raise KeyError(f"No coordinate key '{coord_key}', 'coords', or 'coordinates' in {h5_path}. Keys: {list(f.keys())}")

        self.num_patches = self.embeddings.shape[0]
        if self.coords.shape[0] != self.num_patches:
            raise ValueError(f"Mismatched shapes: embeddings={self.num_patches}, coords={self.coords.shape[0]}")
        if self.coords.shape[1] != 2:
            raise ValueError(f"Coordinates must have shape [N, 2], got {tuple(self.coords.shape)}")
        if self.num_patches < 2:
            raise ValueError(f"{h5_path} has {self.num_patches} patch; at least 2 are required for k-NN inference")

        self.actual_k = min(self.k_neighbors, self.num_patches - 1)
        if self.actual_k < self.k_neighbors:
            print(f"Warning: {h5_path.name} has only {self.num_patches} patches, using k={self.actual_k}.")
        self.knn_indices = build_knn_graph(self.coords, self.actual_k, exclude_self=True)

        self.coord_scale = 1.0
        if self.coord_norm == "median_knn":
            try:
                rel = self.coords[self.knn_indices] - self.coords.unsqueeze(1)
                dists = torch.norm(rel, dim=-1).reshape(-1)
                dists = dists[dists > 0]
                if dists.numel() > 0:
                    self.coord_scale = float(torch.median(dists).item())
            except Exception:
                self.coord_scale = 1.0
        if self.coord_scale <= 0:
            self.coord_scale = 1.0

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int):
        center_emb = self.embeddings[idx]
        center_coord = self.coords[idx]
        neighbor_idx = self.knn_indices[idx]
        neighbor_embs = self.embeddings[neighbor_idx]
        rel_coords = self.coords[neighbor_idx] - center_coord.unsqueeze(0)
        if self.coord_norm == "median_knn":
            rel_coords = rel_coords / max(self.coord_scale, 1e-6)
        valid_mask = torch.ones(self.actual_k, dtype=torch.bool)
        return center_emb, neighbor_embs, rel_coords, valid_mask


def collate_fn_inference(batch):
    center_embs, neighbor_embs, rel_coords, valid_masks = zip(*batch)
    max_k = max(n.shape[0] for n in neighbor_embs)
    padded_neighbors = []
    padded_coords = []
    attention_masks = []

    for n_emb, r_coord, valid_mask in zip(neighbor_embs, rel_coords, valid_masks):
        k = n_emb.shape[0]
        if k < max_k:
            n_emb = torch.cat([n_emb, torch.zeros(max_k - k, n_emb.shape[1], dtype=n_emb.dtype)], dim=0)
            r_coord = torch.cat([r_coord, torch.zeros(max_k - k, 2, dtype=r_coord.dtype)], dim=0)
            valid_mask = torch.cat([valid_mask, torch.zeros(max_k - k, dtype=torch.bool)], dim=0)
        padded_neighbors.append(n_emb)
        padded_coords.append(r_coord)
        attention_masks.append(valid_mask)

    return (
        torch.stack(center_embs),
        torch.stack(padded_neighbors),
        torch.stack(padded_coords),
        torch.stack(attention_masks),
    )


def load_barcodes(h5_path: Path, patches_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        for key in ("barcode", "barcodes"):
            if key in f:
                return np.array([b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b) for b in f[key][:]], dtype=str)

    if patches_dir is None:
        return None

    candidates = [
        patches_dir / h5_path.stem / "patches.h5",
        patches_dir / h5_path.parent.name / "patches.h5",
    ]
    for patches_file in candidates:
        if not patches_file.exists():
            continue
        try:
            with h5py.File(patches_file, "r") as f:
                for key in ("barcode", "barcodes"):
                    if key in f:
                        return np.array([b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b) for b in f[key][:]], dtype=str)
        except Exception as exc:
            print(f"Warning: Could not load barcodes from {patches_file}: {exc}")
    return None


def save_scores_h5(out_path: Path, scores: np.ndarray, output_key: str, program_names: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        ds = f.create_dataset(output_key, data=scores, compression="gzip")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        ds.attrs["program_order"] = np.array(program_names, dtype=string_dtype)
        f.create_dataset("program_names", data=np.array(program_names, dtype=string_dtype))


def save_scores_csv(out_path: Path, scores: np.ndarray, program_names: List[str], barcodes: Optional[np.ndarray] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if barcodes is None or len(barcodes) != scores.shape[0]:
        if barcodes is not None:
            print(f"Warning: {len(barcodes)} barcodes for {scores.shape[0]} scores. Using patch indices.")
        barcodes = np.array([f"patch_{i}" for i in range(scores.shape[0])], dtype=str)
    pd.DataFrame(scores, index=barcodes, columns=program_names).to_csv(out_path, index=True, index_label="barcode")
    print(f"Saved CSV predictions to {out_path} (shape: {scores.shape})")


def get_file_info(h5_path: Path, embed_key: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"file_size_mb": h5_path.stat().st_size / (1024 * 1024)}
    with h5py.File(h5_path, "r") as f:
        for key in (embed_key, "embeddings", "features"):
            if key in f:
                info["num_patches"] = f[key].shape[0]
                info["embedding_dim"] = f[key].shape[1] if len(f[key].shape) > 1 else None
                break
    return info


def build_model(checkpoint: Dict[str, Any], model_type: str, device: torch.device) -> torch.nn.Module:
    ckpt_args = checkpoint.get("args", {})
    program_names = checkpoint.get("program_names", PROGRAMS)
    kwargs = {
        "input_dim": checkpoint_arg(ckpt_args, "input_dim", 1536),
        "output_dim": checkpoint_arg(ckpt_args, "output_dim", len(program_names)),
        "hidden_dim": checkpoint_arg(ckpt_args, "hidden_dim", 256),
        "num_layers": checkpoint_arg(ckpt_args, "num_layers", 3),
        "num_heads": checkpoint_arg(ckpt_args, "num_heads", 8),
        "dropout": checkpoint_arg(ckpt_args, "dropout", 0.1),
    }
    if model_type == "meanresid":
        kwargs["num_organs"] = checkpoint_arg(ckpt_args, "num_organs", 0)
        kwargs["num_techs"] = checkpoint_arg(ckpt_args, "num_techs", 0)
        model = MeanResidSpatialTransformer(**kwargs)
    else:
        model = OriginalSpatialTransformer(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_slide(
    model: torch.nn.Module,
    model_type: str,
    dataloader: DataLoader,
    device: torch.device,
    slide_id: str,
    checkpoint: Dict[str, Any],
    program_names: List[str],
    no_denorm: bool = False,
) -> np.ndarray:
    """Run inference for one slide.

    Mean-residual checkpoints are reconstructed with the same whole-slide
    logic used by knn_transformer_meanresid.py: average the mean-head output
    across all patches, add that slide mean to every residual prediction, then
    de-normalize back to program-score space.
    """
    all_preds = []
    all_resid = []
    all_mean_patches = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inferencing {slide_id}"):
            center_emb, neighbor_embs, rel_coords, attn_mask = batch
            center_emb = center_emb.to(device)
            neighbor_embs = neighbor_embs.to(device)
            rel_coords = rel_coords.to(device)
            attn_mask = attn_mask.to(device)

            if model_type == "meanresid":
                resid_pred, mean_pred_patch = model(
                    center_emb, neighbor_embs, rel_coords, attn_mask=attn_mask
                )
                all_resid.append(resid_pred.cpu())
                all_mean_patches.append(mean_pred_patch.cpu())
            else:
                preds = model(center_emb, neighbor_embs, rel_coords, attn_mask=attn_mask)
                all_preds.append(preds.cpu().numpy())

    norm_stats = checkpoint.get("norm_stats")
    label_norm = checkpoint.get("label_norm", "none")

    if model_type != "meanresid":
        if not all_preds:
            return np.empty((0, len(program_names)), dtype=np.float32)
        raw = np.concatenate(all_preds, axis=0)
        return raw if no_denorm else denormalize_predictions_array(raw, norm_stats, label_norm, program_names)

    if not all_resid:
        return np.empty((0, len(program_names)), dtype=np.float32)

    resid = torch.cat(all_resid, dim=0).numpy()
    mean_patches = torch.cat(all_mean_patches, dim=0).numpy()
    slide_mean_pred = np.nanmean(mean_patches, axis=0, keepdims=True)
    predictions_norm = resid + slide_mean_pred

    return predictions_norm if no_denorm else denormalize_predictions_array(
        predictions_norm, norm_stats, label_norm, program_names
    )


def output_paths(h5_path: Path, embedding_dir: Path, output_dir: Path, csv_filename: str) -> tuple[str, Path, Path]:
    if h5_path.name == "embeddings.h5" and h5_path.parent != embedding_dir:
        slide_id = h5_path.parent.name
        rel_parent = h5_path.parent.relative_to(embedding_dir)
        out_h5_path = output_dir / rel_parent / "gep_scores.h5"
        out_csv_path = output_dir / rel_parent / csv_filename
    else:
        slide_id = h5_path.stem
        out_h5_path = output_dir / f"{slide_id}.h5"
        out_csv_path = output_dir / f"{slide_id}_{csv_filename}"
    return slide_id, out_h5_path, out_csv_path


def write_metadata_yaml(
    output_dir: Path,
    model_path: Path,
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    num_files: int,
    model_type: str,
    program_names: List[str],
    coord_norm: str,
    k_neighbors: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_args_raw = checkpoint.get("args", {})
    model_args = vars(model_args_raw) if hasattr(model_args_raw, "__dict__") else model_args_raw if isinstance(model_args_raw, dict) else {}
    metadata = {
        "prediction_info": {
            "timestamp": datetime.now().isoformat(),
            "num_input_files": num_files,
            "num_programs": len(program_names),
        },
        "model": {
            "checkpoint_path": str(model_path.resolve()),
            "model_type": model_type,
            "args": model_args,
            "program_order": program_names,
        },
        "input": {
            "embedding_dir": str(args.embedding_dir.resolve()),
            "embed_key": args.embed_key,
            "coord_key": args.coord_key,
            "patches_dir": str(args.patches_dir.resolve()) if args.patches_dir else None,
        },
        "output": {
            "output_dir": str(args.output_dir.resolve()),
            "output_format": args.output_format,
            "output_key": args.output_key if args.output_format in ("h5", "both") else None,
            "csv_filename": args.csv_filename if args.output_format in ("csv", "both") else None,
        },
        "inference": {
            "batch_size": args.batch_size,
            "k_neighbors": k_neighbors,
            "coord_norm": coord_norm,
            "overwrite": args.overwrite,
        },
        "programs": program_names,
    }
    metadata_path = output_dir / "prediction_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Saved prediction metadata to {metadata_path}")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_distributed = dist.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    try:
        model_path = resolve_model_path(args)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_type = infer_model_type(checkpoint, args.model_type)
        ckpt_args = checkpoint.get("args", {})
        program_names = list(checkpoint.get("program_names", PROGRAMS))
        if len(program_names) != 40:
            raise ValueError(
                f"SPARC-Risk requires exactly 40 ordered programs; checkpoint provides {len(program_names)}"
            )

        k_neighbors = args.k_neighbors
        if k_neighbors is None:
            k_neighbors = int(checkpoint_arg(ckpt_args, "k_neighbors", 16))

        if args.coord_norm == "auto":
            coord_norm = checkpoint_arg(ckpt_args, "coord_norm", "median_knn" if model_type == "meanresid" else "none")
        else:
            coord_norm = args.coord_norm

        if is_main_process():
            print(f"Model type: {model_type}")
            print(f"Model checkpoint args: {ckpt_args}")
            print(f"Using k_neighbors={k_neighbors}, coord_norm={coord_norm}")

        model = build_model(checkpoint, model_type, device)
        embed_files = list_embedding_files(args.embedding_dir, recursive=args.recursive)
        requested_ids = parse_ids_arg(args.ids)
        if requested_ids is not None:
            by_slide = {slide_id_for_embedding(path, args.embedding_dir): path for path in embed_files}
            missing = sorted(requested_ids - set(by_slide))
            if missing:
                raise FileNotFoundError(f"Requested slide IDs missing embeddings: {missing[:20]}")
            embed_files = [by_slide[slide_id] for slide_id in sorted(requested_ids)]
        total_files = len(embed_files)

        if is_distributed:
            embed_files = embed_files[rank::world_size]
            if is_main_process():
                print(f"Found {total_files} embedding files total across {world_size} ranks.")
        else:
            print(f"Found {total_files} embedding files to process")

        if torch.cuda.is_available() and is_main_process():
            print(f"CUDA device: {torch.cuda.get_device_name(local_rank)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(local_rank).total_memory / (1024**3):.2f} GB total")

        if is_main_process():
            write_metadata_yaml(args.output_dir, model_path, checkpoint, args, total_files, model_type, program_names, coord_norm, k_neighbors)

        failed_slides: List[str] = []
        for h5_path in embed_files:
            slide_id, out_h5_path, out_csv_path = output_paths(h5_path, args.embedding_dir, args.output_dir, args.csv_filename)
            need_h5 = args.output_format in ("h5", "both")
            need_csv = args.output_format in ("csv", "both")
            h5_exists = out_h5_path.exists() if need_h5 else False
            csv_exists = out_csv_path.exists() if need_csv else False

            if not args.overwrite and ((need_h5 and h5_exists) or (need_csv and csv_exists)):
                if (not need_h5 or h5_exists) and (not need_csv or csv_exists):
                    print(f"[skip] Outputs exist for {slide_id}.")
                    continue

            print(f"Processing {slide_id} from {h5_path}...")
            try:
                file_info = get_file_info(h5_path, args.embed_key)
                print(f"  File size: {file_info.get('file_size_mb', 0):.2f} MB")
                if file_info.get("num_patches"):
                    print(f"  Estimated patches: {file_info['num_patches']}")

                barcodes = load_barcodes(h5_path, args.patches_dir) if need_csv else None
                dataset = KNNInferenceDataset(
                    h5_path=h5_path,
                    embed_key=args.embed_key,
                    coord_key=args.coord_key,
                    k_neighbors=k_neighbors,
                    coord_norm=coord_norm,
                )
                print(f"  Dataset loaded: {dataset.num_patches} patches")

                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_inference,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )

                gep_scores = predict_slide(
                    model=model,
                    model_type=model_type,
                    dataloader=dataloader,
                    device=device,
                    slide_id=slide_id,
                    checkpoint=checkpoint,
                    program_names=program_names,
                    no_denorm=args.no_denorm,
                )

                if gep_scores.shape[0] == 0:
                    print(f"ERROR: No predictions generated for {slide_id}, skipping.")
                    failed_slides.append(slide_id)
                    continue

                if gep_scores.shape[0] != dataset.num_patches:
                    raise AssertionError(
                        f"Prediction count {gep_scores.shape[0]} does not match input patch count {dataset.num_patches} for {slide_id}"
                    )

                print(f"  Generated predictions shape: {gep_scores.shape}")

                if need_h5 and (args.overwrite or not h5_exists):
                    print(f"  Saving HDF5 predictions to {out_h5_path}...")
                    save_scores_h5(out_h5_path, gep_scores, args.output_key, program_names)
                if need_csv and (args.overwrite or not csv_exists):
                    print(f"  Saving CSV predictions to {out_csv_path}...")
                    save_scores_csv(out_csv_path, gep_scores, program_names, barcodes)

                del dataset, dataloader, gep_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Completed processing {slide_id}\n")

            except torch.cuda.OutOfMemoryError as exc:
                print(f"ERROR: CUDA out of memory while processing {slide_id}: {exc}")
                failed_slides.append(slide_id)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as exc:
                print(f"ERROR: Fatal error processing {slide_id}: {exc}")
                failed_slides.append(slide_id)
                import traceback

                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("  Continuing with next file...\n")
                continue
        if failed_slides:
            raise SystemExit(f"Failed to process {len(failed_slides)} slide(s): {failed_slides[:20]}")
    finally:
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
