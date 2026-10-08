from pathlib import Path

import pandas as pd
import pytest

from inference.generic import build_cancer_map, validate_inputs
from preprocessing.extract import parse_gpus
from preprocessing.pipeline import build_map_command, feature_dir
from preprocessing import pipeline


def test_gpu_and_output_path_parsing(scratch_dir):
    assert parse_gpus("0,2,3") == [0, 2, 3]
    with pytest.raises(ValueError):
        parse_gpus("0,0")
    assert feature_dir(scratch_dir, 20, 224, 0).name == "features_hoptimus1"


def test_map_command_preserves_paths_with_spaces(scratch_dir):
    command, environment = build_map_command(
        scratch_dir / "features with spaces", scratch_dir / "map output",
        scratch_dir / "model file.pth", 3, overwrite=True,
    )
    assert str(scratch_dir / "features with spaces") in command
    assert command[-1] == "--overwrite"
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"


def test_generic_cancer_mapping_and_missing_gep(scratch_dir):
    assert set(build_cancer_map(["a", "b"], cancer_type="TCGA-LUAD")) == {"a", "b"}
    mapping = scratch_dir / "mapping.csv"
    pd.DataFrame({"slide_id": ["a", "b"], "cancer_type": ["TCGA-LUAD", "TCGA-LUSC"]}).to_csv(mapping, index=False)
    assert set(build_cancer_map(["a", "b"], cancer_map_csv=mapping)) == {"a", "b"}
    with pytest.raises(ValueError, match="exactly one"):
        build_cancer_map(["a"])

    embeddings = scratch_dir / "embeddings"
    gep = scratch_dir / "gep"
    embeddings.mkdir()
    gep.mkdir()
    (embeddings / "a.h5").touch()
    with pytest.raises(FileNotFoundError, match="Missing SPARC-Map"):
        validate_inputs(embeddings, gep)


def test_all_stage_handoffs(monkeypatch, scratch_dir):
    slides = scratch_dir / "slides"
    slides.mkdir()
    embeddings = scratch_dir / "work" / "20x_224px_0px_overlap" / "features_hoptimus1"
    calls = []
    monkeypatch.setattr(pipeline, "run_extract", lambda args: embeddings)
    monkeypatch.setattr(pipeline, "run_map", lambda *args: calls.append(("map", args)))
    monkeypatch.setattr(
        pipeline, "run_generic",
        lambda *args, **kwargs: calls.append(("risk", args, kwargs)),
    )
    pipeline.main([
        "all", "--wsi-dir", str(slides), "--work-dir", str(scratch_dir / "work"),
        "--gpus", "0,1", "--map-model-path", str(scratch_dir / "map.pth"),
        "--risk-checkpoint-dir", str(scratch_dir / "risk-checkpoints"),
        "--cancer-type", "TCGA-LUAD", "--force",
    ])
    assert calls[0][0] == "map"
    assert calls[0][1][0] == embeddings
    assert calls[1][0] == "risk"
    assert calls[1][1][0] == embeddings
    assert calls[1][2]["overwrite"] is True
