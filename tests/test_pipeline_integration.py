"""Opt-in, real-GPU validation of the complete three-stage pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocessing.pipeline import main


@pytest.mark.integration
def test_real_three_stage_pipeline(scratch_dir):
    required = {
        "wsi": os.environ.get("SPARC_SMOKE_WSI_DIR"),
        "map": os.environ.get("SPARC_MAP_CHECKPOINT"),
        "risk": os.environ.get("SPARC_RISK_CHECKPOINT_DIR"),
        "cancer": os.environ.get("SPARC_SMOKE_CANCER_TYPE"),
    }
    if not all(required.values()):
        pytest.skip("Set SPARC_SMOKE_WSI_DIR, SPARC_MAP_CHECKPOINT, SPARC_RISK_CHECKPOINT_DIR, and SPARC_SMOKE_CANCER_TYPE")
    work = scratch_dir / "real-pipeline"
    main([
        "all", "--wsi-dir", required["wsi"], "--work-dir", str(work),
        "--gpus", os.environ.get("SPARC_SMOKE_GPUS", "0"),
        "--map-model-path", required["map"],
        "--risk-checkpoint-dir", required["risk"],
        "--cancer-type", required["cancer"],
    ])
    embeddings = list((work / "20x_224px_0px_overlap" / "features_hoptimus1").glob("*.h5"))
    gep = list((work / "sparc_map").glob("*.h5"))
    risks = list((work / "risk").glob("*.npz"))
    assert len(embeddings) == len(gep) == len(risks) == 1
    with h5py.File(embeddings[0]) as emb, h5py.File(gep[0]) as scores:
        assert emb["features"].shape[0] == scores["features"].shape[0]
        assert scores["features"].shape[1] == 40
    with np.load(risks[0]) as result:
        assert np.isfinite(result["risk"]).all()
