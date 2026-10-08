import h5py
import numpy as np
import pytest
import torch

from preprocessing.map_models import MeanResidSpatialTransformer, OriginalSpatialTransformer
from preprocessing.sparc_map import (
    KNNInferenceDataset, list_embedding_files, parse_ids_arg,
    save_scores_h5, slide_id_for_embedding,
)


def test_synthetic_h5_knn_and_output_contract(scratch_dir):
    embeddings = scratch_dir / "embeddings"
    embeddings.mkdir()
    source = embeddings / "slide-a.h5"
    with h5py.File(source, "w") as handle:
        handle["features"] = np.arange(24, dtype=np.float32).reshape(6, 4)
        handle["coords"] = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
    dataset = KNNInferenceDataset(source, "features", "coords", k_neighbors=3, coord_norm="median_knn")
    center, neighbors, relative, mask = dataset[0]
    assert center.shape == (4,)
    assert neighbors.shape == (3, 4)
    assert relative.shape == (3, 2)
    assert mask.all()
    assert list_embedding_files(embeddings) == [source]
    assert slide_id_for_embedding(source, embeddings) == "slide-a"
    assert parse_ids_arg("slide-a, slide-b") == {"slide-a", "slide-b"}

    destination = scratch_dir / "gep" / "slide-a.h5"
    names = [f"program-{index}" for index in range(40)]
    values = np.zeros((6, 40), dtype=np.float32)
    save_scores_h5(destination, values, "features", names)
    with h5py.File(destination) as handle:
        assert handle["features"].shape == (6, 40)
        assert len(handle["program_names"]) == 40


def test_knn_rejects_single_patch(scratch_dir):
    source = scratch_dir / "one.h5"
    with h5py.File(source, "w") as handle:
        handle["features"] = np.zeros((1, 4), dtype=np.float32)
        handle["coords"] = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2"):
        KNNInferenceDataset(source, "features", "coords")


def test_checkpoint_compatible_model_shapes():
    center = torch.randn(2, 8)
    neighbors = torch.randn(2, 3, 8)
    coords = torch.randn(2, 3, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    original = OriginalSpatialTransformer(8, 40, hidden_dim=16, num_layers=1, num_heads=4)
    mean_resid = MeanResidSpatialTransformer(8, 40, hidden_dim=16, num_layers=1, num_heads=4)
    assert original(center, neighbors, coords, mask).shape == (2, 40)
    residual, mean = mean_resid(center, neighbors, coords, mask)
    assert residual.shape == mean.shape == (2, 40)
