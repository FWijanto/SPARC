from pathlib import Path

import pandas as pd
import pytest

from preprocessing.mpp import prepare_mpp, read_mpp
from preprocessing.splits import prepare_splits


def test_prepare_mpp_discovers_supported_slides_and_records_source(scratch_dir):
    slides = scratch_dir / "slides"
    (slides / "nested").mkdir(parents=True)
    for relative in ("a.svs", "nested/b.TIFF"):
        (slides / relative).touch()
    (slides / "ignore.txt").touch()

    output = scratch_dir / "manifest.csv"
    prepare_mpp(slides, output, reader=lambda path, default: (0.4, "fake"))
    frame = pd.read_csv(output)
    assert [Path(value).name for value in frame.wsi] == ["a.svs", "b.TIFF"]
    assert frame.mpp.tolist() == [0.4, 0.4]
    assert frame.mpp_source.tolist() == ["fake", "fake"]
    with pytest.raises(FileExistsError):
        prepare_mpp(slides, output, reader=lambda path, default: (0.4, "fake"))


def test_prepare_splits_is_balanced_and_deterministic(scratch_dir):
    manifest = scratch_dir / "manifest.csv"
    pd.DataFrame({"wsi": [f"slide-{i}.svs" for i in range(7)], "mpp": [0.25] * 7}).to_csv(manifest, index=False)
    paths = prepare_splits(manifest, scratch_dir / "splits", 3)
    frames = [pd.read_csv(path) for path in paths]
    assert [len(frame) for frame in frames] == [3, 2, 2]
    assert pd.concat(frames).wsi.tolist() == [f"slide-{i}.svs" for i in range(7)]
    with pytest.raises(FileExistsError):
        prepare_splits(manifest, scratch_dir / "splits", 3)

    resized = prepare_splits(manifest, scratch_dir / "splits", 2, overwrite=True)
    resized_frames = [pd.read_csv(path) for path in resized]
    assert [len(frame) for frame in resized_frames] == [4, 3]
    assert pd.concat(resized_frames).wsi.tolist() == [f"slide-{i}.svs" for i in range(7)]
    assert not (scratch_dir / "splits" / "split_02.csv").exists()


def test_tifffile_mpp_fallback(scratch_dir):
    import tifffile
    import numpy as np

    slide = scratch_dir / "aperio.tiff"
    tifffile.imwrite(slide, np.zeros((8, 8, 3), dtype=np.uint8), description="Aperio Image Library|MPP = 0.5000")
    value, source = read_mpp(slide, default_mpp=0.25)
    assert value == 0.5
    assert source == "tifffile"


def test_split_schema_validation(scratch_dir):
    invalid = scratch_dir / "invalid.csv"
    pd.DataFrame({"path": ["x"]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        prepare_splits(invalid, scratch_dir / "splits", 1)
