"""Temporary-directory fixtures for pipeline tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def scratch_dir(tmp_path):
    configured_root = os.environ.get("SPARC_TEST_OUTPUT_ROOT")
    if configured_root is None:
        yield tmp_path
        return

    root = Path(configured_root)
    suite_root = root / "sparc-pipeline-tests"
    suite_root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest-", dir=suite_root))
    yield path
    if os.environ.get("SPARC_KEEP_TEST_OUTPUT") != "1":
        shutil.rmtree(path)
