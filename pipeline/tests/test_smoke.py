"""Smoke tests: package import, bundled imports, bundled CLI help."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from legus_slug_pipeline import PipelinePaths


def _bundle_root() -> Path:
    """``new_pipeline_slug_legus/`` (parent of ``tests/``)."""
    return Path(__file__).resolve().parents[1]


def test_pipeline_paths_apply_env(monkeypatch: pytest.MonkeyPatch) -> None:
    p = PipelinePaths(
        legus_cct_root="/tmp/cct",
        legus_tab_dir="/tmp/tab",
        cluster_slug_lib_dir="/tmp/cslib",
        cluster_slug_lib_name="/tmp/cslib/tang",
        output_mcmc_chains_dir="/tmp/out",
    )
    p.apply_env()
    assert os.environ["LEGUS_CCT_ROOT"] == "/tmp/cct"
    assert os.environ["LEGUS_TAB_DIR"] == "/tmp/tab"
    assert os.environ["CLUSTER_SLUG_LIB_DIR"] == "/tmp/cslib"
    assert os.environ["CLUSTER_SLUG_LIB_NAME"] == "/tmp/cslib/tang"


def test_bundled_pipeline_imports_catalog_readers() -> None:
    """Bundled modules import without slugpy (readers only)."""
    bp = str(_bundle_root() / "bundled_pipeline")
    if bp not in sys.path:
        sys.path.insert(0, bp)
    import catalog_readers  # noqa: PLC0415

    assert hasattr(catalog_readers, "reader_register")


def test_analyze_catalog_mid_mdd_help_uses_bundled_copy() -> None:
    """CLI help from the tarball copy (requires slugpy: same as production driver)."""
    pytest.importorskip("slugpy.cluster_slug")
    script = _bundle_root() / "bundled_pipeline" / "analyze_catalog_mid_mdd.py"
    r = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=str(_bundle_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "libdir" in out
    assert "--output-mcmc-chains-dir" in out
    assert "--nn-dir" in out
