"""NN completeness using bundled fixture checkpoints (no slugpy)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BP = ROOT / "bundled_pipeline"
FIX = ROOT / "fixtures" / "legus_hlsp" / "nn_comp_smoke"


@pytest.fixture(autouse=True)
def _path():
    sys.path.insert(0, str(BP))
    yield
    try:
        sys.path.remove(str(BP))
    except ValueError:
        pass


def test_predict_with_bundled_nn_fixture():
    from completeness_io import predict_catalog_completeness_with_nn  # noqa: PLC0415

    scaler = FIX / "ngc628-c_scaler.pkl"
    model = FIX / "ngc628-c_model.pt"
    assert scaler.is_file() and model.is_file(), "bundle fixture missing; copy from repo tests/data"

    phot = np.zeros((2, 5), dtype=float)
    filters = [
        "WFC3_UVIS_F275W",
        "WFC3_UVIS_F336W",
        "ACS_F435W",
        "ACS_F555W",
        "ACS_F814W",
    ]
    p = predict_catalog_completeness_with_nn(
        phot,
        galaxy_fullname="ngc628-c",
        nn_dir=None,
        subset_filters=filters,
        full_filter_order=filters,
        nn_scaler_path=str(scaler),
        nn_model_path=str(model),
    )
    assert p.shape == (2,)
    assert np.all(np.isfinite(p)) and np.all((p >= 0.0) & (p <= 1.0))
