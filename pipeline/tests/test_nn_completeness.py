"""
Tests for NN completeness resolution, dense photometry features, inference,
and analyze_catalog_mid_mdd.py CLI rules.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("python3_sc"))

from completeness_io import (
    _build_dense_photo_features,
    _nn_scaler_model_paths,
    legus_nn_missing_uv_u_fills,
    predict_catalog_completeness_with_nn,
)
from sklearn.preprocessing import StandardScaler


def test_nn_scaler_model_paths_explicit():
    s, m = _nn_scaler_model_paths(
        nn_dir=None,
        galaxy_fullname="x",
        nn_scaler_path="/tmp/a.pkl",
        nn_model_path="/tmp/b.pt",
    )
    assert s == "/tmp/a.pkl" and m == "/tmp/b.pt"


def test_nn_scaler_model_paths_requires_both_explicit():
    with pytest.raises(ValueError, match="both"):
        _nn_scaler_model_paths(
            nn_dir=None,
            galaxy_fullname="",
            nn_scaler_path="/a.pkl",
            nn_model_path=None,
        )
    with pytest.raises(ValueError, match="both"):
        _nn_scaler_model_paths(
            nn_dir=None,
            galaxy_fullname="",
            nn_scaler_path=None,
            nn_model_path="/b.pt",
        )


def test_nn_scaler_model_paths_requires_nn_dir_if_no_paths():
    with pytest.raises(ValueError, match="nn_dir"):
        _nn_scaler_model_paths(
            nn_dir=None,
            galaxy_fullname="ngc628-c",
            nn_scaler_path=None,
            nn_model_path=None,
        )


def test_build_dense_photo_features_fills_missing_with_scaler_mean():
    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    full = ["A", "B", "C"]
    subset = ["A", "C"]
    phot = np.array([[10.0, 30.0], [11.0, 31.0]])
    dense = _build_dense_photo_features(phot, subset, full, scaler=scaler)
    assert dense.shape == (2, 3)
    assert np.allclose(dense[:, 0], [10.0, 11.0])
    assert np.allclose(dense[:, 2], [30.0, 31.0])
    assert np.allclose(dense[:, 1], scaler.mean_[1])


def test_build_dense_photo_features_missing_band_fills_override():
    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    full = ["WFC3_UVIS_F275W", "B", "C"]
    subset = ["B", "C"]
    phot = np.array([[10.0, 30.0], [11.0, 31.0]])
    fills = {"WFC3_UVIS_F275W": 99.5}
    dense = _build_dense_photo_features(
        phot, subset, full, scaler=scaler, missing_band_fills=fills
    )
    assert np.allclose(dense[:, 0], [99.5, 99.5])
    assert np.allclose(dense[:, 1], [10.0, 11.0])


def test_legus_nn_missing_uv_u_fills_apparent():
    filters_cat = ["WFC3_UVIS_F275W", "WFC3_UVIS_F336W", "ACS_F435W"]
    phot = np.array(
        [
            [-10.0, -9.0, -8.0],
            [-12.0, np.nan, -8.5],
        ]
    )
    detect = np.array(
        [
            [True, True, True],
            [True, False, True],
        ]
    )
    dmod = 29.0
    full_order = list(filters_cat)
    subset = ["WFC3_UVIS_F336W", "ACS_F435W"]
    fills_abs = legus_nn_missing_uv_u_fills(
        filters_cat,
        phot,
        detect,
        dmod,
        full_order,
        subset,
        use_apparent_magnitude=False,
    )
    assert fills_abs["WFC3_UVIS_F275W"] == pytest.approx(-12.0 + 0.5)

    fills_app = legus_nn_missing_uv_u_fills(
        filters_cat,
        phot,
        detect,
        dmod,
        full_order,
        subset,
        use_apparent_magnitude=True,
    )
    assert fills_app["WFC3_UVIS_F275W"] == pytest.approx(-12.0 + dmod + 0.5)


def test_predict_catalog_nn_tiny_checkpoint(tmp_path):
    import joblib
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3)).astype(np.float64)
    scaler = StandardScaler().fit(X)

    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.GELU(),
        nn.Linear(4, 1),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.05)

    class MLP(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.net = core

        def forward(self, x):
            return self.net(x)

    mlp = MLP(model)
    pt = tmp_path / "m.pt"
    torch.save(
        {
            "model_state_dict": mlp.state_dict(),
            "model_config": {"input_dim": 3, "hidden_dim": 4, "n_hidden": 1},
        },
        pt,
    )
    pk = tmp_path / "s.pkl"
    joblib.dump({"scaler_photo": scaler}, pk)

    phot = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]], dtype=float)
    filters = ["f0", "f1", "f2"]
    p = predict_catalog_completeness_with_nn(
        phot,
        galaxy_fullname="ignored",
        nn_dir=None,
        subset_filters=filters,
        full_filter_order=filters,
        nn_scaler_path=str(pk),
        nn_model_path=str(pt),
    )
    assert p.shape == (2,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_analyze_catalog_mid_mdd_cli_requires_nn_combo():
    pytest.importorskip("slugpy.cluster_slug")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(root, "bundled_pipeline", "analyze_catalog_mid_mdd.py")
    cat = os.path.join(
        root,
        "tests",
        "data",
        "legus_hlsp",
        "hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab",
    )
    base = [
        sys.executable,
        script,
        "/tmp/cluster_slug_lib_placeholder",
        "m.pdf",
        "a.pdf",
        "v.pdf",
        cat,
    ]

    r = subprocess.run(base, cwd=root, capture_output=True, text=True)
    assert r.returncode == 2
    assert "Provide --nn-comp-dir" in (r.stderr + r.stdout)

    r2 = subprocess.run(
        base + ["--nn-scaler", "only.pkl"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 2
    assert "together" in (r2.stderr + r2.stdout).lower()


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "nn_dir", "best_model_phot_model0.pt")
        )
    ),
    reason="repo nn_dir checkpoint not present",
)
def test_predict_with_repo_nn_dir_glob():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nn_dir = os.path.join(root, "nn_dir")
    scaler_guess = os.path.join(nn_dir, "scaler_phot_model0.pkl")
    if not os.path.isfile(scaler_guess):
        scaler_guess = os.path.join(nn_dir, "best_model_phot_model0.pkl")
    if not os.path.isfile(scaler_guess):
        pytest.skip("no scaler pkl in nn_dir")

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
        nn_dir=nn_dir,
        subset_filters=filters,
        full_filter_order=filters,
    )
    assert p.shape == (2,)
    assert np.all(np.isfinite(p)) and np.all((p >= 0.0) & (p <= 1.0))


def test_clean_legus_requires_nn_source():
    pytest.importorskip("slugpy")
    from clean_legus import clean_legus

    with pytest.raises(ValueError, match="nn_dir or both"):
        clean_legus([], verbose=False, nn_dir=None, nn_scaler_path=None, nn_model_path=None)

    with pytest.raises(ValueError, match="together"):
        clean_legus(
            [],
            verbose=False,
            nn_dir=None,
            nn_scaler_path="a.pkl",
            nn_model_path=None,
        )
