from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numexpr as ne
import numpy as np
import pytest


def _load_libwgts_class():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "bundled_pipeline"
        / "analyze_catalog_mid_mdd_test_pobs.py"
    )
    tree = ast.parse(source_path.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "libwgts"
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "args": SimpleNamespace(mdd=False),
        "ne": ne,
        "np": np,
    }
    exec(compile(module, str(source_path), "exec"), ns)
    return ns["libwgts"], ns


@pytest.mark.parametrize("mid", [True, False])
def test_cached_library_prior_weights_match_uncached_path(mid):
    libwgts, ns = _load_libwgts_class()
    rng = np.random.default_rng(123)
    n = 4096
    actual_mass = 10.0 ** rng.uniform(2.0, 8.0, n)
    eval_time = 10.0 ** rng.uniform(4.0, 10.0, n)
    a_v = rng.uniform(0.0, 3.0, n)
    lib_physprop = np.column_stack((np.log10(actual_mass), np.log10(eval_time), a_v))

    params = np.array(
        [-2.1, 6.2, -1.1 if mid else 0.55, 7.1 if mid else 5.4, -0.45, -0.48, -0.50, -0.47, -0.49, -0.46]
    )

    ns.update(
        {
            "lib_physprop": lib_physprop,
            "lib_prior_av": lib_physprop[:, 2],
            "lib_prior_mass": actual_mass,
            "lib_prior_time": eval_time,
            "lib_prior_delta_av": 3.0 / 6,
        }
    )
    ns["lib_prior_av_valid"] = (ns["lib_prior_av"] >= 0.0) & (ns["lib_prior_av"] < 3.0)
    ns["lib_prior_av_bin"] = np.floor(ns["lib_prior_av"] / ns["lib_prior_delta_av"]).astype(int)
    ns["lib_prior_av_bin"] = np.clip(ns["lib_prior_av_bin"], 0, 5)
    ns["lib_prior_av_frac"] = (
        ns["lib_prior_av"] - ns["lib_prior_av_bin"] * ns["lib_prior_delta_av"]
    ) / ns["lib_prior_delta_av"]

    weights = libwgts(params, mid=mid)
    cached = weights.wgts(lib_physprop)
    uncached = weights.wgts(np.array(lib_physprop, copy=True))

    np.testing.assert_allclose(cached, uncached, rtol=1e-12, atol=1e-30)
