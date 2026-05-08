"""Unit tests for bundled_pipeline.hybrid_libcomp."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    """Return bin centers and mean y per bin (ignore empty bins)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    digit = np.digitize(x, bins) - 1
    digit = np.clip(digit, 0, len(bins) - 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(len(centers), np.nan, dtype=float)
    for k in range(len(centers)):
        sel = digit == k
        if np.any(sel):
            means[k] = float(np.mean(y[sel]))
    ok = np.isfinite(means)
    return centers[ok], means[ok]

ROOT = Path(__file__).resolve().parents[1]
BP = ROOT / "bundled_pipeline"


@pytest.fixture(autouse=True)
def _path():
    sys.path.insert(0, str(BP))
    yield
    try:
        sys.path.remove(str(BP))
    except ValueError:
        pass


def test_hybrid_rule_ok_b_or_i_not_and():
    from hybrid_libcomp import hybrid_rule_ok_mask

    # 5 bands: indices V=3, B=2, I=4 (like 275,336,435,555,814)
    n, nf = 6, 5
    v_cat, b_cat, i_cat = 3, 2, 4
    p = np.zeros((n, nf), dtype=np.float32)
    # Row0: V=1, B=0, I=1, four other bands 1 -> should PASS (B or I)
    p[0, :] = 1.0
    p[0, b_cat] = 0.0
    p[0, i_cat] = 1.0
    # Row1: V=1, B=0, I=0, rest 1 -> should FAIL (both B and I zero)
    p[1, :] = 1.0
    p[1, b_cat] = 0.0
    p[1, i_cat] = 0.0
    # Row2: V=1, B=1, I=0 -> PASS
    p[2, :] = 1.0
    p[2, i_cat] = 0.0
    # Row3: V=0 -> FAIL
    p[3, :] = 1.0
    p[3, v_cat] = 0.0
    # Row4: only 3 bands on -> FAIL min 4
    p[4, :] = 0.0
    p[4, v_cat] = 1.0
    p[4, b_cat] = 1.0
    p[4, i_cat] = 1.0
    # Row5: all five 1 -> PASS
    p[5, :] = 1.0

    ok = hybrid_rule_ok_mask(p, v_cat, b_cat, i_cat, min_bands_nonzero=4)
    assert ok[0]
    assert not ok[1]
    assert ok[2]
    assert not ok[3]
    assert not ok[4]
    assert ok[5]


def test_hybrid_rule_ok_allows_missing_b_or_i_column():
    from hybrid_libcomp import hybrid_rule_ok_mask

    # 4-band filterset with V and I but no B should still pass B-or-I.
    p_i_only = np.ones((2, 4), dtype=np.float32)
    p_i_only[1, 3] = 0.0
    ok_i_only = hybrid_rule_ok_mask(
        p_i_only, v_cat_idx=2, b_cat_idx=None, i_cat_idx=3, min_bands_nonzero=4
    )
    assert ok_i_only[0]
    assert not ok_i_only[1]

    # 4-band filterset with V and B but no I should also pass.
    p_b_only = np.ones((2, 4), dtype=np.float32)
    p_b_only[1, 1] = 0.0
    ok_b_only = hybrid_rule_ok_mask(
        p_b_only, v_cat_idx=2, b_cat_idx=1, i_cat_idx=None, min_bands_nonzero=4
    )
    assert ok_b_only[0]
    assert not ok_b_only[1]


def test_hybrid_per_band_v_column_is_catalog_index():
    from hybrid_libcomp import hybrid_per_band_hard_flags

    # 2 rows, 3 "catalog" bands; lib has 3 columns same order
    lib_cols = np.array([0, 1, 2], dtype=int)
    bounds_hi = [-5.0, -5.0, -6.0]  # V last
    phot = np.array(
        [
            [-6.5, -6.5, -5.5],  # V=-5.5 is fainter than M_V=-6 (algebraically larger)
            [-8.0, -8.0, -7.0],
        ],
        dtype=float,
    )
    v_lib_idx = 2
    v_cat_idx = 2
    flags = hybrid_per_band_hard_flags(
        phot, lib_cols, bounds_hi, v_lib_idx, v_cat_idx, lib_vmag_max=-6.0
    )
    assert flags.shape == (2, 3)
    # Row0: V=-5.5 > -6 → faint V → V catalog column forced 0; other bands still 1 if <= hi
    assert flags[0, 2] == 0.0
    assert flags[0, 0] == 1.0 and flags[0, 1] == 1.0
    # Row1: V=-7, not faint; all <= hi
    assert np.all(flags[1] == 1.0)


def test_inside_5d_abs_box():
    from hybrid_libcomp import inside_5d_abs_box

    phot = np.array(
        [
            [-10.0, -10.0, -7.0],
            [-10.0, -10.0, -5.0],
        ]
    )
    lib_cols = np.array([0, 1, 2])
    lo = [-12.0, -12.0, -8.0]
    hi = [-8.0, -8.0, -6.0]
    m = inside_5d_abs_box(phot, lib_cols, lo, hi)
    assert m[0] and not m[1]


def test_legus_catalog_abs_bounds():
    from hybrid_libcomp import legus_catalog_abs_bounds

    phot = np.array([[-9.0, -8.0], [-7.0, -6.0]], dtype=float)
    det = np.ones((2, 2), dtype=bool)
    lo, hi = legus_catalog_abs_bounds(phot, det, ["A", "B"], range_margin=0.1)
    assert lo[0] == pytest.approx(-9.1)
    assert hi[1] == pytest.approx(-5.9)


def test_hybrid_libcomp_summary_breakdown():
    from hybrid_libcomp import format_hybrid_libcomp_summary, hybrid_libcomp_summary

    n = 5
    inside = np.array([0, 1, 1, 1, 1], dtype=bool)
    rule = np.array([0, 0, 1, 1, 1], dtype=bool)
    cnn = np.array([0.0, 0.0, 0.0, 0.5, 0.9], dtype=float)
    ch = cnn * rule.astype(float)
    s = hybrid_libcomp_summary(n, inside, rule, cnn, ch)
    assert s["n_total"] == 5
    assert s["n_kept"] == 2
    assert s["n_dropped"] == 3
    assert s["dropped_outside_5d"] == 1
    assert s["dropped_inside_rule_fail"] == 1
    assert s["dropped_inside_nn_zero"] == 1
    assert s["kept_comp_hybrid_mean"] == pytest.approx(0.7)
    txt = format_hybrid_libcomp_summary(s)
    assert "Kept (comp_hybrid > 0): 2" in txt
    assert "Dropped: 3" in txt


def test_hybrid_completeness_curves_visualization(tmp_path):
    """
    Real LEGUS observed 5D apparent limits + real cluster_slug sample + real NN:
    save per-band binned mean ``comp_nn`` vs apparent mag for in-5D clusters.
    Opens with any viewer from pytest tmp_path.

    Requires env vars:
      HYBRID_NN_SCALER, HYBRID_NN_MODEL, HYBRID_SLUG_LIB, HYBRID_LEGUS_TAB
    Optional:
      HYBRID_NN_GALAXY (default: ngc628-c)
      HYBRID_SAMPLE_N (default: 20000)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from catalog_readers import reader_register
    from completeness_io import predict_catalog_completeness_with_nn
    from hybrid_libcomp import lib_column_indices

    nn_scaler = os.environ.get("HYBRID_NN_SCALER", "").strip()
    nn_model = os.environ.get("HYBRID_NN_MODEL", "").strip()
    slug_lib = os.environ.get("HYBRID_SLUG_LIB", "").strip()
    legus_tab = os.environ.get("HYBRID_LEGUS_TAB", "").strip()
    if not nn_scaler or not nn_model or not slug_lib or not legus_tab:
        pytest.skip(
            "Set HYBRID_NN_SCALER, HYBRID_NN_MODEL, HYBRID_SLUG_LIB, HYBRID_LEGUS_TAB to run real-NN visualization test"
        )
    if (
        not Path(nn_scaler).is_file()
        or not Path(nn_model).is_file()
        or not Path(slug_lib).exists()
        or not Path(legus_tab).is_file()
    ):
        pytest.skip("HYBRID_NN_SCALER/HYBRID_NN_MODEL/HYBRID_SLUG_LIB/HYBRID_LEGUS_TAB path not found")

    galaxy_fullname = os.environ.get("HYBRID_NN_GALAXY", "ngc628-c").strip() or "ngc628-c"
    sample_n = int(os.environ.get("HYBRID_SAMPLE_N", "20000"))

    # Keep fixed UV,U,B,V,I ordering for NN input.
    filters_cat = [
        "WFC3_UVIS_F275W",
        "WFC3_UVIS_F336W",
        "ACS_F435W",
        "ACS_F555W",
        "ACS_F814W",
    ]
    nband = len(filters_cat)

    legus = reader_register["LEGUS"].read(legus_tab)
    legus_filters = [str(f) for f in legus["filters"]]
    if legus_filters != filters_cat:
        pytest.skip(f"LEGUS filter order mismatch: {legus_filters}")
    cat_dmod = float(legus["dmod"])
    cat_abs = np.asarray(legus["phot"], dtype=float)
    cat_det = np.asarray(legus["detect"], dtype=bool)
    cat_app = cat_abs + cat_dmod

    app_lo = []
    app_hi = []
    for j in range(nband):
        m = cat_det[:, j] & np.isfinite(cat_app[:, j])
        if not np.any(m):
            pytest.skip(f"No valid catalog detections for band {filters_cat[j]}")
        app_lo.append(float(np.min(cat_app[m, j])))
        app_hi.append(float(np.max(cat_app[m, j])))
    try:
        from slugpy import read_cluster
    except ImportError:
        pytest.skip("slugpy not installed")

    lib = read_cluster('/g/data/jh2/jt4478/cluster_slug/tang', photsystem="Vega", read_filters=filters_cat)
    lib_filter_names = [str(f) for f in lib.filter_names]
    phot_all = np.asarray(lib.phot_neb_ex, dtype=float)
    if phot_all.ndim != 2 or phot_all.shape[0] < 10:
        pytest.skip("cluster_slug library phot_neb_ex is empty/invalid")
    rng = np.random.default_rng(2026)
    n = min(int(sample_n), int(phot_all.shape[0]))
    idx = rng.choice(phot_all.shape[0], size=n, replace=False)
    phot_abs = phot_all[idx]
    phot_app = phot_abs + cat_dmod
    lib_cols = lib_column_indices(lib_filter_names, filters_cat)

    inside = np.ones(n, dtype=bool)
    for j in range(nband):
        x = phot_app[:, int(lib_cols[j])]
        inside &= np.isfinite(x) & (x >= app_lo[j]) & (x <= app_hi[j])
    n_inside = int(np.sum(inside))
    assert n_inside > 0

    comp_nn = np.zeros(n, dtype=float)
    idx_in = np.flatnonzero(inside)
    if idx_in.size > 0:
        sub_abs = phot_abs[np.ix_(idx_in, lib_cols)]
        sub_app = sub_abs + cat_dmod
        comp_nn[idx_in] = predict_catalog_completeness_with_nn(
            sub_app,
            galaxy_fullname=galaxy_fullname,
            nn_dir=None,
            subset_filters=filters_cat,
            full_filter_order=filters_cat,
            nn_scaler_path=nn_scaler,
            nn_model_path=nn_model,
        )

    assert comp_nn.shape == (n,)
    assert np.all((comp_nn >= 0) & (comp_nn <= 1))
    assert int(np.sum(comp_nn[~inside] > 0)) == 0

    from hybrid_libcomp import format_hybrid_libcomp_summary, hybrid_libcomp_summary

    summ = hybrid_libcomp_summary(
        n_total=n,
        inside_5d=inside,
        rule_ok=inside,
        comp_nn=comp_nn,
        comp_hybrid=comp_nn,
    )
    assert summ["n_kept"] + summ["n_dropped"] == n
    assert summ["n_kept"] == n_inside
    (tmp_path / "hybrid_libcomp_summary.txt").write_text(
        format_hybrid_libcomp_summary(summ), encoding="utf-8"
    )

    keep_mask = inside
    out_dir = tmp_path / "hybrid_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bins = 36

    fig_all, axes = plt.subplots(1, nband, figsize=(14, 3.2), sharey=True)
    for j, fname in enumerate(filters_cat):
        mag = phot_app[keep_mask, int(lib_cols[j])]
        lo_p, hi_p = np.percentile(mag[np.isfinite(mag)], [0.5, 99.5])
        bins = np.linspace(lo_p, hi_p, n_bins + 1)
        xc, ym = _binned_mean(mag, comp_nn[keep_mask], bins)
        ax = axes[j]
        ax.plot(xc, ym, "b.-", lw=1.2, label="NN (in 5D)")
        ax.axvline(app_hi[j], color="r", ls="--", lw=1, label="max_obs app")
        ax.axvline(app_lo[j], color="orange", ls=":", lw=1, label="min_obs app")
        ax.set_xlabel(fname.split("_")[-1], fontsize=8)
        ax.set_ylabel("mean comp" if j == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, loc="best")
        ax.tick_params(axis="x", labelsize=7)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xc, ym, "b.-", lw=1.5, label="NN (binned mean, in 5D)")
        ax1.axvline(app_hi[j], color="r", ls="--", label="catalog faint limit (app)")
        ax1.axvline(app_lo[j], color="orange", ls=":", label="catalog bright end (app)")
        ax1.set_xlabel(f"library apparent magnitude ({fname})")
        ax1.set_ylabel("mean completeness in bin")
        ax1.set_title(f"NN completeness curve — {fname}\n(inside 5D observed range, N={n_inside})")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        fig1.tight_layout()
        safe = "".join(c if c.isalnum() else "_" for c in fname)
        fig1.savefig(out_dir / f"hybrid_comp_curve_{safe}.png", dpi=144)
        plt.close(fig1)

    axes[0].set_ylabel("mean completeness")
    fig_all.suptitle("NN vs apparent mag (inside 5D observed range)", fontsize=11)
    fig_all.tight_layout()
    fig_all.savefig(out_dir / "hybrid_comp_curves_all_bands.png", dpi=144)
    plt.close(fig_all)

    assert len(list(out_dir.glob("hybrid_comp_curve_*.png"))) == nband
    assert (out_dir / "hybrid_comp_curves_all_bands.png").is_file()
    assert (tmp_path / "hybrid_libcomp_summary.txt").is_file()


def test_hybrid_all_library_interp_plus1mag_print_summary():
    """
    Hybrid test on all library clusters:
    - read real LEGUS catalog 5D ABS ranges
    - expand by +1 mag margin (both bright/faint edges, same as range_margin logic)
    - run NN only inside this interpolation box
    - outside box libcomp is 0
    - print kept/dropped counts

    Requires env vars:
      HYBRID_NN_SCALER, HYBRID_NN_MODEL, HYBRID_SLUG_LIB, HYBRID_LEGUS_TAB
    Optional:
      HYBRID_NN_GALAXY (default: ngc628-c), HYBRID_RANGE_MARGIN (default: 1.0),
      HYBRID_VMAG_ABS_CUT (default: -6.0)
    """
    from catalog_readers import reader_register
    from hybrid_libcomp import (
        HybridLegusLibCompletenessCalculator,
        format_hybrid_libcomp_summary,
        legus_catalog_abs_bounds,
    )

    nn_scaler = os.environ.get("HYBRID_NN_SCALER", "").strip()
    nn_model = os.environ.get("HYBRID_NN_MODEL", "").strip()
    slug_lib = os.environ.get("HYBRID_SLUG_LIB", "").strip()
    legus_tab = os.environ.get("HYBRID_LEGUS_TAB", "").strip()
    if not nn_scaler or not nn_model or not slug_lib or not legus_tab:
        pytest.skip(
            "Set HYBRID_NN_SCALER, HYBRID_NN_MODEL, HYBRID_SLUG_LIB, HYBRID_LEGUS_TAB to run this test"
        )
    if (
        not Path(nn_scaler).is_file()
        or not Path(nn_model).is_file()
        or not Path(slug_lib).exists()
        or not Path(legus_tab).is_file()
    ):
        pytest.skip("HYBRID_NN_SCALER/HYBRID_NN_MODEL/HYBRID_SLUG_LIB/HYBRID_LEGUS_TAB path not found")

    try:
        from slugpy import read_cluster
    except ImportError:
        pytest.skip("slugpy not installed")

    galaxy_fullname = os.environ.get("HYBRID_NN_GALAXY", "ngc628-c").strip() or "ngc628-c"
    range_margin = float(os.environ.get("HYBRID_RANGE_MARGIN", "1.0"))
    vmag_abs_cut = float(os.environ.get("HYBRID_VMAG_ABS_CUT", "-6.0"))

    data = reader_register["LEGUS"].read(legus_tab)
    filters_cat = [str(f) for f in data["filters"]]
    dmod = float(data["dmod"])
    bounds_lo, bounds_hi = legus_catalog_abs_bounds(
        np.asarray(data["phot"], dtype=float),
        np.asarray(data["detect"], dtype=bool),
        filters_cat,
        range_margin=range_margin,
    )

    lib = read_cluster('/g/data/jh2/jt4478/cluster_slug/tang', photsystem="Vega", read_filters=filters_cat)
    lib_filter_names = [str(f) for f in lib.filter_names]
    phot_neb_ex = np.asarray(lib.phot_neb_ex, dtype=float)
    ntot = int(phot_neb_ex.shape[0])
    assert ntot > 0

    calc = HybridLegusLibCompletenessCalculator(
        filters_cat,
        bounds_lo,
        bounds_hi,
        lib_filter_names,
        nn_scaler,
        nn_model,
        lib_vmag_max=-6.0,
        min_bands_nonzero=4,
        nn_batch_rows=65536,
    )
    out = calc.compute(phot_neb_ex, dmod=dmod, galaxy_fullname=galaxy_fullname)

    comp_nn = out["comp_nn"]
    inside = out["inside_5d"]
    summ = out["summary"]

    n_inside = int(np.sum(inside))
    n_keep_nn = int(np.sum(comp_nn > 0))
    n_drop_nn = int(ntot - n_keep_nn)
    assert n_inside == n_keep_nn

    v_idx = None
    for cand in ("ACS_F555W", "WFC3_UVIS_F555W"):
        if cand in lib_filter_names:
            v_idx = lib_filter_names.index(cand)
            break
    if v_idx is None:
        for i, fname in enumerate(lib_filter_names):
            if str(fname).endswith("F555W"):
                v_idx = i
                break
    assert v_idx is not None, "Could not resolve V band in library filters"

    v_abs = phot_neb_ex[:, v_idx]
    vcut_mask = np.isfinite(v_abs) & (v_abs > vmag_abs_cut)
    comp_nn_vcut = np.array(comp_nn, copy=True)
    pre_pos = comp_nn_vcut > 0
    comp_nn_vcut[vcut_mask] = 0.0
    post_pos = comp_nn_vcut > 0
    n_cut_by_v = int(np.sum(pre_pos & ~post_pos))
    n_keep_after_vcut = int(np.sum(post_pos))

    print("\n[hybrid interp +1mag report]")
    print(f"total clusters: {ntot}")
    print(f"inside 5D (+/-{range_margin:.2f} mag): {n_inside}")
    print(f"kept (comp_nn>0): {n_keep_nn}")
    print(f"applied V absolute cut: M_V > {vmag_abs_cut:.3f} -> comp=0")
    print(f"cut by M_V rule: {n_cut_by_v}")
    print(f"kept after M_V cut: {n_keep_after_vcut}")
    print(f"dropped (outside 5D): {n_drop_nn}")
    print(format_hybrid_libcomp_summary(summ))
