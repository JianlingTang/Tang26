#!/usr/bin/env python3
"""
Quick HPC-side diagnostics for NN completeness inputs.

Checks:
1) cluster_slug library photometry for non-finite / extreme values
2) convert absolute -> apparent magnitudes via distance modulus
3) filter non-finite rows and run scaler + NN inference
4) save V-band (x) vs binned completeness (y) figure for visual sanity checks

Example:
  python scripts/diagnose_nn_inputs.py \
    /g/data/jh2/jt4478/cluster_slug/tang \
    --filters WFC3_UVIS_F275W WFC3_UVIS_F336W ACS_F435W ACS_F555W ACS_F814W \
    --distance-modulus 29.98 \
    --nn-scaler /g/data/jh2/jt4478/Tang26B/nn_models/scaler_phot_ngc628-c.pkl \
    --nn-model /g/data/jh2/jt4478/Tang26B/nn_models/best_model_phot_ngc628-c.pt \
    --output-plot /g/data/jh2/jt4478/Tang26B/output_io/diag_libcomp_vs_vmag_ngc628c.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from slugpy import read_cluster

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "bundled_pipeline"))
from completeness_io import _load_pickle, predict_catalog_completeness_with_nn  # noqa: E402


def _summarize_matrix(x: np.ndarray, names: list[str], huge_threshold: float) -> int:
    bad_total = int(np.sum(~np.isfinite(x)))
    print(f"rows={x.shape[0]} cols={x.shape[1]}")
    print(f"non-finite total: {bad_total}")
    print(f"|value| > {huge_threshold:g}: {int(np.sum(np.abs(x) > huge_threshold))}")
    for j, name in enumerate(names):
        col = x[:, j]
        finite = np.isfinite(col)
        n_nan = int(np.sum(np.isnan(col)))
        n_inf = int(np.sum(np.isinf(col)))
        if np.any(finite):
            cmin = float(np.min(col[finite]))
            cmax = float(np.max(col[finite]))
            print(
                f"  {name:<22} nan={n_nan:<8} inf={n_inf:<8} "
                f"min={cmin: .6e} max={cmax: .6e}"
            )
        else:
            print(f"  {name:<22} nan={n_nan:<8} inf={n_inf:<8} min=NA max=NA")
    return bad_total


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "libdir",
        help="cluster_slug library path passed to slugpy.read_cluster()",
    )
    p.add_argument(
        "--filters",
        nargs="+",
        required=True,
        help="filter names to read and diagnose (same names used in catalogs)",
    )
    p.add_argument(
        "--photsystem",
        default="Vega",
        help="photsystem passed to read_cluster()",
    )
    p.add_argument(
        "--nn-scaler",
        required=True,
        help="scaler pickle/joblib path used by NN completeness model",
    )
    p.add_argument(
        "--nn-model",
        required=True,
        help="NN model path (.pt checkpoint)",
    )
    p.add_argument(
        "--distance-modulus",
        type=float,
        required=True,
        help="distance modulus used to convert library absolute mags to apparent mags",
    )
    p.add_argument(
        "--output-plot",
        required=True,
        help="output PNG path for binned completeness-vs-Vmag plot",
    )
    p.add_argument(
        "--v-filter",
        default="ACS_F555W",
        help="filter name to use on x-axis (V-band proxy)",
    )
    p.add_argument(
        "--nbins",
        type=int,
        default=30,
        help="number of magnitude bins in output plot",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="cap rows for quick diagnostics (0 = all rows)",
    )
    p.add_argument(
        "--huge-threshold",
        type=float,
        default=1.0e6,
        help="flag values with abs(value) above this threshold",
    )
    p.add_argument(
        "--show-bad-rows",
        type=int,
        default=5,
        help="print up to N offending row indices with any non-finite values",
    )
    p.add_argument(
        "--print-rows",
        type=int,
        default=5,
        help="print first N phot_neb_ex rows for manual sanity check",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print("Loading cluster_slug library...")
    lib = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=args.filters)
    x_abs = np.asarray(lib.phot_neb_ex, dtype=np.float64)
    if args.max_rows > 0 and x_abs.shape[0] > args.max_rows:
        x_abs = x_abs[: args.max_rows]
        print(f"[diag] truncated to first {x_abs.shape[0]} rows (--max-rows)")
    x = x_abs
    print("=== Raw library phot_neb_ex diagnostics ===")
    bad_total = _summarize_matrix(x, args.filters, args.huge_threshold)

    if args.print_rows > 0:
        n_show = min(args.print_rows, x.shape[0])
        print(f"\n=== First {n_show} rows of phot_neb_ex (as stored) ===")
        for i in range(n_show):
            print(f"row={i} {x[i]}")
        print(
            f"\n=== Same rows converted to apparent m "
            f"(distance modulus={args.distance_modulus:.5f}) ==="
        )
        for i in range(n_show):
            m_app = x[i] + args.distance_modulus
            print(f"row={i} M={x[i]}  ->  m={m_app}")

    if bad_total > 0 and args.show_bad_rows > 0:
        bad_rows = np.where(np.any(~np.isfinite(x), axis=1))[0]
        print(f"first bad row ids ({min(len(bad_rows), args.show_bad_rows)} shown):")
        for ridx in bad_rows[: args.show_bad_rows]:
            row = x[ridx]
            print(f"  row={int(ridx)} values={row}")

    print("\n=== apparent conversion + finite-row filtering ===")
    x_app = x + args.distance_modulus
    finite_rows = np.all(np.isfinite(x_app), axis=1)
    n_bad_rows = int(np.sum(~finite_rows))
    print(f"non-finite rows after conversion: {n_bad_rows} / {len(finite_rows)}")
    x_app_finite = x_app[finite_rows]
    if x_app_finite.size == 0:
        print("No finite rows available for NN inference.")
        return 2

    print("\n=== scaler/NN inference diagnostics ===")
    try:
        scaler = _load_pickle(args.nn_scaler)
        _ = scaler.transform(x_app_finite[: min(1000, len(x_app_finite))])
        print("scaler.transform sanity check: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"scaler.transform FAILED: {type(exc).__name__}: {exc}")
        return 2

    try:
        comp = np.zeros(len(finite_rows), dtype=float)
        comp[finite_rows] = predict_catalog_completeness_with_nn(
            x_app_finite,
            galaxy_fullname="diagnostic",
            nn_dir=None,
            subset_filters=args.filters,
            full_filter_order=args.filters,
            nn_scaler_path=args.nn_scaler,
            nn_model_path=args.nn_model,
        )
        print(
            f"NN inference OK: n={len(comp)} finite={np.isfinite(comp).all()} "
            f"range=[{float(np.min(comp)):.6f}, {float(np.max(comp)):.6f}]"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"NN inference FAILED: {type(exc).__name__}: {exc}")
        return 2

    if args.v_filter not in args.filters:
        print(f"v-filter '{args.v_filter}' not found in --filters {args.filters}")
        return 2
    v_idx = args.filters.index(args.v_filter)
    vmag = x_app[:, v_idx]
    ok = np.isfinite(vmag)
    vmag = vmag[ok]
    comp_ok = comp[ok]

    edges = np.linspace(float(np.min(vmag)), float(np.max(vmag)), args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.full(args.nbins, np.nan, dtype=float)
    nbin = np.zeros(args.nbins, dtype=int)
    for i in range(args.nbins):
        if i < args.nbins - 1:
            sel = (vmag >= edges[i]) & (vmag < edges[i + 1])
        else:
            sel = (vmag >= edges[i]) & (vmag <= edges[i + 1])
        n = int(np.sum(sel))
        nbin[i] = n
        if n > 0:
            y[i] = float(np.mean(comp_ok[sel]))

    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, y, marker="o", linewidth=1.5, label="Mean completeness")
    ax.set_xlabel(f"{args.v_filter} apparent magnitude")
    ax.set_ylabel("Completeness")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(centers, nbin, width=(edges[1] - edges[0]) * 0.9, alpha=0.15, color="gray")
    ax2.set_ylabel("Count per bin")
    ax.set_title("Library V-mag vs NN completeness (binned)")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)
    print(f"saved plot: {out_plot}")

    print("\nDIAGNOSIS: completed (absolute->apparent conversion, non-finite filtering, scaler+NN check, plot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
