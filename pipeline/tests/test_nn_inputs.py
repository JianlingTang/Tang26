#!/usr/bin/env python3
"""
Quick HPC-side diagnostics for NN completeness inputs.

Checks:
1) cluster_slug library photometry for non-finite / extreme values
2) optional scaler.transform reproducibility on selected filter columns

Example:
  python scripts/diagnose_nn_inputs.py \
    /g/data/jh2/jt4478/cluster_slug/tang \
    --filters WFC3_UVIS_F275W WFC3_UVIS_F336W ACS_F435W ACS_F555W ACS_F814W \
    --nn-scaler /g/data/jh2/jt4478/Tang26B/nn_models/ngc628-c_scaler.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

read_cluster = pytest.importorskip("slugpy").read_cluster

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "bundled_pipeline"))
from completeness_io import _load_pickle  # noqa: E402


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
        default=None,
        help="optional scaler pickle/joblib path; run scaler.transform check if set",
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
    p.add_argument(
        "--distance-mpc",
        type=float,
        default=None,
        help=(
            "if set, also print approximate apparent mags using "
            "m = M + 5*(log10(d_pc)-1)"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print("Loading cluster_slug library...")
    lib = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=args.filters)
    x = np.asarray(lib.phot_neb_ex, dtype=np.float64)
    print("=== Raw library phot_neb_ex diagnostics ===")
    bad_total = _summarize_matrix(x, args.filters, args.huge_threshold)

    if args.print_rows > 0:
        n_show = min(args.print_rows, x.shape[0])
        print(f"\n=== First {n_show} rows of phot_neb_ex (as stored) ===")
        for i in range(n_show):
            print(f"row={i} {x[i]}")
        if args.distance_mpc is not None:
            d_pc = float(args.distance_mpc) * 1.0e6
            dist_mod = 5.0 * (np.log10(d_pc) - 1.0)
            print(
                f"\n=== Same rows interpreted as absolute M and converted to apparent m "
                f"(distance={args.distance_mpc} Mpc, dist_mod={dist_mod:.5f}) ==="
            )
            for i in range(n_show):
                m_app = x[i] + dist_mod
                print(f"row={i} M={x[i]}  ->  m~={m_app}")

    if bad_total > 0 and args.show_bad_rows > 0:
        bad_rows = np.where(np.any(~np.isfinite(x), axis=1))[0]
        print(f"first bad row ids ({min(len(bad_rows), args.show_bad_rows)} shown):")
        for ridx in bad_rows[: args.show_bad_rows]:
            row = x[ridx]
            print(f"  row={int(ridx)} values={row}")

    if args.nn_scaler:
        print("\n=== scaler.transform diagnostics ===")
        scaler = _load_pickle(args.nn_scaler)
        x_try = x.copy()
        non_finite = ~np.isfinite(x_try)
        n_non_finite = int(np.sum(non_finite))
        if n_non_finite > 0:
            if hasattr(scaler, "mean_") and len(np.asarray(scaler.mean_).ravel()) == x_try.shape[1]:
                mean_row = np.asarray(scaler.mean_, dtype=np.float64).ravel()
                fill = np.broadcast_to(mean_row, x_try.shape)
                x_try[non_finite] = fill[non_finite]
                print(f"imputed {n_non_finite} non-finite values with scaler.mean_ before transform")
            else:
                x_try[non_finite] = 0.0
                print(f"imputed {n_non_finite} non-finite values with 0 before transform")
        try:
            x_scaled = scaler.transform(x_try)
            print(f"transform OK: scaled shape={x_scaled.shape}, finite={np.isfinite(x_scaled).all()}")
        except Exception as exc:  # noqa: BLE001
            print(f"transform FAILED: {type(exc).__name__}: {exc}")
            return 2

    if bad_total > 0:
        print("\nDIAGNOSIS: non-finite values detected in library photometry.")
        print("Suggestion: keep NN fallback/sanitization or filter bad rows before scaler.transform.")
        return 1

    print("\nDIAGNOSIS: no non-finite values found in selected filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
