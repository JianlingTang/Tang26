#!/usr/bin/env python3
"""
Quick pruning diagnostic for library completeness thresholding.

This script mirrors the core pruning logic used in analyze_catalog_mid_mdd.py:
1) read cluster_slug library photometry (absolute mags)
2) convert to apparent mags using distance modulus
3) drop non-finite rows before NN/scaler inference
4) compute NN completeness and apply keep = (libcomp >= comp_threshold)

Example:
  python scripts/test_lib_pruning.py \
    /g/data/jh2/jt4478/cluster_slug/tang \
    --filters WFC3_UVIS_F275W WFC3_UVIS_F336W ACS_F435W ACS_F555W ACS_F814W \
    --distance-modulus 29.98 \
    --nn-scaler /g/data/jh2/jt4478/Tang26B/nn_models/scaler_phot_ngc628-c.pkl \
    --nn-model /g/data/jh2/jt4478/Tang26B/nn_models/best_model_phot_ngc628-c.pt \
    --comp-threshold 0.01
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
from completeness_io import predict_catalog_completeness_with_nn  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("libdir", help="Path passed to slugpy.read_cluster().")
    p.add_argument(
        "--filters",
        nargs="+",
        required=True,
        help="Filter names for NN completeness input.",
    )
    p.add_argument(
        "--photsystem",
        default="Vega",
        help="Photometric system for read_cluster().",
    )
    p.add_argument(
        "--distance-modulus",
        type=float,
        required=True,
        help="Distance modulus to convert absolute -> apparent magnitudes.",
    )
    p.add_argument(
        "--nn-scaler",
        required=True,
        help="Scaler file used by the NN completeness model.",
    )
    p.add_argument(
        "--nn-model",
        required=True,
        help="Model checkpoint (.pt) used by the NN completeness model.",
    )
    p.add_argument(
        "--comp-threshold",
        type=float,
        default=0.01,
        help="Keep library rows with libcomp >= threshold.",
    )
    p.add_argument(
        "--lib-vmag-max",
        type=float,
        default=-6.0,
        help="Force comp=0 when absolute V-band magnitude is below this value.",
    )
    p.add_argument(
        "--v-filter",
        default="auto",
        help="V-band filter name in --filters (default: auto-detect *F555W).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 means all rows).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("Loading library...")
    lib = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=args.filters)
    x_abs = np.asarray(lib.phot_neb_ex, dtype=np.float64)
    lib_filters = [str(f) for f in lib.filter_names]
    n_init = x_abs.shape[0]
    print(f"Initial library rows: {n_init}")

    if args.max_rows > 0 and n_init > args.max_rows:
        x_abs = x_abs[: args.max_rows]
        print(f"[diag] using first {x_abs.shape[0]} rows (--max-rows)")

    x_app = x_abs + args.distance_modulus
    finite_rows = np.all(np.isfinite(x_app), axis=1)
    n_finite = int(np.sum(finite_rows))
    n_nonfinite = int(len(finite_rows) - n_finite)
    print(f"Finite rows (for NN): {n_finite}")
    print(f"Non-finite rows (auto-pruned): {n_nonfinite}")

    if n_finite == 0:
        print("No finite rows left; aborting.")
        return 2

    v_idx = None
    if args.v_filter != "auto":
        if args.v_filter not in lib_filters:
            print(f"Requested --v-filter '{args.v_filter}' not present in library filters.")
            return 2
        v_idx = lib_filters.index(args.v_filter)
    else:
        for i, fname in enumerate(lib_filters):
            if str(fname).endswith("F555W"):
                v_idx = i
                break

    faint_v_mask = np.zeros(len(finite_rows), dtype=bool)
    if v_idx is not None:
        v_abs = x_abs[:, v_idx]
        faint_v_mask = np.isfinite(v_abs) & (v_abs > args.lib_vmag_max)
        print(
            f"V-band cutoff: filter={lib_filters[v_idx]} cut={args.lib_vmag_max:.3f} "
            f"affected_rows={int(np.sum(faint_v_mask))}"
        )
    else:
        print("V-band cutoff: no *F555W filter detected, skipping this rule.")

    comp = np.zeros(len(finite_rows), dtype=float)
    comp[finite_rows] = predict_catalog_completeness_with_nn(
        x_app[finite_rows],
        galaxy_fullname="diagnostic",
        nn_dir=None,
        subset_filters=lib_filters,
        full_filter_order=lib_filters,
        nn_scaler_path=args.nn_scaler,
        nn_model_path=args.nn_model,
    )
    comp[faint_v_mask] = 0.0

    keep = comp >= args.comp_threshold
    n_keep = int(np.sum(keep))
    n_drop = int(len(keep) - n_keep)

    near_zero = (comp > 0.0) & (comp < args.comp_threshold)
    n_near_zero = int(np.sum(near_zero))
    n_exact_zero = int(np.sum(comp == 0.0))

    print("\n=== Pruning summary ===")
    print(f"comp_threshold: {args.comp_threshold:.6f}")
    print(f"Kept rows      : {n_keep}")
    print(f"Dropped rows   : {n_drop}")
    print(f"Keep fraction  : {n_keep / len(keep):.6f}")
    print(f"Dropped 0<comp<threshold: {n_near_zero}")
    print(f"Exact zeros (includes non-finite rows): {n_exact_zero}")
    print(f"Forced zero by V cutoff: {int(np.sum(faint_v_mask))}")
    print(f"Comp range     : [{float(np.min(comp)):.6f}, {float(np.max(comp)):.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
