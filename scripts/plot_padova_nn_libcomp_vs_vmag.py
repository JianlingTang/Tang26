#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline/bundled_pipeline"))
from completeness_io import predict_catalog_completeness_with_nn  # noqa: E402


DEFAULT_FILTERS = [
    "WFC3_UVIS_F275W",
    "WFC3_UVIS_F336W",
    "ACS_F435W",
    "ACS_F555W",
    "ACS_F814W",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read a SLUG Padova library, sample phot_neb_ex[::stride], "
            "predict NN completeness, and plot it against apparent magnitude "
            "from one selected column."
        )
    )
    p.add_argument(
        "--lib",
        default="/g/data/jh2/jt4478/cluster_slug/tang_padova",
        help="Path prefix passed to slugpy.read_cluster().",
    )
    p.add_argument(
        "--use-slugpy",
        action="store_true",
        help="Use slugpy.read_cluster instead of direct FITS column reads.",
    )
    p.add_argument(
        "--filters",
        nargs="+",
        default=DEFAULT_FILTERS,
        help="Filters to read, in NN training order.",
    )
    p.add_argument("--photsystem", default="Vega")
    p.add_argument("--galaxy", default="ngc628-c", help="NN model directory key.")
    p.add_argument(
        "--nn-dir",
        default=str(ROOT / "nn_models"),
        help="Directory containing NN scaler/model artifacts.",
    )
    p.add_argument(
        "--dmod",
        type=float,
        default=29.98,
        help="Distance modulus added to library absolute magnitudes.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Use phot_neb_ex[::stride].",
    )
    p.add_argument(
        "--v-col",
        type=int,
        default=-1,
        help="Column used for x-axis apparent magnitude; default matches phot_neb_ex[::100, -1].",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=65536,
        help="Rows per NN inference batch.",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "output_io/padova_nn_libcomp_vs_vmag_stride100.png"),
    )
    p.add_argument(
        "--out-npz",
        default=str(ROOT / "output_io/padova_nn_libcomp_vs_vmag_stride100.npz"),
    )
    p.add_argument(
        "--phot-cache",
        default=None,
        help=(
            "Optional sampled phot_neb_ex cache. If omitted, uses "
            "output_io/padova_phot_neb_ex_stride{stride}.npz."
        ),
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore an existing sampled photometry cache and rebuild it from FITS/slugpy.",
    )
    return p.parse_args()


def _cluster_phot_fits_path(lib_prefix: str) -> Path:
    path = Path(lib_prefix)
    if path.suffix == ".fits":
        return path
    return Path(str(path) + "_cluster_phot.fits")


def _read_phot_neb_ex_fast(lib_prefix: str, filters: list[str], stride: int) -> np.ndarray:
    phot_path = _cluster_phot_fits_path(lib_prefix)
    cols = [f"{f}_neb_ex" for f in filters]
    with fits.open(phot_path, memmap=True) as hdul:
        tab = hdul[1].data
        missing = [c for c in cols if c not in tab.names]
        if missing:
            raise KeyError(f"Missing FITS columns in {phot_path}: {missing}")
        arrays = [np.asarray(tab[c][::stride], dtype=float) for c in cols]
    return np.column_stack(arrays)


def _read_phot_neb_ex_slugpy(
    lib_prefix: str,
    photsystem: str,
    filters: list[str],
    stride: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    from slugpy import read_cluster

    lib_all = read_cluster(lib_prefix, photsystem=photsystem, read_filters=filters)
    phot_abs_all = np.asarray(lib_all.phot_neb_ex, dtype=float)
    return phot_abs_all[::stride, :], phot_abs_all.shape


def main() -> int:
    args = parse_args()
    filters = [str(f) for f in args.filters]
    stride = int(args.stride)
    if stride <= 0:
        raise ValueError("--stride must be positive")

    cache_path = (
        Path(args.phot_cache)
        if args.phot_cache
        else ROOT / "output_io" / f"padova_phot_neb_ex_stride{stride}.npz"
    )

    print(f"[read] lib={args.lib}", flush=True)
    print(f"[read] filters={filters}", flush=True)
    if cache_path.exists() and not args.refresh_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_filters = [str(f) for f in cached["filters"]]
        if cached_filters != filters:
            raise ValueError(
                f"Photometry cache filters do not match request: {cached_filters} != {filters}"
            )
        if int(cached["stride"]) != stride:
            raise ValueError(f"Photometry cache stride does not match request: {cached['stride']} != {stride}")
        phot_abs = np.asarray(cached["phot_abs"], dtype=float)
        full_shape = tuple(int(x) for x in cached["full_shape"])
        read_mode = "cache"
    else:
        if args.use_slugpy:
            phot_abs, full_shape = _read_phot_neb_ex_slugpy(
                args.lib,
                args.photsystem,
                filters,
                stride,
            )
            read_mode = "slugpy"
        else:
            phot_abs = _read_phot_neb_ex_fast(args.lib, filters, stride)
            full_shape = (phot_abs.shape[0] * stride, phot_abs.shape[1])
            read_mode = "fits-columns"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            phot_abs=phot_abs,
            filters=np.array(filters),
            stride=stride,
            full_shape=np.array(full_shape, dtype=np.int64),
        )
        print(f"[cache] wrote {cache_path}", flush=True)
    phot_app = phot_abs + float(args.dmod)

    v_col = int(args.v_col)
    if v_col < 0:
        v_col = phot_app.shape[1] + v_col
    if v_col < 0 or v_col >= phot_app.shape[1]:
        raise IndexError(f"--v-col {args.v_col} is outside photometry width {phot_app.shape[1]}")

    v_app = phot_app[:, v_col]
    finite = np.all(np.isfinite(phot_app), axis=1)
    comp = np.full(phot_app.shape[0], np.nan, dtype=float)
    idx = np.flatnonzero(finite)
    bs = max(1024, int(args.batch_rows))

    print(
        f"[sample] read_mode={read_mode} approx_full_shape={full_shape} "
        f"sampled_shape={phot_abs.shape} stride={stride}"
    )
    print(f"[x] v_col={v_col} filter={filters[v_col]} dmod={args.dmod:g}")
    if "F555W" not in filters[v_col].upper() and "F547M" not in filters[v_col].upper():
        print(f"[warn] selected x-axis filter does not look like V: {filters[v_col]}")
    print(f"[nn] galaxy={args.galaxy} finite_rows={idx.size}/{phot_app.shape[0]}")

    for start in range(0, idx.size, bs):
        rows = idx[start : start + bs]
        comp[rows] = predict_catalog_completeness_with_nn(
            phot_app[rows],
            galaxy_fullname=args.galaxy,
            nn_dir=args.nn_dir,
            subset_filters=filters,
            full_filter_order=filters,
        )

    ok = np.isfinite(v_app) & np.isfinite(comp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    ax.scatter(v_app[ok], comp[ok], s=2.5, alpha=0.22, linewidths=0)
    ax.set_xlabel(f"{filters[v_col]} apparent magnitude")
    ax.set_ylabel("Predicted NN library completeness")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        filters=np.array(filters),
        stride=stride,
        dmod=float(args.dmod),
        v_col=v_col,
        v_filter=filters[v_col],
        v_app=v_app,
        comp=comp,
        finite=finite,
    )
    print(f"[write] {out}")
    print(f"[write] {out_npz}")
    print(
        f"[summary] comp_range=[{np.nanmin(comp):.6g}, {np.nanmax(comp):.6g}] "
        f"v_range=[{np.nanmin(v_app):.6g}, {np.nanmax(v_app):.6g}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
