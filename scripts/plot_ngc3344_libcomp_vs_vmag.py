#!/usr/bin/env python
from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path("/g/data/jh2/jt4478/Tang26B")
BUNDLE = ROOT / "pipeline" / "bundled_pipeline"
sys.path.insert(0, str(BUNDLE))
os.chdir(BUNDLE)

from catalog_readers import reader_register  # noqa: E402
from clean_legus import clean_legus  # noqa: E402
from completeness_io import (  # noqa: E402
    _nn_scaler_model_paths,
    legus_nn_missing_uv_u_fills,
)
from hybrid_libcomp import (  # noqa: E402
    HybridLegusLibCompletenessCalculator,
    legus_catalog_abs_bounds,
)


GALAXY = "ngc3344"
LIBDIR = "/g/data/jh2/jt4478/cluster_slug/tang"
CLUSTER_SLUG_LIB_DIR = "/g/data/jh2/jt4478/cluster_slug"
CAT_PATH = (
    "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc3344/"
    "hlsp_legus_hst_wfc3_ngc3344_multiband_v1_padagb-mwext-avgapcor.tab"
)
NN_DIR = str(ROOT / "nn_models")
OUTDIR = ROOT / "output_io"
OUT_PNG = OUTDIR / "ngc3344_libcomp_vs_vmag_libcomp_noperband_critera.png"
OUT_PDF = OUTDIR / "ngc3344_libcomp_vs_vmag_libcomp_noperband_critera.pdf"
OUT_NPZ = OUTDIR / "ngc3344_libcomp_vs_vmag_libcomp_noperband_critera.npz"
PHOT_FITS = Path("/g/data/jh2/jt4478/cluster_slug/tang_cluster_phot.fits")
V_FILTER = "WFC3_UVIS_F555W"


def filt_wave_key(filt: str) -> tuple[int, int, str]:
    match = re.search(r"F(\d+)W", str(filt))
    if match is not None:
        return (0, int(match.group(1)), str(filt))
    return (1, 9999, str(filt))


def short_filterset(filters: list[str]) -> str:
    return "+".join(re.search(r"(F\d+W)", f).group(1) for f in filters)


def prepare_catalog() -> dict:
    cat = reader_register["LEGUS"].read(CAT_PATH)
    cat["galaxy_fullname"] = GALAXY
    filtersets: list[list[str]] = []
    filtersets_detect: list[np.ndarray] = []
    fset = np.zeros(len(cat["phot"]))
    for i, d in enumerate(cat["detect"]):
        filt = list(np.array(cat["filters"])[d])
        if filt not in filtersets:
            filtersets.append(filt)
            filtersets_detect.append(np.copy(d))
        fset[i] = filtersets.index(filt)
    cat["filtersets"] = filtersets
    cat["filtersets_index"] = fset
    cat["filtersets_detect"] = filtersets_detect
    cat["cid_filterset"] = []
    cat["phot_filterset"] = []
    cat["photerr_filterset"] = []
    for i, d in enumerate(filtersets_detect):
        idx = fset == i
        cat["cid_filterset"].append(cat["cid"][idx])
        cat["phot_filterset"].append(cat["phot"][idx][:, d])
        cat["photerr_filterset"].append(cat["photerr"][idx][:, d])
    return cat


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    os.environ["LEGUS_CCT_ROOT"] = "/g/data/jh2/jt4478/make_LEGUS_CCT"
    os.environ["LEGUS_TAB_DIR"] = str(ROOT / "cluster_data")
    os.environ["CLUSTER_SLUG_LIB_DIR"] = CLUSTER_SLUG_LIB_DIR
    os.environ["CLUSTER_SLUG_LIB_NAME"] = LIBDIR

    print("[read] catalog", flush=True)
    catalogs = [prepare_catalog()]
    clean_legus(
        catalogs,
        verbose=False,
        nn_dir=NN_DIR,
        nn_scaler_path=None,
        nn_model_path=None,
        comp_threshold=0.0,
        enforce_hybrid_criteria=True,
        lib_vmag_max=-6.0,
        min_bands_nonzero=4,
    )
    cat = catalogs[0]
    print(f"[catalog] after clean n={len(cat['phot'])}", flush=True)

    allfilters = sorted(list(cat["filters"]), key=filt_wave_key)
    filter_names = allfilters
    scaler_path, model_path = _nn_scaler_model_paths(NN_DIR, GALAXY, None, None)
    print(f"[nn] scaler={scaler_path}", flush=True)
    print(f"[nn] model={model_path}", flush=True)

    calculators = []
    for d in cat["filtersets_detect"]:
        requested_filters = [str(f) for f in np.array(cat["filters"])[d]]
        subset_filters = sorted(requested_filters, key=filt_wave_key)
        if V_FILTER not in subset_filters:
            continue
        all_cat_filters = [str(f) for f in cat["filters"]]
        nn_full_order = (
            sorted(all_cat_filters, key=filt_wave_key)
            if set(subset_filters) < set(all_cat_filters)
            else subset_filters
        )
        dmod = float(cat["dmod"])
        phot_cat_full = np.asarray(cat["phot"], dtype=float)
        detect_cat_full = np.asarray(cat["detect"], dtype=bool)
        nn_uv_u_fills = legus_nn_missing_uv_u_fills(
            cat["filters"],
            phot_cat_full,
            detect_cat_full,
            dmod,
            nn_full_order,
            subset_filters,
            use_apparent_magnitude=True,
        )
        bounds_lo, bounds_hi = legus_catalog_abs_bounds(
            phot_cat_full[:, d],
            detect_cat_full[:, d],
            subset_filters,
            range_margin=0.5,
        )
        calculators.append(
            (
                subset_filters,
                HybridLegusLibCompletenessCalculator(
                    subset_filters,
                    bounds_lo,
                    bounds_hi,
                    filter_names,
                    scaler_path,
                    model_path,
                    lib_vmag_max=-6.0,
                    min_bands_nonzero=min(4, len(subset_filters)),
                    nn_full_filter_order=[str(f) for f in cat["filters"]],
                    nn_batch_rows=65536,
                    nn_missing_band_fills=nn_uv_u_fills,
                ),
            )
        )

    edges = np.linspace(-12.0, -4.0, 65)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nbin = len(centers)
    stats = {}
    for k, (subset_filters, _calc) in enumerate(calculators):
        stats[k] = {
            "count": np.zeros(nbin, dtype=float),
            "comp_hybrid": np.zeros(nbin, dtype=float),
            "comp_nn": np.zeros(nbin, dtype=float),
            "inside_5d": np.zeros(nbin, dtype=float),
            "rule_ok": np.zeros(nbin, dtype=float),
        }

    v_idx = allfilters.index(V_FILTER)
    chunk_rows = 200_000
    n_seen = 0
    print(f"[read] chunked SLUG library phot={PHOT_FITS.name}", flush=True)
    with fits.open(PHOT_FITS, memmap=True) as phot_hdul:
        phot_tab = phot_hdul[1].data
        n_total = len(phot_tab)
        print(f"[library] rows={n_total} filters={filter_names}", flush=True)
        for start in range(0, n_total, chunk_rows):
            stop = min(start + chunk_rows, n_total)
            sl = slice(start, stop)
            phot_neb_ex = np.column_stack(
                [np.asarray(phot_tab[f"{f}_neb_ex"][sl], dtype=float) for f in allfilters]
            )
            vmag = phot_neb_ex[:, v_idx]
            finite_v = np.isfinite(vmag)
            for k, (_subset_filters, calc) in enumerate(calculators):
                out = calc.compute(phot_neb_ex, dmod=float(cat["dmod"]), galaxy_fullname=GALAXY)
                bin_count, _ = np.histogram(vmag[finite_v], bins=edges)
                stats[k]["count"] += bin_count
                for key in ("comp_hybrid", "comp_nn", "inside_5d", "rule_ok"):
                    vals = np.asarray(out[key], dtype=float)
                    ok = finite_v & np.isfinite(vals)
                    sums, _ = np.histogram(vmag[ok], bins=edges, weights=vals[ok])
                    stats[k][key] += sums
            n_seen += stop - start
            if n_seen == n_total or n_seen % 1_000_000 == 0:
                print(f"[library] processed {n_seen}/{n_total}", flush=True)

    curves = {}
    for k in stats:
        count = stats[k]["count"]
        curves[k] = {
            key: safe_ratio(stats[k][key], count)
            for key in ("comp_hybrid", "comp_nn", "inside_5d", "rule_ok")
        }
        print(
            f"[summary] {short_filterset(calculators[k][0])}: "
            f"inside_rows={np.nansum(stats[k]['inside_5d']):.0f}",
            flush=True,
        )

    np.savez(
        OUT_NPZ,
        vmag_centers=centers,
        vmag_edges=edges,
        filtersets=np.array([short_filterset(fs) for fs, _calc in calculators]),
        **{
            f"{key}_fs{k}": curves[k][key]
            for k in curves
            for key in ("comp_hybrid", "comp_nn", "inside_5d", "rule_ok")
        },
        **{f"count_fs{k}": stats[k]["count"] for k in stats},
    )

    fig, axs = plt.subplots(len(calculators), 1, figsize=(7.2, 6.4), dpi=180, sharex=True)
    if len(calculators) == 1:
        axs = [axs]
    for k, ax in enumerate(axs):
        label = short_filterset(calculators[k][0])
        ax.plot(centers, curves[k]["comp_hybrid"], color="#ed1c23", lw=1.7, label="comp_hybrid")
        ax.plot(centers, curves[k]["comp_nn"], color="#7b3294", lw=1.3, label="NN pobs")
        ax.plot(
            centers,
            curves[k]["inside_5d"],
            color="#0018a9",
            lw=1.5,
            label="inside_5d (noperband libcomp)",
        )
        ax.plot(centers, curves[k]["rule_ok"], color="#00843d", lw=1.2, label="rule_ok")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.24)
        ax.text(0.03, 0.9, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylabel("Mean libcomp", fontsize=9)
    axs[0].legend(loc="lower left", fontsize=8, frameon=True)
    axs[-1].set_xlabel("SLUG F555W absolute magnitude [mag]", fontsize=10)
    axs[-1].invert_xaxis()
    fig.suptitle("NGC3344 library completeness vs V magnitude", fontsize=12)
    fig.tight_layout(rect=(0.06, 0.05, 1, 0.96))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {OUT_PNG}", flush=True)
    print(f"[write] {OUT_PDF}", flush=True)
    print(f"[write] {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
