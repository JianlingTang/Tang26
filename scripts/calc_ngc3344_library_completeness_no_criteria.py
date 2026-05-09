#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = Path("/g/data/jh2/jt4478/Tang26B")
BUNDLE = ROOT / "pipeline" / "bundled_pipeline"
sys.path.insert(0, str(BUNDLE))
os.chdir(BUNDLE)

from catalog_readers import reader_register  # noqa: E402
from completeness_io import predict_catalog_completeness_with_nn  # noqa: E402


GALAXY = "ngc3344"
CAT_PATH = (
    "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc3344/"
    "hlsp_legus_hst_wfc3_ngc3344_multiband_v1_padagb-mwext-avgapcor.tab"
)
PHOT_FITS = Path("/g/data/jh2/jt4478/cluster_slug/tang_cluster_phot.fits")
SCALER_PATH = ROOT / "nn_models/ngc3344/scaler_phot_use_nn_gpu_DELTAV_2MAG_ngc3344.pkl"
MODEL_PATH = ROOT / "nn_models/ngc3344/best_model_phot_use_nn_gpu_DELTAV_2MAG_ngc3344.pt"
OUTDIR = ROOT / "output_io"
OUT_NPY = OUTDIR / "ngc3344_library_completeness_no_criteria.npy"
OUT_NPZ = OUTDIR / "ngc3344_library_completeness_no_criteria_summary.npz"
OUT_PNG = OUTDIR / "ngc3344_library_completeness_no_criteria_vs_vmag.png"


FILTERS = [
    "WFC3_UVIS_F275W",
    "WFC3_UVIS_F336W",
    "WFC3_UVIS_F438W",
    "WFC3_UVIS_F555W",
    "WFC3_UVIS_F814W",
]
V_FILTER = "WFC3_UVIS_F555W"


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cat = reader_register["LEGUS"].read(CAT_PATH)
    dmod = float(cat["dmod"])
    print(f"[catalog] galaxy={GALAXY} dmod={dmod}", flush=True)
    print(f"[nn] scaler={SCALER_PATH}", flush=True)
    print(f"[nn] model={MODEL_PATH}", flush=True)

    edges = np.linspace(-12.0, -4.0, 65)
    centers = 0.5 * (edges[:-1] + edges[1:])
    count = np.zeros(len(centers), dtype=np.int64)
    finite_count = np.zeros(len(centers), dtype=np.int64)
    comp_sum = np.zeros(len(centers), dtype=np.float64)

    chunk_rows = 200_000
    pred_rows = 65_536
    v_idx = FILTERS.index(V_FILTER)
    n_finite_input = 0
    n_nonfinite_input = 0

    with fits.open(PHOT_FITS, memmap=True) as phot_hdul:
        phot_tab = phot_hdul[1].data
        n_total = len(phot_tab)
        comp_all = np.lib.format.open_memmap(
            OUT_NPY,
            mode="w+",
            dtype=np.float32,
            shape=(n_total,),
        )
        comp_all[:] = np.nan
        print(f"[library] rows={n_total} phot={PHOT_FITS}", flush=True)

        for start in range(0, n_total, chunk_rows):
            stop = min(start + chunk_rows, n_total)
            sl = slice(start, stop)
            phot_abs = np.column_stack(
                [np.asarray(phot_tab[f"{f}_neb_ex"][sl], dtype=np.float64) for f in FILTERS]
            )
            vmag_abs = phot_abs[:, v_idx]
            finite_v = np.isfinite(vmag_abs)
            hist_count, _ = np.histogram(vmag_abs[finite_v], bins=edges)
            count += hist_count

            finite_input = np.all(np.isfinite(phot_abs), axis=1)
            n_finite_input += int(np.sum(finite_input))
            n_nonfinite_input += int(finite_input.size - np.sum(finite_input))
            local_comp = np.full(stop - start, np.nan, dtype=np.float32)
            idx = np.flatnonzero(finite_input)

            for p0 in range(0, idx.size, pred_rows):
                rows = idx[p0 : p0 + pred_rows]
                phot_app = phot_abs[rows] + dmod
                local_comp[rows] = predict_catalog_completeness_with_nn(
                    phot_app,
                    galaxy_fullname=GALAXY,
                    nn_dir=None,
                    subset_filters=FILTERS,
                    full_filter_order=FILTERS,
                    nn_scaler_path=str(SCALER_PATH),
                    nn_model_path=str(MODEL_PATH),
                ).astype(np.float32)

            comp_all[start:stop] = local_comp
            ok = finite_v & np.isfinite(local_comp)
            hist_finite, _ = np.histogram(vmag_abs[ok], bins=edges)
            hist_sum, _ = np.histogram(vmag_abs[ok], bins=edges, weights=local_comp[ok])
            finite_count += hist_finite
            comp_sum += hist_sum

            done = stop
            if done == n_total or done % 1_000_000 == 0:
                print(f"[library] processed {done}/{n_total}", flush=True)

        comp_all.flush()

    mean_comp = safe_ratio(comp_sum, finite_count)
    np.savez(
        OUT_NPZ,
        galaxy=GALAXY,
        dmod=dmod,
        filters=np.array(FILTERS),
        scaler_path=str(SCALER_PATH),
        model_path=str(MODEL_PATH),
        phot_fits=str(PHOT_FITS),
        completeness_npy=str(OUT_NPY),
        vmag_edges=edges,
        vmag_centers=centers,
        count=count,
        finite_input_count=finite_count,
        completeness_sum=comp_sum,
        mean_completeness=mean_comp,
        n_total=n_total,
        n_finite_input=n_finite_input,
        n_nonfinite_input=n_nonfinite_input,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.plot(centers, mean_comp, color="#1f77b4", lw=1.8)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("SLUG F555W absolute magnitude [mag]")
    ax.set_ylabel("Mean NN library completeness")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    finite_values = np.load(OUT_NPY, mmap_mode="r")
    valid = np.isfinite(finite_values)
    print(
        "[summary] "
        f"finite_input={n_finite_input}/{n_total} "
        f"nonfinite_input={n_nonfinite_input} "
        f"mean={float(np.nanmean(finite_values[valid])):.6g} "
        f"median={float(np.nanmedian(finite_values[valid])):.6g}",
        flush=True,
    )
    print(f"[write] {OUT_NPY}", flush=True)
    print(f"[write] {OUT_NPZ}", flush=True)
    print(f"[write] {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
