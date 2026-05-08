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
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
from slugpy import slug_pdf


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


PARAMS = np.array(
    [
        -0.62551775,
        3.3421738,
        -0.56350507,
        6.76797074,
        -0.1376423,
        -0.73181411,
        -1.04168456,
        -1.77589126,
        -1.92182861,
        -0.34082426,
    ],
    dtype=float,
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
OUT_PNG = OUTDIR / "ngc3344_observed_vs_slug_predicted_magnitudes_nn_pobs.png"
OUT_PDF = OUTDIR / "ngc3344_observed_vs_slug_predicted_magnitudes_nn_pobs.pdf"
OUT_NPZ = OUTDIR / "ngc3344_observed_vs_slug_predicted_magnitudes_nn_pobs_hist.npz"
PHOT_FITS = Path("/g/data/jh2/jt4478/cluster_slug/tang_cluster_phot.fits")
PROP_FITS = Path("/g/data/jh2/jt4478/cluster_slug/tang_cluster_prop.fits")


class LibWgts:
    """Numpy equivalent of analyze_catalog_mid_mdd.libwgts for MID."""

    def __init__(self, p: np.ndarray):
        self.alpha_m = float(p[0])
        self.m_break = 10.0 ** float(p[1])
        self.alpha_t = float(p[2])
        self.t_mid = 10.0 ** float(p[3])
        self.nav = len(p) - 4
        self.delta_av = 3.0 / self.nav
        self.av_grid = np.arange(0.0, 3.0 + self.delta_av / 2.0, self.delta_av)
        self.pav = np.zeros(self.nav + 1)
        self.pav[:-1] = 10.0 ** np.asarray(p[4:], dtype=float)
        self.pav[-1] = (
            2.0 / self.delta_av
            - self.pav[-2]
            - np.sum(self.pav[:-2] + self.pav[1:-1])
        )

    def wgts(self, physprop: np.ndarray) -> np.ndarray:
        logm = physprop[:, 0]
        logt = physprop[:, 1]
        av = physprop[:, 2]
        logt_mid = np.log10(self.t_mid)
        mass_term = (10.0 ** ((self.alpha_m + 1.0) * logm)) * np.exp(
            -(10.0**logm) / self.m_break
        )
        time_term = np.where(
            logt <= logt_mid,
            10.0**logt / self.t_mid,
            (10.0**logt / self.t_mid) ** (self.alpha_t + 1.0),
        )
        wgt = mass_term * time_term
        for i in range(self.nav):
            avlo = self.av_grid[i]
            avhi = self.av_grid[i + 1]
            pavlo = self.pav[i]
            pavhi = self.pav[i + 1]
            avslope = (pavhi - pavlo) / (avhi - avlo)
            in_bin = (av >= avlo) & (av < avhi)
            wgt = wgt * np.where(in_bin, pavlo + (av - avlo) * avslope, 1.0)
        return wgt


class SampleDen:
    def __init__(self, mpdf, tpdf, avpdf):
        self.mpdf = mpdf
        self.tpdf = tpdf
        self.avpdf = avpdf

    def sample_den(self, physprop: np.ndarray) -> np.ndarray:
        mass = 10.0 ** physprop[:, 0]
        mass = np.clip(mass, self.mpdf.bkpts[0], self.mpdf.bkpts[-1])
        age = 10.0 ** physprop[:, 1]
        av = physprop[:, 2]
        return self.mpdf(mass) * self.tpdf(age) * self.avpdf(av) * mass * age


def filt_wave_key(filt: str) -> tuple[int, int, str]:
    match = re.search(r"F(\d+)W", str(filt))
    if match is not None:
        return (0, int(match.group(1)), str(filt))
    return (1, 9999, str(filt))


def short_filter(filt: str) -> str:
    match = re.search(r"(F\d+W)", str(filt))
    return match.group(1) if match else str(filt)


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
    cat = catalogs[0]
    print(
        f"[catalog] filters={cat['filters']} dmod={cat['dmod']} n={len(cat['phot'])}",
        flush=True,
    )
    print(
        "[catalog] filtersets="
        + repr([(fs, len(p)) for fs, p in zip(cat["filtersets"], cat["phot_filterset"])]),
        flush=True,
    )

    clean_legus(
        catalogs,
        verbose=False,
        nn_dir=NN_DIR,
        nn_scaler_path=None,
        nn_model_path=None,
        comp_threshold=0.01,
        enforce_hybrid_criteria=True,
        lib_vmag_max=-6.0,
        min_bands_nonzero=4,
    )
    cat = catalogs[0]
    print(f"[catalog] after clean n={len(cat['phot'])}", flush=True)

    print("[weights] sample density + MID best params", flush=True)
    lib_den = SampleDen(
        slug_pdf(f"{CLUSTER_SLUG_LIB_DIR}/lib_mass.pdf"),
        slug_pdf(f"{CLUSTER_SLUG_LIB_DIR}/lib_time.pdf"),
        slug_pdf(f"{CLUSTER_SLUG_LIB_DIR}/lib_av.pdf"),
    )
    libwgts = LibWgts(PARAMS)

    allfilters = sorted(list(cat["filters"]), key=filt_wave_key)
    filter_names = allfilters
    scaler_path, model_path = _nn_scaler_model_paths(NN_DIR, GALAXY, None, None)
    print(f"[nn] scaler={scaler_path}", flush=True)
    print(f"[nn] model={model_path}", flush=True)
    calculators = []
    for d in cat["filtersets_detect"]:
        requested_filters = [str(f) for f in np.array(cat["filters"])[d]]
        subset_filters = sorted(requested_filters, key=filt_wave_key)
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
        calc = HybridLegusLibCompletenessCalculator(
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
        )
        calculators.append((subset_filters, calc))

    n_per_fs = np.array([len(p) for p in cat["phot_filterset"]], dtype=float)
    ncr = n_per_fs / np.sum(n_per_fs)
    nbin = 45
    hist_obs: dict[str, np.ndarray] = {}
    model_counts: dict[str, np.ndarray] = {}
    bin_centers: dict[str, np.ndarray] = {}
    bin_edges: dict[str, np.ndarray] = {}

    for filt in allfilters:
        cat_idx = list(cat["filters"]).index(filt)
        obs_mask = np.asarray(cat["detect"][:, cat_idx], dtype=bool) & np.isfinite(
            cat["phot"][:, cat_idx]
        )
        obs = np.asarray(cat["phot"][obs_mask, cat_idx], dtype=float)
        lo = float(np.nanmin(obs)) - 0.75
        hi = float(np.nanmax(obs)) + 0.75
        edges = np.linspace(lo, hi, nbin + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        h_obs, _ = np.histogram(obs, bins=edges, density=True)
        hist_obs[filt] = h_obs
        model_counts[filt] = np.zeros(nbin, dtype=float)
        bin_centers[filt] = centers
        bin_edges[filt] = edges

    chunk_rows = 200_000
    n_seen = 0
    pobs_kept = [0 for _ in calculators]
    pobs_positive_sum = [0.0 for _ in calculators]
    pobs_positive_n = [0 for _ in calculators]
    print(
        f"[read] chunked SLUG library phot={PHOT_FITS.name} prop={PROP_FITS.name}",
        flush=True,
    )
    with fits.open(PHOT_FITS, memmap=True) as phot_hdul, fits.open(PROP_FITS, memmap=True) as prop_hdul:
        phot_tab = phot_hdul[1].data
        prop_tab = prop_hdul[1].data
        n_total = len(prop_tab)
        print(f"[library] rows={n_total} filters={filter_names}", flush=True)
        for start in range(0, n_total, chunk_rows):
            stop = min(start + chunk_rows, n_total)
            sl = slice(start, stop)
            target_mass = np.asarray(prop_tab["TargetMass"][sl], dtype=float)
            eval_time = np.asarray(prop_tab["Time"][sl], dtype=float)
            a_v = np.asarray(prop_tab["A_V"][sl], dtype=float)
            phot_neb_ex = np.column_stack(
                [np.asarray(phot_tab[f"{f}_neb_ex"][sl], dtype=float) for f in allfilters]
            )
            phys = np.column_stack([np.log10(target_mass), np.log10(eval_time), a_v])
            base_w = libwgts.wgts(phys) / lib_den.sample_den(phys)
            base_w = np.where(np.isfinite(base_w) & (base_w > 0.0), base_w, 0.0)
            for k, (subset_filters, calc) in enumerate(calculators):
                out = calc.compute(phot_neb_ex, dmod=float(cat["dmod"]), galaxy_fullname=GALAXY)
                comp = out["comp_hybrid"]
                positive = comp > 0.0
                pobs_kept[k] += int(np.sum(comp >= 0.01))
                pobs_positive_sum[k] += float(np.sum(comp[positive]))
                pobs_positive_n[k] += int(np.sum(positive))
                weights = base_w * comp * ncr[k]
                ok_w = np.isfinite(weights) & (weights > 0.0)
                if not np.any(ok_w):
                    continue
                for filt in subset_filters:
                    lib_idx = allfilters.index(filt)
                    x = phot_neb_ex[:, lib_idx]
                    ok = ok_w & np.isfinite(x)
                    if np.any(ok):
                        counts, _ = np.histogram(
                            x[ok], bins=bin_edges[filt], weights=weights[ok]
                        )
                        model_counts[filt] += counts
            n_seen += stop - start
            if n_seen == n_total or n_seen % 1_000_000 == 0:
                print(f"[library] processed {n_seen}/{n_total}", flush=True)

    hist_model: dict[str, np.ndarray] = {}
    for filt in allfilters:
        area = float(np.sum(model_counts[filt] * np.diff(bin_edges[filt])))
        hist_model[filt] = model_counts[filt] / area if area > 0.0 else model_counts[filt]

    for k, (subset_filters, _calc) in enumerate(calculators):
        mean_positive = (
            pobs_positive_sum[k] / pobs_positive_n[k] if pobs_positive_n[k] > 0 else 0.0
        )
        print(
            f"[pobs] {subset_filters} kept>0.01={pobs_kept[k]} "
            f"mean_positive={mean_positive:.4g}",
            flush=True,
        )

    np.savez(
        OUT_NPZ,
        filters=np.array(allfilters),
        params=PARAMS,
        **{f"obs_{short_filter(f)}": hist_obs[f] for f in allfilters},
        **{f"model_{short_filter(f)}": hist_model[f] for f in allfilters},
        **{f"centers_{short_filter(f)}": bin_centers[f] for f in allfilters},
    )

    labels = [short_filter(f) for f in allfilters]
    fig, axs = plt.subplots(len(allfilters), 1, figsize=(4.2, 8.4), dpi=220, sharex=False)
    for ax, filt, label in zip(axs, allfilters, labels):
        x = bin_centers[filt]
        obs = hist_obs[filt]
        model = hist_model[filt]
        ax.plot(x, obs, color="#0018a9", lw=1.3, label="Observed")
        ax.fill_between(x, np.maximum(obs, 1e-8), color="#0018a9", alpha=0.16)
        ax.plot(x, model, color="#ed1c23", lw=1.4, label="SLUG predicted")
        ax.fill_between(x, np.maximum(model, 1e-8), color="#ed1c23", alpha=0.16)
        ax.set_yscale("log")
        ymax = max(1.2, float(max(np.nanmax(obs), np.nanmax(model))) * 1.5)
        ax.set_ylim(1e-3, ymax)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.text(0.97, 0.78, label, transform=ax.transAxes, ha="right", fontsize=9)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[0].legend(loc="upper left", fontsize=7, frameon=True)
    fig.suptitle(
        "NGC3344 observed vs SLUG predicted magnitudes\nMID best params, NN hybrid pobs",
        fontsize=10,
        y=0.995,
    )
    fig.text(0.5, 0.035, "Absolute magnitude [mag]", ha="center", fontsize=10)
    fig.text(0.035, 0.5, "PDF", ha="center", rotation=90, fontsize=10)
    fig.tight_layout(rect=(0.07, 0.055, 1, 0.965))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {OUT_PNG}", flush=True)
    print(f"[write] {OUT_PDF}", flush=True)
    print(f"[write] {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
