#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import warnings
from pathlib import Path

import emcee
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

import catalog_readers as catalog_readers_mod  # noqa: E402
from catalog_readers import reader_register  # noqa: E402
from clean_legus import clean_legus  # noqa: E402
from completeness_io import _nn_scaler_model_paths  # noqa: E402
from hybrid_libcomp import (  # noqa: E402
    HybridLegusLibCompletenessCalculator,
    inside_5d_abs_box,
    lib_column_indices,
    legus_catalog_abs_bounds,
)

OUTDIR = ROOT / "output_io"
NN_DIR = str(ROOT / "nn_models")
CLUSTER_SLUG_LIB_DIR = "/g/data/jh2/jt4478/cluster_slug"
OLD_ROOT = Path("/g/data/jh2/jt4478/legus_slug23")

TANG_CATALOGS = [
    Path("/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-c/hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab"),
    Path("/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-e/hlsp_legus_hst_acs-wfc3_ngc628-e_multiband_v1_padagb-mwext-avgapcor.tab"),
]
OLD_CATALOGS = [
    OLD_ROOT / "cluster_catalogs/hlsp_628c.tab",
    OLD_ROOT / "cluster_catalogs/hlsp_628e.tab",
]

SCENARIOS = [
    {
        "name": "newread_maxuvfill_hybrid",
        "title": "new reader + max UV fill + hybrid pobs",
        "chain": ROOT / "output_chains/ngc628_mid_newread_maxuvfill_tangnn_hybrid.h5",
        "lib_prefix": "tang_padova",
        "reader": "newread",
        "pobs_mode": "hybrid",
        "clean_threshold": 1e-12,
        "enforce_hybrid_clean": False,
        "lib_vmag_max": -6.0,
        "range_margin": 0.05,
    },
    {
        "name": "tangreader_observed_box",
        "title": "old Tang24 reader + observed-box pobs",
        "chain": ROOT / "output_chains/ngc628_mid_pobs_observed_box.h5",
        "lib_prefix": "tang",
        "reader": "tang",
        "pobs_mode": "observed-box",
        "clean_threshold": 0.01,
        "enforce_hybrid_clean": False,
        "lib_vmag_max": -6.0,
        "range_margin": 0.05,
    },
]

LIB_STRIDE = int(os.environ.get("NGC628_LIB_STRIDE", "10"))
CHUNK_ROWS = int(os.environ.get("NGC628_CHUNK_ROWS", "500000"))
NBIN = 42


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short_filter(filt: str) -> str:
    m = re.search(r"(F\d+W)", str(filt))
    return m.group(1) if m else str(filt)


def filt_wave_key(filt: str) -> tuple[int, int, str]:
    m = re.search(r"F(\d+)W", str(filt))
    if m:
        return (0, int(m.group(1)), str(filt))
    return (1, 9999, str(filt))


def is_uv_or_u(filt: str) -> bool:
    s = str(filt).upper()
    return "F275W" in s or "F336W" in s


def corrected_missing_uv_u_fills(filters_cat, phot, detect, dmod, full_filter_order, subset_filters, use_apparent_magnitude=True):
    subset_set = {str(f) for f in subset_filters}
    names = [str(f) for f in filters_cat]
    phot = np.asarray(phot, dtype=float)
    detect = np.asarray(detect, dtype=bool)
    fills = {}
    for f in full_filter_order:
        fs = str(f)
        if fs in subset_set or not is_uv_or_u(fs) or fs not in names:
            continue
        j = names.index(fs)
        ok = detect[:, j] & np.isfinite(phot[:, j])
        if not np.any(ok):
            continue
        mag = phot[ok, j]
        if use_apparent_magnitude:
            mag = mag + float(dmod)
        fills[fs] = float(np.max(mag)) + 0.5
    return fills


clean_legus.__globals__["legus_nn_missing_uv_u_fills"] = corrected_missing_uv_u_fills


def read_legus_hlsp_no_err_cut(fname, classcut=(0, 3.5)):
    readme_path = catalog_readers_mod._find_hlsp_readme_for_catalog(str(fname))
    dmod, filters, first_mag_index, class_index, ra_index, _debug = catalog_readers_mod._parse_hlsp_readme_metadata(readme_path)
    data = catalog_readers_mod.asc.read(str(fname))
    cid = np.array(data["col1"], dtype=int)
    nc = len(cid)
    nf = len(filters)
    phot = np.zeros((nc, nf))
    photerr = np.zeros((nc, nf))
    detect = np.ones((nc, nf), dtype=bool)
    nondetect_flags = np.array([44.444, 66.666, 99.999])
    for i in range(nf):
        mag_obs = np.array(data[f"col{2 * i + first_mag_index}"], dtype=float)
        err_obs = np.array(data[f"col{2 * i + first_mag_index + 1}"], dtype=float)
        is_flagged = np.isclose(mag_obs[:, None], nondetect_flags[None, :], atol=1e-6).any(axis=1)
        bad = is_flagged | ~np.isfinite(mag_obs) | ~np.isfinite(err_obs)
        detect[:, i] = ~bad
        phot[:, i] = np.where(detect[:, i], mag_obs - dmod, np.nan)
        photerr[:, i] = np.where(detect[:, i], err_obs, np.nan)
    ra = np.array(data[f"col{ra_index}"], dtype=float)
    dec = np.array(data[f"col{ra_index + 1}"], dtype=float)
    classification = np.array(data[f"col{class_index}"], dtype=int)
    keep_class = (classification > classcut[0]) & (classification < classcut[1])
    v_idx = None
    for token in ("F555W", "F606W"):
        for i, f in enumerate(filters):
            if token in f:
                v_idx = i
                break
        if v_idx is not None:
            break
    if v_idx is None:
        raise ValueError(f"No V-like filter in {fname}: {filters}")
    b_candidates = [i for i, f in enumerate(filters) if "F435W" in f or "F438W" in f]
    i_candidates = [i for i, f in enumerate(filters) if "F814W" in f]
    b_detect = detect[:, b_candidates].any(axis=1) if b_candidates else np.zeros(nc, dtype=bool)
    i_detect = detect[:, i_candidates].any(axis=1) if i_candidates else np.zeros(nc, dtype=bool)
    keep = keep_class & detect[:, v_idx] & np.where(detect[:, v_idx], phot[:, v_idx] < -6.0, False) & (np.sum(detect, axis=1) >= 4) & (b_detect | i_detect)
    return {
        "path": str(fname),
        "basename": fname.stem,
        "galaxy_fullname": catalog_readers_mod._infer_galaxy_fullname_from_path(str(fname)),
        "cid": cid[keep],
        "phot": phot[keep],
        "photerr": photerr[keep],
        "detect": detect[keep],
        "filters": filters,
        "dmod": dmod,
        "ra": ra[keep],
        "dec": dec[keep],
        "class": classification[keep],
    }


def prepare_filtersets(cat: dict) -> None:
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(cat["phot"]), dtype=int)
    filters = np.asarray(cat["filters"])
    for i, detected in enumerate(cat["detect"]):
        fs = list(filters[detected])
        if fs not in filtersets:
            filtersets.append(fs)
            filtersets_detect.append(np.copy(detected))
        fset[i] = filtersets.index(fs)
    cat["filtersets"] = filtersets
    cat["filtersets_index"] = fset
    cat["filtersets_detect"] = filtersets_detect
    cat["cid_filterset"] = []
    cat["phot_filterset"] = []
    cat["photerr_filterset"] = []
    for i, detected in enumerate(filtersets_detect):
        idx = fset == i
        cat["cid_filterset"].append(cat["cid"][idx])
        cat["phot_filterset"].append(cat["phot"][idx][:, detected])
        cat["photerr_filterset"].append(cat["photerr"][idx][:, detected])


def read_catalogs(reader_kind: str) -> list[dict]:
    os.environ["LEGUS_CCT_ROOT"] = "/g/data/jh2/jt4478/make_LEGUS_CCT"
    os.environ["LEGUS_TAB_DIR"] = str(ROOT / "cluster_data")
    catalogs = []
    for path in TANG_CATALOGS:
        if reader_kind == "newread":
            cat = read_legus_hlsp_no_err_cut(path)
        elif reader_kind == "tang":
            cat = reader_register["LEGUS"].read(str(path))
        else:
            raise ValueError(reader_kind)
        prepare_filtersets(cat)
        catalogs.append(cat)
    return catalogs


def best_params(chain_path: Path) -> tuple[np.ndarray, float]:
    backend = emcee.backends.HDFBackend(str(chain_path), read_only=True)
    chain = backend.get_chain(flat=True)
    logp = backend.get_log_prob(flat=True)
    ok = np.isfinite(logp)
    idx = np.flatnonzero(ok)[np.argmax(logp[ok])]
    return np.asarray(chain[idx], dtype=float), float(logp[idx])


class LibWgts:
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
        self.pav[-1] = 2.0 / self.delta_av - self.pav[-2] - np.sum(self.pav[:-2] + self.pav[1:-1])

    def wgts(self, physprop: np.ndarray) -> np.ndarray:
        logm = physprop[:, 0]
        logt = physprop[:, 1]
        av = physprop[:, 2]
        mass = 10.0 ** logm
        time_eval = 10.0 ** logt
        wgt = mass ** (self.alpha_m + 1.0) * np.exp(-mass / self.m_break)
        wgt *= np.where(time_eval <= self.t_mid, time_eval / self.t_mid, (time_eval / self.t_mid) ** (self.alpha_t + 1.0))
        for i in range(self.nav):
            avlo = self.av_grid[i]
            avhi = self.av_grid[i + 1]
            pavlo = self.pav[i]
            pavhi = self.pav[i + 1]
            avslope = (pavhi - pavlo) / (avhi - avlo)
            in_bin = (av >= avlo) & (av < avhi)
            wgt *= np.where(in_bin, pavlo + (av - avlo) * avslope, 1.0)
        return wgt


class SampleDen:
    def __init__(self, lib_dir: str):
        self.mpdf = slug_pdf(f"{lib_dir}/lib_mass.pdf")
        self.tpdf = slug_pdf(f"{lib_dir}/lib_time.pdf")
        self.avpdf = slug_pdf(f"{lib_dir}/lib_av.pdf")

    def sample_den(self, physprop: np.ndarray) -> np.ndarray:
        mass = 10.0 ** physprop[:, 0]
        mass = np.clip(mass, self.mpdf.bkpts[0], self.mpdf.bkpts[-1])
        age = 10.0 ** physprop[:, 1]
        av = physprop[:, 2]
        return self.mpdf(mass) * self.tpdf(age) * self.avpdf(av) * mass * age


def build_specs(catalogs: list[dict], filter_names: list[str], scenario: dict) -> tuple[list[dict], np.ndarray]:
    specs = []
    n_obs = []
    for cat in catalogs:
        galaxy = cat.get("galaxy_fullname", cat["basename"])
        scaler_path, model_path = _nn_scaler_model_paths(NN_DIR, galaxy, None, None)
        phot_cat_full = np.asarray(cat["phot"], dtype=float)
        detect_cat_full = np.asarray(cat["detect"], dtype=bool)
        for d, phot_fs in zip(cat["filtersets_detect"], cat["phot_filterset"]):
            subset_filters = sorted([str(f) for f in np.asarray(cat["filters"])[d]], key=filt_wave_key)
            all_cat_filters = [str(f) for f in cat["filters"]]
            nn_full_order = sorted(all_cat_filters, key=filt_wave_key) if set(subset_filters) < set(all_cat_filters) else subset_filters
            fills = corrected_missing_uv_u_fills(cat["filters"], phot_cat_full, detect_cat_full, float(cat["dmod"]), nn_full_order, subset_filters, True)
            bounds_lo, bounds_hi = legus_catalog_abs_bounds(phot_cat_full[:, d], detect_cat_full[:, d], subset_filters, range_margin=float(scenario["range_margin"]))
            calc = HybridLegusLibCompletenessCalculator(
                subset_filters,
                bounds_lo,
                bounds_hi,
                filter_names,
                scaler_path,
                model_path,
                lib_vmag_max=float(scenario["lib_vmag_max"]),
                min_bands_nonzero=min(4, len(subset_filters)),
                nn_full_filter_order=[str(f) for f in cat["filters"]],
                nn_batch_rows=65536,
                nn_missing_band_fills=fills if fills else None,
            )
            specs.append({
                "cat": cat["basename"],
                "galaxy": galaxy,
                "dmod": float(cat["dmod"]),
                "subset_filters": subset_filters,
                "lib_cols": lib_column_indices(filter_names, subset_filters),
                "calc": calc,
            })
            n_obs.append(len(phot_fs))
    n_obs = np.asarray(n_obs, dtype=float)
    return specs, n_obs / np.sum(n_obs)


def prepare_observed_hist(catalogs: list[dict], filters: list[str]):
    obs_vals = {f: [] for f in filters}
    for cat in catalogs:
        for f in filters:
            if f not in list(cat["filters"]):
                continue
            j = list(cat["filters"]).index(f)
            ok = np.asarray(cat["detect"][:, j], dtype=bool) & np.isfinite(cat["phot"][:, j])
            obs_vals[f].append(np.asarray(cat["phot"][ok, j], dtype=float))
    obs_vals = {f: np.concatenate(v) if v else np.array([], dtype=float) for f, v in obs_vals.items()}
    edges = {}
    centers = {}
    hobs = {}
    herr = {}
    for f, vals in obs_vals.items():
        lo = float(np.nanmin(vals)) - 0.75
        hi = float(np.nanmax(vals)) + 0.75
        e = np.linspace(lo, hi, NBIN + 1)
        c = 0.5 * (e[:-1] + e[1:])
        counts, _ = np.histogram(vals, bins=e)
        bw = np.diff(e)
        norm = max(float(np.sum(counts)), 1.0)
        edges[f] = e
        centers[f] = c
        hobs[f] = counts / (norm * bw)
        herr[f] = np.sqrt(counts) / (norm * bw)
    return obs_vals, edges, centers, hobs, herr


def model_hist_for_scenario(scenario: dict, catalogs: list[dict], filters: list[str], edges: dict[str, np.ndarray], params: np.ndarray):
    lib_prefix = scenario["lib_prefix"]
    phot_fits = Path(CLUSTER_SLUG_LIB_DIR) / f"{lib_prefix}_cluster_phot.fits"
    prop_fits = Path(CLUSTER_SLUG_LIB_DIR) / f"{lib_prefix}_cluster_prop.fits"
    specs, ncr = build_specs(catalogs, filters, scenario)
    lib_den = SampleDen(CLUSTER_SLUG_LIB_DIR)
    libwgts = LibWgts(params)
    counts = {f: np.zeros(len(edges[f]) - 1, dtype=float) for f in filters}
    n_processed = 0
    with fits.open(phot_fits, memmap=True) as phot_hdul, fits.open(prop_fits, memmap=True) as prop_hdul:
        phot_tab = phot_hdul[1].data
        prop_tab = prop_hdul[1].data
        n_total = len(prop_tab)
        for start in range(0, n_total, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n_total)
            sl = slice(start, stop, LIB_STRIDE)
            mass = np.asarray(prop_tab["TargetMass"][sl], dtype=float)
            t = np.asarray(prop_tab["Time"][sl], dtype=float)
            av = np.asarray(prop_tab["A_V"][sl], dtype=float)
            phot = np.column_stack([np.asarray(phot_tab[f"{f}_neb_ex"][sl], dtype=float) for f in filters])
            phys = np.column_stack([np.log10(mass), np.log10(t), av])
            base = libwgts.wgts(phys) / lib_den.sample_den(phys)
            base = np.where(np.isfinite(base) & (base > 0), base, 0.0)
            for si, spec in enumerate(specs):
                out = spec["calc"].compute(phot, dmod=spec["dmod"], galaxy_fullname=str(spec["galaxy"]))
                if scenario["pobs_mode"] == "hybrid":
                    comp = np.asarray(out["comp_hybrid"], dtype=float)
                elif scenario["pobs_mode"] == "observed-box":
                    comp = np.asarray(out["comp_nn"], dtype=float) * np.asarray(out["inside_5d"], dtype=float)
                else:
                    raise ValueError(scenario["pobs_mode"])
                w = base * comp * ncr[si]
                ok_w = np.isfinite(w) & (w > 0.0)
                if not np.any(ok_w):
                    continue
                for f in spec["subset_filters"]:
                    j = filters.index(f)
                    x = phot[:, j]
                    ok = ok_w & np.isfinite(x)
                    if np.any(ok):
                        h, _ = np.histogram(x[ok], bins=edges[f], weights=w[ok])
                        counts[f] += h
            n_processed += len(mass)
            print(f"[{scenario['name']}] processed sampled rows {n_processed} (raw stop {stop}/{n_total})", flush=True)
    hist = {}
    for f in filters:
        area = float(np.sum(counts[f] * np.diff(edges[f])))
        hist[f] = counts[f] / area if area > 0 else counts[f]
    return hist


def plot_obs_vs_model():
    warnings.filterwarnings("ignore")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    metadata = {"library_stride": LIB_STRIDE, "chunk_rows": CHUNK_ROWS, "scenarios": []}
    row_data = []
    for scenario in SCENARIOS:
        print(f"[scenario] {scenario['name']}", flush=True)
        params, logp = best_params(scenario["chain"])
        metadata["scenarios"].append({**{k: str(v) if isinstance(v, Path) else v for k, v in scenario.items()}, "best_logp": logp, "best_params": params.tolist()})
        catalogs = read_catalogs(scenario["reader"])
        clean_legus(
            catalogs,
            verbose=False,
            nn_dir=NN_DIR,
            nn_scaler_path=None,
            nn_model_path=None,
            comp_threshold=float(scenario["clean_threshold"]),
            enforce_hybrid_criteria=bool(scenario["enforce_hybrid_clean"]),
            lib_vmag_max=float(scenario["lib_vmag_max"]),
            min_bands_nonzero=4,
        )
        filters = sorted({str(f) for cat in catalogs for f in cat["filters"]}, key=filt_wave_key)
        obs_vals, edges, centers, hobs, herr = prepare_observed_hist(catalogs, filters)
        hmodel = model_hist_for_scenario(scenario, catalogs, filters, edges, params)
        row_data.append((scenario, params, logp, filters, centers, hobs, herr, hmodel, {f: len(obs_vals[f]) for f in filters}))

    filters = row_data[0][3]
    fig, axs = plt.subplots(len(row_data), len(filters), figsize=(15.5, 6.3), dpi=180, sharey=True, squeeze=False)
    colors = {"obs": "#0018a9", "model": "#d62728"}
    for ri, (scenario, params, logp, filters, centers, hobs, herr, hmodel, nobs) in enumerate(row_data):
        for ci, f in enumerate(filters):
            ax = axs[ri, ci]
            x = centers[f]
            ax.plot(x, hobs[f], color=colors["obs"], lw=1.25, label="observed")
            ax.errorbar(x, hobs[f], yerr=herr[f], fmt="none", ecolor=colors["obs"], elinewidth=0.7, capsize=1.3, alpha=0.75)
            ax.plot(x, hmodel[f], color=colors["model"], lw=1.35, label="library+pobs")
            ax.set_yscale("log")
            ymax = max(1.0, float(max(np.nanmax(hobs[f]), np.nanmax(hmodel[f]))) * 1.5)
            ax.set_ylim(1e-3, ymax)
            ax.invert_xaxis()
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            if ri == 0:
                ax.set_title(f"{short_filter(f)}\nN={nobs[f]}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{scenario['title']}\nPDF", fontsize=8)
            if ri == len(row_data) - 1:
                ax.set_xlabel("absolute magnitude", fontsize=8)
            ax.text(0.03, 0.06, f"$\\alpha_M$={params[0]:.2f}\nlogL={logp:.1f}", transform=ax.transAxes, fontsize=7, va="bottom")
    axs[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("NGC628 observed vs library-predicted magnitudes at max-logL params", y=0.997, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.965), w_pad=0.55, h_pad=0.75)
    out_png = OUTDIR / "ngc628_newread_hybrid_vs_observedbox_bestparams_observed_vs_library.png"
    out_pdf = OUTDIR / "ngc628_newread_hybrid_vs_observedbox_bestparams_observed_vs_library.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    (OUTDIR / "ngc628_newread_hybrid_vs_observedbox_bestparams_observed_vs_library.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[write] {out_png}", flush=True)
    print(f"[write] {out_pdf}", flush=True)


def uv_values(cats: list[dict], band: str) -> np.ndarray:
    vals = []
    for c in cats:
        for i, f in enumerate(c["filters"]):
            if band in str(f):
                ok = np.asarray(c["detect"][:, i], dtype=bool) & np.isfinite(c["phot"][:, i])
                vals.append(np.asarray(c["phot"][ok, i], dtype=float))
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def plot_uv_lf():
    with_err_cut = read_catalogs("tang")
    no_err_cut = read_catalogs("newread")
    bands = ["F275W", "F336W"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=220, sharey=True)
    summary = {}
    for ax, band in zip(axs, bands):
        cut_v = uv_values(with_err_cut, band)
        nocut_v = uv_values(no_err_cut, band)
        if len(cut_v) == 0 or len(nocut_v) == 0:
            raise RuntimeError(f"No values available for {band}")
        lo = min(np.nanmin(cut_v), np.nanmin(nocut_v)) - 0.4
        hi = max(np.nanmax(cut_v), np.nanmax(nocut_v)) + 0.4
        bins = np.arange(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2 + 0.25, 0.25)
        ax.hist(
            cut_v,
            bins=bins,
            histtype="step",
            density=True,
            lw=2.0,
            color="#1f77b4",
            label=f"old reader, err<=0.3 (N={len(cut_v)})",
        )
        ax.hist(
            nocut_v,
            bins=bins,
            histtype="step",
            density=True,
            lw=2.0,
            color="#d62728",
            label=f"new reader, no err cut (N={len(nocut_v)})",
        )
        ax.axvline(-6.0, color="0.35", ls=":", lw=1.0)
        ax.invert_xaxis()
        ax.set_title(band)
        ax.set_xlabel("absolute magnitude")
        ax.set_ylabel("normalized density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        summary[band] = {
            "with_err_cut_n": int(len(cut_v)),
            "no_err_cut_n": int(len(nocut_v)),
            "with_err_cut_median": float(np.median(cut_v)),
            "no_err_cut_median": float(np.median(nocut_v)),
            "with_err_cut_n_faint_m6": int(np.sum(cut_v > -6.0)),
            "no_err_cut_n_faint_m6": int(np.sum(nocut_v > -6.0)),
        }
    fig.suptitle("NGC628 observed UV/U LF: 0.3 mag error cut vs no error cut", y=0.99)
    fig.tight_layout()
    out_png = OUTDIR / "ngc628_reader_errcut_vs_noerrcut_uv_u_lf_normalized.png"
    out_pdf = OUTDIR / "ngc628_reader_errcut_vs_noerrcut_uv_u_lf_normalized.pdf"
    out_json = OUTDIR / "ngc628_reader_errcut_vs_noerrcut_uv_u_lf_normalized.json"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[write] {out_png}", flush=True)
    print(f"[write] {out_pdf}", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("NGC628_ONLY_UV_LF") == "1":
        plot_uv_lf()
        return
    plot_obs_vs_model()
    plot_uv_lf()


if __name__ == "__main__":
    main()
