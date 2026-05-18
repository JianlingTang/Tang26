#!/usr/bin/env python3
"""Test missing-UV/U completeness-fill effects for NGC628.

Two diagnostics are produced, without running MCMC:

1. Library p(obs) comparison for missing-UV/U filtersets under three fill modes:
   ``current`` = existing min(detected mag)+0.5, ``faint`` = max(detected
   mag)+0.5, and ``scaler-mean`` = no explicit fill.
2. Fixed-parameter logL comparison for two alpha_M values under those fill
   modes.

Defaults target the Tang NGC628-c/e catalogs and the tang_padova library. The
library is loaded through slugpy and then optionally downsampled for fast tests.
Use ``--max-lib-rows 0`` for the full library.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from slugpy import read_cluster, slug_pdf
from slugpy.cluster_slug import cluster_slug


ROOT = Path("/g/data/jh2/jt4478/Tang26B")
BUNDLE = ROOT / "pipeline" / "bundled_pipeline"
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from catalog_readers import reader_register  # noqa: E402
from clean_legus import clean_legus  # noqa: E402
from completeness_io import (  # noqa: E402
    _nn_scaler_model_paths,
    legus_nn_missing_uv_u_fills,
    predict_catalog_completeness_with_nn,
)
from hybrid_libcomp import (  # noqa: E402
    HybridLegusLibCompletenessCalculator,
    inside_5d_abs_box,
    lib_column_indices,
    legus_catalog_abs_bounds,
)


DEFAULT_CATALOGS = [
    "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-c/"
    "hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab",
    "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-e/"
    "hlsp_legus_hst_acs-wfc3_ngc628-e_multiband_v1_padagb-mwext-avgapcor.tab",
]
DEFAULT_BASE_PARAMS = np.array(
    [
        -1.45,
        4.00,
        -0.50,
        9.00,
        -0.47712125471966244,
        -0.47712125471966244,
        -0.47712125471966244,
        -0.47712125471966244,
        -0.47712125471966244,
        -0.47712125471966244,
    ],
    dtype=float,
)
UV_U_TOKENS = ("F275W", "F336W")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogs", nargs="+", default=DEFAULT_CATALOGS)
    p.add_argument("--libdir", default="/g/data/jh2/jt4478/cluster_slug/tang_padova")
    p.add_argument("--cluster-slug-lib-dir", default="/g/data/jh2/jt4478/cluster_slug")
    p.add_argument("--nn-comp-dir", default=str(ROOT / "nn_models"))
    p.add_argument("--nn-scaler", default=None)
    p.add_argument("--nn-model", default=None)
    p.add_argument("--photsystem", default="Vega")
    p.add_argument("--pobs-mode", choices=("hybrid", "observed-box", "nn"), default="hybrid")
    p.add_argument("--comp-threshold", type=float, default=0.01)
    p.add_argument("--lib-vmag-max", type=float, default=-6.0)
    p.add_argument("--hybrid-range-margin", type=float, default=0.0)
    p.add_argument("--hybrid-min-bands-nonzero", type=int, default=4)
    p.add_argument("--hybrid-nn-batch-rows", type=int, default=65536)
    p.add_argument("--tol", type=float, default=1.0e-2)
    p.add_argument("--bwphot", type=float, default=0.1)
    p.add_argument("--bwphys", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-lib-rows", type=int, default=300_000)
    p.add_argument(
        "--fill-modes",
        nargs="+",
        choices=("current", "faint", "scaler-mean"),
        default=["current", "faint", "scaler-mean"],
    )
    p.add_argument("--faint-fill-extra", type=float, default=0.5)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.45, -2.2])
    p.add_argument(
        "--base-params",
        default=",".join(f"{x:.12g}" for x in DEFAULT_BASE_PARAMS),
        help="Comma-separated MID params; alpha_M is overwritten by --alphas.",
    )
    p.add_argument("--skip-logl", action="store_true")
    p.add_argument("--mass-pdf", default=None)
    p.add_argument("--age-pdf", default=None)
    p.add_argument("--av-pdf", default=None)
    p.add_argument(
        "--out-stem",
        default=str(ROOT / "output_io/ngc628_missing_uv_fill_hypothesis"),
    )
    return p.parse_args()


def filt_wave_key(filt: str, lib_filter_names: list[str] | None = None) -> tuple:
    m = re.search(r"F(\d+)W", str(filt))
    if m is not None:
        return (0, int(m.group(1)), str(filt))
    if lib_filter_names is not None and str(filt) in lib_filter_names:
        return (1, lib_filter_names.index(str(filt)), str(filt))
    return (2, str(filt))


def band_token(filt: str) -> str | None:
    for token in UV_U_TOKENS:
        if token in str(filt):
            return token
    return None


def parse_params(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.replace(" ", "").split(",") if x], dtype=float)
    if vals.size != 10:
        raise ValueError(f"--base-params must have 10 values; got {vals.size}")
    return vals


def prepare_filtersets(cat: dict) -> None:
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(cat["phot"]), dtype=int)
    filters = np.asarray(cat["filters"])
    for i, detected in enumerate(np.asarray(cat["detect"], dtype=bool)):
        filt = list(filters[detected])
        if filt not in filtersets:
            filtersets.append(filt)
            filtersets_detect.append(np.copy(detected))
        fset[i] = filtersets.index(filt)

    cat["filtersets"] = filtersets
    cat["filtersets_index"] = fset
    cat["filtersets_detect"] = filtersets_detect
    cat["cid_filterset"] = []
    cat["phot_filterset"] = []
    cat["photerr_filterset"] = []
    cat["detect_filterset"] = []
    for i, detected in enumerate(filtersets_detect):
        idx = fset == i
        cat["cid_filterset"].append(cat["cid"][idx])
        cat["phot_filterset"].append(cat["phot"][idx][:, detected])
        cat["photerr_filterset"].append(cat["photerr"][idx][:, detected])
        cat["detect_filterset"].append(cat["detect"][idx][:, detected])


def read_and_clean_catalogs(args: argparse.Namespace) -> list[dict]:
    os.environ["LEGUS_CCT_ROOT"] = "/g/data/jh2/jt4478/make_LEGUS_CCT"
    os.environ["LEGUS_TAB_DIR"] = str(ROOT / "cluster_data")
    catalogs = []
    for path in args.catalogs:
        cat = reader_register["LEGUS"].read(str(path))
        prepare_filtersets(cat)
        catalogs.append(cat)

    before = {c["basename"]: int(len(c["cid"])) for c in catalogs}
    clean_legus(
        catalogs,
        verbose=False,
        nn_dir=args.nn_comp_dir,
        nn_scaler_path=args.nn_scaler,
        nn_model_path=args.nn_model,
        comp_threshold=float(args.comp_threshold),
        enforce_hybrid_criteria=(args.pobs_mode == "hybrid"),
        lib_vmag_max=float(args.lib_vmag_max),
        min_bands_nonzero=int(args.hybrid_min_bands_nonzero),
    )
    after = {c["basename"]: int(len(c["cid"])) for c in catalogs}
    print(f"[catalog] before clean: {before}", flush=True)
    print(f"[catalog] after clean : {after}", flush=True)
    for cat in catalogs:
        for filt, phot in zip(cat["filtersets"], cat["phot_filterset"]):
            print(f"[catalog] {cat['basename']} {list(map(str, filt))} n={len(phot)}", flush=True)
    return catalogs


def all_catalog_filters(catalogs: list[dict]) -> list[str]:
    out: list[str] = []
    for cat in catalogs:
        for filt in cat["filters"]:
            fs = str(filt)
            if fs not in out:
                out.append(fs)
    return sorted(out, key=filt_wave_key)


def read_library(args: argparse.Namespace, filters: list[str]) -> dict:
    print(f"[library] read_cluster {args.libdir}", flush=True)
    lib_all = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=filters)
    n_total = len(lib_all.actual_mass)
    idx = np.arange(n_total)
    if 0 < int(args.max_lib_rows) < n_total:
        rng = np.random.default_rng(int(args.seed))
        idx = np.sort(rng.choice(n_total, size=int(args.max_lib_rows), replace=False))
        print(f"[library] sampled {idx.size}/{n_total} rows", flush=True)
    else:
        print(f"[library] using full rows={n_total}", flush=True)

    data = {
        "cid": np.asarray(lib_all.id)[idx],
        "actual_mass": np.asarray(lib_all.actual_mass, dtype=float)[idx],
        "eval_time": np.asarray(lib_all.time, dtype=float)[idx],
        "form_time": np.asarray(lib_all.form_time, dtype=float)[idx],
        "A_V": np.asarray(lib_all.A_V, dtype=float)[idx],
        "phot_neb_ex": np.asarray(lib_all.phot_neb_ex, dtype=float)[idx],
        "filter_names": [str(f) for f in lib_all.filter_names],
        "filter_units": list(lib_all.filter_units),
        "n_total": int(n_total),
    }
    del lib_all
    return data


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
        self.pav[-1] = 2.0 / self.delta_av - self.pav[-2] - np.sum(
            self.pav[:-2] + self.pav[1:-1]
        )

    @property
    def valid(self) -> bool:
        return bool(np.all(np.isfinite(self.pav)) and np.all(self.pav >= 0.0))

    def wgts(self, physprop: np.ndarray) -> np.ndarray:
        logm = physprop[:, 0]
        logt = physprop[:, 1]
        av = physprop[:, 2]
        mass = 10.0**logm
        time_eval = 10.0**logt
        wgt = mass ** (self.alpha_m + 1.0) * np.exp(-mass / self.m_break)
        wgt *= np.where(
            time_eval <= self.t_mid,
            time_eval / self.t_mid,
            (time_eval / self.t_mid) ** (self.alpha_t + 1.0),
        )
        for i in range(self.nav):
            avlo = self.av_grid[i]
            avhi = self.av_grid[i + 1]
            pavlo = self.pav[i]
            pavhi = self.pav[i + 1]
            slope = (pavhi - pavlo) / (avhi - avlo)
            in_bin = (av >= avlo) & (av < avhi)
            wgt *= np.where(in_bin, pavlo + (av - avlo) * slope, 1.0)
        return wgt



def missing_fills_for_mode(
    mode: str,
    cat: dict,
    nn_full_order: list[str],
    subset_filters: list[str],
    dmod: float,
    faint_fill_extra: float,
) -> dict[str, float]:
    if mode == "scaler-mean":
        return {}
    if mode == "current":
        return legus_nn_missing_uv_u_fills(
            cat["filters"],
            np.asarray(cat["phot"], dtype=float),
            np.asarray(cat["detect"], dtype=bool),
            dmod,
            nn_full_order,
            subset_filters,
            use_apparent_magnitude=True,
        )
    if mode != "faint":
        raise ValueError(f"unknown fill mode {mode}")

    subset_set = {str(f) for f in subset_filters}
    names = [str(f) for f in cat["filters"]]
    phot = np.asarray(cat["phot"], dtype=float)
    detect = np.asarray(cat["detect"], dtype=bool)
    fills: dict[str, float] = {}
    for filt in nn_full_order:
        fs = str(filt)
        if fs in subset_set or band_token(fs) is None or fs not in names:
            continue
        j = names.index(fs)
        ok = detect[:, j] & np.isfinite(phot[:, j])
        if np.any(ok):
            # Magnitudes: max is faintest. This is the hypothesis-test fill.
            fills[fs] = float(np.max(phot[ok, j]) + float(dmod) + faint_fill_extra)
    return fills


def predict_libcomp_no_criteria(
    args: argparse.Namespace,
    phot_neb_ex: np.ndarray,
    lib_indices: list[int],
    dmod: float,
    galaxy_fullname: str,
    subset_filters: list[str],
    full_filter_order: list[str],
    missing_band_fills: dict[str, float],
) -> np.ndarray:
    lib_phot_subset = np.asarray(phot_neb_ex[:, lib_indices], dtype=float) + float(dmod)
    finite_rows = np.all(np.isfinite(lib_phot_subset), axis=1)
    comp = np.zeros(len(finite_rows), dtype=float)
    idx = np.flatnonzero(finite_rows)
    bs = max(1024, int(args.hybrid_nn_batch_rows))
    for start in range(0, idx.size, bs):
        rows = idx[start : start + bs]
        comp[rows] = predict_catalog_completeness_with_nn(
            lib_phot_subset[rows],
            galaxy_fullname=galaxy_fullname,
            nn_dir=args.nn_comp_dir,
            subset_filters=subset_filters,
            full_filter_order=full_filter_order,
            nn_scaler_path=args.nn_scaler,
            nn_model_path=args.nn_model,
            missing_band_fills=missing_band_fills if missing_band_fills else None,
        )
    return comp


def build_specs(args: argparse.Namespace, catalogs: list[dict], library: dict, mode: str) -> list[dict]:
    specs: list[dict] = []
    lib_filter_names = library["filter_names"]
    scaler_cache: dict[str, tuple[str, str]] = {}
    for cat in catalogs:
        for i, detect_mask in enumerate(cat["filtersets_detect"]):
            requested = [str(f) for f in np.asarray(cat["filters"])[detect_mask]]
            subset_filters = sorted(requested, key=lambda f: filt_wave_key(f, lib_filter_names))
            all_cat_filters = [str(f) for f in cat["filters"]]
            nn_full_order = (
                sorted(all_cat_filters, key=lambda f: filt_wave_key(f, lib_filter_names))
                if set(subset_filters) < set(all_cat_filters)
                else subset_filters
            )
            dmod = float(cat["dmod"])
            fills = missing_fills_for_mode(
                mode, cat, nn_full_order, subset_filters, dmod, args.faint_fill_extra
            )
            missing_bands = [
                str(f)
                for f in nn_full_order
                if str(f) not in set(subset_filters) and band_token(str(f)) is not None
            ]
            spec = {
                "cat": cat,
                "filterset_index": i,
                "subset_filters": subset_filters,
                "nn_full_order": nn_full_order,
                "lib_indices": [lib_filter_names.index(f) for f in subset_filters],
                "missing_bands": missing_bands,
                "missing_fills": fills,
                "galaxy_fullname": str(cat.get("galaxy_fullname", cat["basename"])),
                "dmod": dmod,
            }
            if args.pobs_mode == "hybrid":
                gal = spec["galaxy_fullname"]
                if gal not in scaler_cache:
                    scaler_cache[gal] = _nn_scaler_model_paths(
                        args.nn_comp_dir, gal, args.nn_scaler, args.nn_model
                    )
                scaler_path, model_path = scaler_cache[gal]
                phot = np.asarray(cat["phot"], dtype=float)
                detect = np.asarray(cat["detect"], dtype=bool)
                bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                    phot[:, detect_mask],
                    detect[:, detect_mask],
                    subset_filters,
                    range_margin=float(args.hybrid_range_margin),
                )
                spec["calc"] = HybridLegusLibCompletenessCalculator(
                    subset_filters,
                    bounds_lo,
                    bounds_hi,
                    lib_filter_names,
                    scaler_path,
                    model_path,
                    lib_vmag_max=float(args.lib_vmag_max),
                    min_bands_nonzero=min(int(args.hybrid_min_bands_nonzero), len(subset_filters)),
                    nn_full_filter_order=nn_full_order,
                    nn_batch_rows=int(args.hybrid_nn_batch_rows),
                    nn_missing_band_fills=fills if fills else None,
                )
            specs.append(spec)
    return specs


def compute_pobs_for_mode(
    args: argparse.Namespace, catalogs: list[dict], library: dict, mode: str
) -> tuple[list[dict], list[np.ndarray]]:
    specs = build_specs(args, catalogs, library, mode)
    comps: list[np.ndarray] = []
    phot_neb_ex = library["phot_neb_ex"]
    lib_filter_names = library["filter_names"]
    print(f"[pobs] fill_mode={mode}", flush=True)
    for spec in specs:
        if args.pobs_mode == "hybrid":
            out = spec["calc"].compute(
                phot_neb_ex,
                dmod=float(spec["dmod"]),
                galaxy_fullname=str(spec["galaxy_fullname"]),
            )
            comp = np.asarray(out["comp_hybrid"], dtype=float)
            spec["comp_nn"] = np.asarray(out["comp_nn"], dtype=float)
            spec["inside_box"] = np.asarray(out["inside_5d"], dtype=bool)
        else:
            comp_nn = predict_libcomp_no_criteria(
                args,
                phot_neb_ex,
                spec["lib_indices"],
                float(spec["dmod"]),
                str(spec["galaxy_fullname"]),
                spec["subset_filters"],
                spec["nn_full_order"],
                spec["missing_fills"],
            )
            spec["comp_nn"] = comp_nn
            if args.pobs_mode == "observed-box":
                cat = spec["cat"]
                detect_mask = cat["filtersets_detect"][spec["filterset_index"]]
                bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                    np.asarray(cat["phot"], dtype=float)[:, detect_mask],
                    np.asarray(cat["detect"], dtype=bool)[:, detect_mask],
                    spec["subset_filters"],
                    range_margin=float(args.hybrid_range_margin),
                )
                lib_cols = lib_column_indices(lib_filter_names, spec["subset_filters"])
                inside = inside_5d_abs_box(phot_neb_ex, lib_cols, bounds_lo, bounds_hi)
                comp = comp_nn * inside.astype(float)
                spec["inside_box"] = inside
            else:
                comp = comp_nn
                spec["inside_box"] = np.ones(len(comp), dtype=bool)
        comps.append(comp)
        print(
            f"[pobs] {mode} {spec['cat']['basename']} fs={spec['subset_filters']} "
            f"missing={spec['missing_bands']} fill={spec['missing_fills']} "
            f"kept={int(np.sum(comp >= args.comp_threshold))}",
            flush=True,
        )
    return specs, comps


def resolve_v_index(filter_names: list[str]) -> int:
    for token in ("F555W", "F606W"):
        for i, filt in enumerate(filter_names):
            if token in str(filt):
                return i
    raise ValueError(f"no V-like filter in {filter_names}")



def summarize_pobs(
    args: argparse.Namespace,
    library: dict,
    specs_by_mode: dict[str, list[dict]],
    comps_by_mode: dict[str, list[np.ndarray]],
    out_stem: Path,
) -> list[dict]:
    rows: list[dict] = []
    phot = library["phot_neb_ex"]
    lib_filters = library["filter_names"]
    v_abs = phot[:, resolve_v_index(lib_filters)]
    base_masks = {
        "all": np.ones(len(v_abs), dtype=bool),
        "MV_lt_-8": v_abs < -8.0,
        "MV_-8_-7": (v_abs >= -8.0) & (v_abs < -7.0),
        "MV_-7_-6": (v_abs >= -7.0) & (v_abs < -6.0),
        "MV_ge_-6": v_abs >= -6.0,
    }
    for mode, specs in specs_by_mode.items():
        for spec, comp in zip(specs, comps_by_mode[mode]):
            if not spec["missing_bands"]:
                continue
            masks = dict(base_masks)
            for band in spec["missing_bands"]:
                true_abs = phot[:, lib_filters.index(band)]
                masks[f"{band}_true_gt_-6"] = true_abs > -6.0
                masks[f"{band}_true_-7_-6"] = (true_abs >= -7.0) & (true_abs < -6.0)
                masks[f"{band}_true_lt_-8"] = true_abs < -8.0
            for mask_name, mask0 in masks.items():
                mask = np.isfinite(comp) & mask0
                vals = comp[mask]
                pos = vals[vals > 0.0]
                rows.append(
                    {
                        "fill_mode": mode,
                        "catalog": spec["cat"]["basename"],
                        "filterset_index": spec["filterset_index"],
                        "subset_filters": "|".join(spec["subset_filters"]),
                        "missing_bands": "|".join(spec["missing_bands"]),
                        "missing_fills": json.dumps(spec["missing_fills"], sort_keys=True),
                        "mask": mask_name,
                        "n_rows": int(np.sum(mask)),
                        "n_positive": int(pos.size),
                        "n_ge_threshold": int(np.sum(vals >= args.comp_threshold)),
                        "frac_ge_threshold": float(np.mean(vals >= args.comp_threshold)) if vals.size else np.nan,
                        "mean_positive": float(np.mean(pos)) if pos.size else 0.0,
                        "median_positive": float(np.median(pos)) if pos.size else 0.0,
                        "p84_positive": float(np.percentile(pos, 84)) if pos.size else 0.0,
                    }
                )
    path = out_stem.with_name(out_stem.name + "_pobs_stats.csv")
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}", flush=True)
    return rows


def plot_pobs_profiles(
    library: dict,
    specs_by_mode: dict[str, list[dict]],
    comps_by_mode: dict[str, list[np.ndarray]],
    out_stem: Path,
) -> None:
    phot = library["phot_neb_ex"]
    lib_filters = library["filter_names"]
    colors = {"current": "tab:blue", "faint": "tab:red", "scaler-mean": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, token in zip(axes, UV_U_TOKENS):
        filt = next((f for f in lib_filters if token in str(f)), None)
        if filt is None:
            ax.set_axis_off()
            continue
        x = phot[:, lib_filters.index(filt)]
        bins = np.linspace(-13.0, 1.0, 42)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for mode, specs in specs_by_mode.items():
            total = np.zeros_like(centers)
            count = np.zeros_like(centers)
            for spec, comp in zip(specs, comps_by_mode[mode]):
                if filt not in spec["missing_bands"]:
                    continue
                ok = np.isfinite(x) & np.isfinite(comp)
                b = np.digitize(x[ok], bins) - 1
                keep = (b >= 0) & (b < len(centers))
                np.add.at(total, b[keep], comp[ok][keep])
                np.add.at(count, b[keep], 1.0)
            mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
            ax.plot(centers, mean, lw=1.8, color=colors.get(mode), label=mode)
        ax.axvline(-6.0, color="0.35", ls=":", lw=1.0)
        ax.invert_xaxis()
        ax.set_xlabel(f"true library M_{token}")
        ax.set_ylabel("mean library p(obs)")
        ax.set_title(f"filtersets missing {token}")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png = out_stem.with_name(out_stem.name + "_pobs_profiles.png")
    pdf = out_stem.with_name(out_stem.name + "_pobs_profiles.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[write] {png}", flush=True)


def build_cluster_slug_objects(
    args: argparse.Namespace,
    library: dict,
    specs: list[dict],
    comps: list[np.ndarray],
    lib_den: SampleDen,
) -> tuple[list[dict], np.ndarray]:
    global_keep = np.zeros(len(library["actual_mass"]), dtype=bool)
    for comp in comps:
        global_keep |= np.asarray(comp) >= float(args.comp_threshold)
    if not np.any(global_keep):
        raise RuntimeError("no library rows survive pobs threshold")
    print(f"[logL] global keep {int(np.sum(global_keep))}/{len(global_keep)}", flush=True)

    lib_physprop = np.column_stack(
        [
            np.log10(library["actual_mass"]),
            np.log10(library["eval_time"]),
            library["A_V"],
        ]
    )
    objects: list[dict] = []
    for spec, comp in zip(specs, comps):
        cat = spec["cat"]
        i = spec["filterset_index"]
        keep_i = global_keep & (np.asarray(comp) >= float(args.comp_threshold))
        idx_cat = [library["filter_names"].index(str(f)) for f in cat["filtersets"][i]]
        lib_type = namedtuple(
            "cluster_data",
            ["id", "actual_mass", "time", "form_time", "A_V", "phot_neb_ex", "filter_names", "filter_units"],
        )
        lib = lib_type(
            np.copy(library["cid"][keep_i]),
            np.copy(library["actual_mass"][keep_i]),
            np.copy(library["eval_time"][keep_i]),
            np.copy(library["form_time"][keep_i]),
            np.copy(library["A_V"][keep_i]),
            np.copy(library["phot_neb_ex"][:, idx_cat][keep_i]),
            [str(f) for f in cat["filtersets"][i]],
            list(np.asarray(library["filter_units"])[: len(idx_cat)]),
        )
        cs = cluster_slug(
            lib=lib,
            sample_density=lib_den.sample_den,
            reltol=float(args.tol),
            bw_phot=float(args.bwphot),
            bw_phys=float(args.bwphys),
        )
        cs.add_filters(lib.filter_names, pobs=np.asarray(comp)[keep_i])
        cs.make_cache(range(3), filters=lib.filter_names)
        objects.append(
            {
                "cs": cs,
                "keep": keep_i,
                "phot": cat["phot_filterset"][i],
                "photerr": cat["photerr_filterset"][i],
                "n_obs": int(len(cat["phot_filterset"][i])),
            }
        )
        print(
            f"[logL] built {cat['basename']} fs={lib.filter_names} obs={len(cat['phot_filterset'][i])} lib={int(np.sum(keep_i))}",
            flush=True,
        )
    return objects, lib_physprop


def params_valid(params: np.ndarray) -> bool:
    if params[0] < -3.0 or params[0] > 0.0 or params[1] < 2.0 or params[1] > 8.0:
        return False
    if params[2] < -3.0 or params[2] > 0.0 or params[3] < 5.0 or params[3] > 10.0:
        return False
    return LibWgts(params).valid


def evaluate_logl(objects: list[dict], lib_physprop: np.ndarray, params: np.ndarray) -> float:
    if not params_valid(params):
        return -np.inf
    prior = LibWgts(params).wgts(lib_physprop)
    logl = 0.0
    for obj in objects:
        if obj["n_obs"] < 1:
            continue
        obj["cs"].priors = prior[obj["keep"]]
        logl += float(np.sum(obj["cs"].logL(None, obj["phot"], photerr=obj["photerr"], margindim=range(3))))
    return logl


def run_logl_test(
    args: argparse.Namespace,
    library: dict,
    specs_by_mode: dict[str, list[dict]],
    comps_by_mode: dict[str, list[np.ndarray]],
    out_stem: Path,
) -> list[dict]:
    pdf_dir = Path(args.cluster_slug_lib_dir)
    mass_pdf = args.mass_pdf or str(pdf_dir / "lib_mass.pdf")
    age_pdf = args.age_pdf or str(pdf_dir / "lib_time.pdf")
    av_pdf = args.av_pdf or str(pdf_dir / "lib_av.pdf")
    lib_den = SampleDen(slug_pdf(mass_pdf), slug_pdf(age_pdf), slug_pdf(av_pdf))
    base = parse_params(args.base_params)
    rows: list[dict] = []
    for mode in args.fill_modes:
        objects, lib_physprop = build_cluster_slug_objects(args, library, specs_by_mode[mode], comps_by_mode[mode], lib_den)
        for alpha in args.alphas:
            params = np.array(base, dtype=float)
            params[0] = float(alpha)
            logl = evaluate_logl(objects, lib_physprop, params)
            rows.append(
                {
                    "fill_mode": mode,
                    "alpha_M": float(alpha),
                    "logL": float(logl),
                    "params": ",".join(f"{x:.10g}" for x in params),
                    "max_lib_rows": int(args.max_lib_rows),
                    "library_total_rows": int(library["n_total"]),
                }
            )
            print(f"[logL] mode={mode} alpha_M={alpha:g} logL={logl:.6f}", flush=True)
    if len(args.alphas) >= 2:
        first, second = float(args.alphas[0]), float(args.alphas[1])
        for mode in args.fill_modes:
            vals = {float(r["alpha_M"]): float(r["logL"]) for r in rows if r["fill_mode"] == mode}
            if first in vals and second in vals:
                delta = vals[second] - vals[first]
                rows.append(
                    {
                        "fill_mode": mode,
                        "alpha_M": f"delta_{second:g}_minus_{first:g}",
                        "logL": float(delta),
                        "params": "",
                        "max_lib_rows": int(args.max_lib_rows),
                        "library_total_rows": int(library["n_total"]),
                    }
                )
                print(f"[logL] mode={mode} delta({second:g}-{first:g})={delta:.6f}", flush=True)
    path = out_stem.with_name(out_stem.name + "_fixed_logl.csv")
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    catalogs = read_and_clean_catalogs(args)
    library = read_library(args, all_catalog_filters(catalogs))

    specs_by_mode: dict[str, list[dict]] = {}
    comps_by_mode: dict[str, list[np.ndarray]] = {}
    for mode in args.fill_modes:
        specs, comps = compute_pobs_for_mode(args, catalogs, library, mode)
        specs_by_mode[mode] = specs
        comps_by_mode[mode] = comps

    pobs_rows = summarize_pobs(args, library, specs_by_mode, comps_by_mode, out_stem)
    plot_pobs_profiles(library, specs_by_mode, comps_by_mode, out_stem)
    logl_rows: list[dict] = []
    if not args.skip_logl:
        logl_rows = run_logl_test(args, library, specs_by_mode, comps_by_mode, out_stem)

    meta = {
        "catalogs": args.catalogs,
        "libdir": args.libdir,
        "pobs_mode": args.pobs_mode,
        "comp_threshold": args.comp_threshold,
        "fill_modes": args.fill_modes,
        "max_lib_rows": args.max_lib_rows,
        "seed": args.seed,
        "base_params": parse_params(args.base_params).tolist(),
        "alphas": [float(x) for x in args.alphas],
        "outputs": {
            "pobs_stats_csv": str(out_stem.with_name(out_stem.name + "_pobs_stats.csv")),
            "pobs_profiles_png": str(out_stem.with_name(out_stem.name + "_pobs_profiles.png")),
            "fixed_logl_csv": str(out_stem.with_name(out_stem.name + "_fixed_logl.csv")) if logl_rows else None,
        },
        "n_pobs_rows": len(pobs_rows),
        "n_logl_rows": len(logl_rows),
    }
    meta_path = out_stem.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[write] {meta_path}", flush=True)


if __name__ == "__main__":
    main()
