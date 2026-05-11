#!/usr/bin/env python3
"""
Compare one-shot logL under different analysis ablations.

This script rebuilds the core data path used by analyze_catalog_mid_mdd.py,
then evaluates a single log-likelihood at a fixed parameter vector while
toggling:
  1) observed-catalog cleaning route
  2) library completeness source

Typical usage (NGC628 c/e):
  python tests/compare_logl_ablation.py \
    --catalog /g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-c/hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab \
    --catalog /g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-e/hlsp_legus_hst_acs-wfc3_ngc628-e_multiband_v1_padagb-mwext-avgapcor.tab \
    --libdir /g/data/jh2/jt4478/cluster_slug/tang \
    --mass-pdf /g/data/jh2/jt4478/cluster_slug/lib_mass.pdf \
    --age-pdf /g/data/jh2/jt4478/cluster_slug/lib_time.pdf \
    --av-pdf /g/data/jh2/jt4478/cluster_slug/lib_av.pdf \
    --nn-dir /g/data/jh2/jt4478/Tang26B/nn_models \
    --tabulated-comp-dir /scratch/jh2/jt4478/tabulated_comp \
    --clean-mode nn --libcomp-mode nn
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import os.path as osp
import re
import sys
from collections import namedtuple

import numexpr as ne
import numpy as np
from slugpy import read_cluster, slug_pdf
from slugpy.cluster_slug import cluster_slug

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(ROOT, "bundled_pipeline"))

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


class SampleDen:
    def __init__(self, mpdf, tpdf, avpdf):
        self.mpdf = mpdf
        self.tpdf = tpdf
        self.avpdf = avpdf

    def sample_den(self, physprop):
        m = 10.0 ** physprop[:, 0]
        m[m < self.mpdf.bkpts[0]] = self.mpdf.bkpts[0]
        m[m > self.mpdf.bkpts[-1]] = self.mpdf.bkpts[-1]
        t = 10.0 ** physprop[:, 1]
        av = physprop[:, 2]
        return self.mpdf(m) * self.tpdf(t) * self.avpdf(av) * m * t


class LibWgts:
    def __init__(self, p: np.ndarray, nav: int, mid: bool):
        self.alphaM = float(p[0])
        self.mBreak = 10.0 ** float(p[1])
        self.mid = bool(mid)
        if self.mid:
            self.alphaT = float(p[2])
            self.tMid = 10.0 ** float(p[3])
        else:
            self.gammaMdd = float(p[2])
            self.tMddMin = 10.0 ** float(p[3])
        self.nav = int(nav)
        self.delta_av = 3.0 / self.nav
        self.av = np.arange(0, 3.0 + self.delta_av / 2.0, self.delta_av)
        self.pav = np.zeros(self.nav + 1)
        self.pav[:-1] = 10.0 ** p[4:]
        self.pav[-1] = 2.0 / self.delta_av - self.pav[-2] - np.sum(self.pav[:-2] + self.pav[1:-1])

    def wgts(self, physprop):
        logm = physprop[:, 0]
        logt = physprop[:, 1]
        av = physprop[:, 2]

        if self.mid:
            logt_mid = np.log10(self.tMid)
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*exp(-10.**logm/mBreak) * "
                "where(logt <= logt_mid, 10.**logt/tMid, (10.**logt/tMid)**(alphaT+1))",
                local_dict={
                    "alphaM": self.alphaM,
                    "logm": logm,
                    "mBreak": self.mBreak,
                    "logt": logt,
                    "logt_mid": logt_mid,
                    "tMid": self.tMid,
                    "alphaT": self.alphaT,
                },
            )
        else:
            eta = ne.evaluate(
                "(1.0 + gammaMdd*(100.0/10.**logm)**gammaMdd * 10.**logt/tMddMin)**(1.0/gammaMdd)",
                local_dict={
                    "gammaMdd": self.gammaMdd,
                    "logm": logm,
                    "logt": logt,
                    "tMddMin": self.tMddMin,
                },
            )
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*eta**(alphaM+1.0-gammaMdd)*"
                "exp(-10.**logm*eta/mBreak)*10.**logt",
                local_dict={
                    "alphaM": self.alphaM,
                    "logm": logm,
                    "eta": eta,
                    "gammaMdd": self.gammaMdd,
                    "mBreak": self.mBreak,
                    "logt": logt,
                },
            )

        for i in range(self.nav):
            avlo = self.av[i]
            avhi = self.av[i + 1]
            pavlo = self.pav[i]
            pavhi = self.pav[i + 1]
            avslope = (pavhi - pavlo) / (avhi - avlo)
            wgt = ne.evaluate(
                "wgt * where((av >= avlo) & (av < avhi), pavlo + (av-avlo)*avslope, 1.0)",
                local_dict={
                    "wgt": wgt,
                    "av": av,
                    "avlo": avlo,
                    "avhi": avhi,
                    "pavlo": pavlo,
                    "avslope": avslope,
                },
            )
        return wgt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation compare for one-shot logL.")
    p.add_argument("--catalog", action="append", required=True, help="HLSP catalog path; repeat for multiple.")
    p.add_argument("--libdir", required=True)
    p.add_argument("--mass-pdf", required=True)
    p.add_argument("--age-pdf", required=True)
    p.add_argument("--av-pdf", required=True)
    p.add_argument("--nn-dir", required=True)
    p.add_argument("--nn-scaler", default=None)
    p.add_argument("--nn-model", default=None)
    p.add_argument("--tabulated-comp-dir", default="/scratch/jh2/jt4478/tabulated_comp")
    p.add_argument("--clean-mode", choices=["nn", "none"], default="nn")
    p.add_argument("--libcomp-mode", choices=["nn", "tabulated"], default="nn")
    p.add_argument("--pobs-mode", choices=["observed-box", "nn", "hybrid"], default="hybrid")
    p.add_argument("--hybrid-range-margin", type=float, default=0.1)
    p.add_argument("--hybrid-min-bands-nonzero", type=int, default=4)
    p.add_argument("--hybrid-nn-batch-rows", type=int, default=65536)
    p.add_argument("--params", nargs="+", type=float, default=None)
    p.add_argument("--comp-threshold", type=float, default=0.01)
    p.add_argument("--lib-vmag-max", type=float, default=-6.0)
    p.add_argument("--photsystem", default="Vega")
    p.add_argument("--max-lib-rows", type=int, default=0, help="Optional cap for quick runs (0 = all rows).")
    p.add_argument("--tol", type=float, default=1.0e-2)
    p.add_argument("--bwphot", type=float, default=0.05)
    p.add_argument("--bwphys", type=float, default=0.05)
    p.add_argument("--nav", type=int, default=6)
    p.add_argument("--mdd", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_filtersets(data: dict) -> None:
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(data["phot"]))
    for i, d in enumerate(data["detect"]):
        f = list(np.array(data["filters"])[d])
        if f not in filtersets:
            filtersets.append(f)
            filtersets_detect.append(np.copy(d))
        fset[i] = filtersets.index(f)
    data["filtersets"] = filtersets
    data["filtersets_index"] = fset
    data["filtersets_detect"] = filtersets_detect

    cid_filterset = []
    phot_filterset = []
    photerr_filterset = []
    for i, d in enumerate(data["filtersets_detect"]):
        idx = data["filtersets_index"] == i
        cid_filterset.append(data["cid"][idx])
        phot_filterset.append(data["phot"][idx][:, d])
        photerr_filterset.append(data["photerr"][idx][:, d])
    data["cid_filterset"] = cid_filterset
    data["phot_filterset"] = phot_filterset
    data["photerr_filterset"] = photerr_filterset


def infer_galaxy_key(cat: dict) -> str:
    b = str(cat.get("basename", "")).lower()
    g = str(cat.get("galaxy_fullname", "")).lower()
    token = f"{b}_{g}"
    if "628-c" in token or "628c" in token:
        return "628c"
    if "628-e" in token or "628e" in token:
        return "628e"
    raise ValueError(f"Cannot infer galaxy key for tabulated completeness from {cat.get('basename')}")


def load_tabulated_libcomp(tab_dir: str, gal_key: str, n_detected: int) -> np.ndarray:
    if n_detected == 5:
        pats = [f"lib*{gal_key}*full*_comp.npy", f"lib*{gal_key}*fullcomp.npy", f"lib*{gal_key}*full_comp.npy"]
    elif n_detected == 4:
        pats = [f"lib*{gal_key}*noUV*_comp.npy", f"lib*{gal_key}*noUVcomp.npy", f"lib*{gal_key}*no_UV*_comp.npy"]
    else:
        raise ValueError(f"Unsupported detected-band count for tabulated mode: {n_detected}")

    matches = []
    for pat in pats:
        matches.extend(glob.glob(osp.join(tab_dir, pat)))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"No tabulated comp files for gal={gal_key}, n_detected={n_detected} in {tab_dir}")
    return np.load(matches[0])


def evaluate_logl(catalogs: list[dict], params: np.ndarray, nav: int, mdd: bool) -> float:
    wgts = LibWgts(params, nav=nav, mid=not mdd)
    for cat in catalogs:
        for cs in cat["cs"]:
            cs.priors = wgts.wgts

    logl = 0.0
    for cat in catalogs:
        for cs, phot, photerr in zip(cat["cs"], cat["phot_filterset"], cat["photerr_filterset"]):
            if len(phot) < 1:
                continue
            logl += np.sum(cs.logL(None, phot, photerr=photerr, margindim=range(3)))
    return float(logl)


def main() -> int:
    args = parse_args()

    catalogs = []
    allfilters = []
    for cpath in args.catalog:
        d = reader_register["LEGUS"].read(cpath)
        make_filtersets(d)
        catalogs.append(d)
        for f in d["filters"]:
            if f not in allfilters:
                allfilters.append(f)

    before = sum(len(c["phot"]) for c in catalogs)
    if args.clean_mode == "nn":
        clean_legus(
            catalogs,
            verbose=args.verbose,
            nn_dir=args.nn_dir,
            nn_scaler_path=args.nn_scaler,
            nn_model_path=args.nn_model,
            comp_threshold=args.comp_threshold,
            enforce_hybrid_criteria=(args.pobs_mode == "hybrid"),
            lib_vmag_max=float(args.lib_vmag_max),
            min_bands_nonzero=int(args.hybrid_min_bands_nonzero),
        )
    after_clean = sum(len(c["phot"]) for c in catalogs)

    mpdf = slug_pdf(args.mass_pdf)
    tpdf = slug_pdf(args.age_pdf)
    avpdf = slug_pdf(args.av_pdf)
    lib_den = SampleDen(mpdf, tpdf, avpdf)

    lib_all = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=allfilters)
    cid = lib_all.id
    actual_mass = lib_all.actual_mass
    form_time = lib_all.form_time
    eval_time = lib_all.time
    a_v = lib_all.A_V
    phot_neb_ex = lib_all.phot_neb_ex
    filter_names = lib_all.filter_names
    filter_units = lib_all.filter_units
    lib_filter_names = [str(f) for f in filter_names]

    ncl_init = len(actual_mass)
    if args.max_lib_rows > 0 and ncl_init > args.max_lib_rows:
        sl = slice(0, args.max_lib_rows)
        cid = cid[sl]
        actual_mass = actual_mass[sl]
        form_time = form_time[sl]
        eval_time = eval_time[sl]
        a_v = a_v[sl]
        phot_neb_ex = phot_neb_ex[sl]
        ncl_init = len(actual_mass)
    keep = np.zeros(ncl_init, dtype=bool)

    v_idx = None
    for cand in ("ACS_F555W", "WFC3_UVIS_F555W"):
        if cand in lib_filter_names:
            v_idx = lib_filter_names.index(cand)
            break
    if v_idx is None:
        for i, f in enumerate(lib_filter_names):
            if str(f).endswith("F555W"):
                v_idx = i
                break
    faint_v_mask = np.zeros(ncl_init, dtype=bool)
    if v_idx is not None:
        v_abs = phot_neb_ex[:, v_idx]
        faint_v_mask = np.isfinite(v_abs) & (v_abs > args.lib_vmag_max)

    for cat in catalogs:
        cat["libcomp"] = []
        for d in cat["filtersets_detect"]:
            requested_filters = [str(f) for f in np.array(cat["filters"])[d]]

            def _filt_wave_key(filt):
                match = re.search(r"F(\d+)W", str(filt))
                if match is not None:
                    return (0, int(match.group(1)), str(filt))
                return (1, lib_filter_names.index(str(filt)), str(filt))

            subset_filters = sorted(requested_filters, key=_filt_wave_key)

            if args.libcomp_mode == "tabulated":
                gk = infer_galaxy_key(cat)
                comp = load_tabulated_libcomp(args.tabulated_comp_dir, gk, int(np.sum(d)))
                if comp.shape[0] != ncl_init:
                    raise ValueError(
                        f"Tabulated comp length mismatch for {cat['basename']}: "
                        f"{comp.shape[0]} != lib {ncl_init}"
                    )
            else:
                dmod = float(cat["dmod"])
                lib_indices = [lib_filter_names.index(f) for f in subset_filters]
                all_cat_filters = [str(f) for f in cat["filters"]]
                full_filter_order = (
                    sorted(all_cat_filters, key=_filt_wave_key)
                    if set(subset_filters) < set(all_cat_filters)
                    else subset_filters
                )
                fills = legus_nn_missing_uv_u_fills(
                    cat["filters"],
                    np.asarray(cat["phot"], dtype=float),
                    np.asarray(cat["detect"], dtype=bool),
                    dmod,
                    full_filter_order,
                    subset_filters,
                    use_apparent_magnitude=True,
                )
                nn_scaler_path, nn_model_path = _nn_scaler_model_paths(
                    args.nn_dir,
                    str(cat.get("galaxy_fullname", cat["basename"])),
                    args.nn_scaler,
                    args.nn_model,
                )
                if args.pobs_mode == "hybrid":
                    print(
                        f"[libcomp] {cat['basename']} pobs_mode=hybrid filterset={subset_filters!r}",
                        flush=True,
                    )
                    bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                        np.asarray(cat["phot"], dtype=float)[:, d],
                        np.asarray(cat["detect"], dtype=bool)[:, d],
                        subset_filters,
                        range_margin=float(args.hybrid_range_margin),
                    )
                    calc = HybridLegusLibCompletenessCalculator(
                        subset_filters,
                        bounds_lo,
                        bounds_hi,
                        lib_filter_names,
                        nn_scaler_path,
                        nn_model_path,
                        lib_vmag_max=float(args.lib_vmag_max),
                        min_bands_nonzero=min(int(args.hybrid_min_bands_nonzero), len(subset_filters)),
                        nn_full_filter_order=[str(f) for f in cat["filters"]],
                        nn_batch_rows=int(args.hybrid_nn_batch_rows),
                        nn_missing_band_fills=fills,
                    )
                    comp = calc.compute(
                        phot_neb_ex,
                        dmod=dmod,
                        galaxy_fullname=str(cat.get("galaxy_fullname", cat["basename"])),
                    )["comp_hybrid"]
                else:
                    lib_phot_subset = np.asarray(phot_neb_ex[:, lib_indices], dtype=float) + dmod
                    finite_rows = np.all(np.isfinite(lib_phot_subset), axis=1)
                    comp_no_criteria = np.zeros(len(finite_rows), dtype=float)
                    idx = np.flatnonzero(finite_rows)
                    bs = max(1024, int(args.hybrid_nn_batch_rows))
                    print(
                        f"[libcomp] {cat['basename']} pobs_mode={args.pobs_mode} "
                        f"filterset={subset_filters!r} finite_rows={idx.size}",
                        flush=True,
                    )
                    for start in range(0, idx.size, bs):
                        rows = idx[start:start + bs]
                        print(
                            f"[libcomp] NN batch {start // bs + 1}/{(idx.size + bs - 1) // bs}",
                            flush=True,
                        )
                        comp_no_criteria[rows] = predict_catalog_completeness_with_nn(
                            lib_phot_subset[rows],
                            galaxy_fullname=cat.get("galaxy_fullname", cat["basename"]),
                            nn_dir=None,
                            subset_filters=subset_filters,
                            full_filter_order=full_filter_order,
                            nn_scaler_path=nn_scaler_path,
                            nn_model_path=nn_model_path,
                            missing_band_fills=fills if fills else None,
                        )
                    if args.pobs_mode == "observed-box":
                        bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                            np.asarray(cat["phot"], dtype=float)[:, d],
                            np.asarray(cat["detect"], dtype=bool)[:, d],
                            subset_filters,
                            range_margin=float(args.hybrid_range_margin),
                        )
                        inside_box = inside_5d_abs_box(
                            phot_neb_ex,
                            lib_column_indices(lib_filter_names, subset_filters),
                            bounds_lo,
                            bounds_hi,
                        )
                        comp = comp_no_criteria * inside_box.astype(float)
                    elif args.pobs_mode == "nn":
                        comp = comp_no_criteria
                    else:
                        raise ValueError(f"Unknown pobs_mode: {args.pobs_mode}")

            cat["libcomp"].append(np.asarray(comp, dtype=float))
            keep = np.logical_or(keep, cat["libcomp"][-1] >= args.comp_threshold)

    # prune library
    cid = cid[keep]
    actual_mass = actual_mass[keep]
    eval_time = eval_time[keep]
    form_time = form_time[keep]
    a_v = a_v[keep]
    phot_neb_ex = phot_neb_ex[keep]
    for cat in catalogs:
        for i in range(len(cat["libcomp"])):
            cat["libcomp"][i] = cat["libcomp"][i][keep]
    after_lib = len(actual_mass)

    # build cluster_slug per filterset
    for cat in catalogs:
        cat["cs"] = []
        for i in range(len(cat["filtersets"])):
            idx_cat = [filter_names.index(idxx) for idxx in cat["filtersets"][i]]
            idx = cat["filtersets_detect"][i]
            keep_fs = cat["libcomp"][i] >= args.comp_threshold
            fields = [
                np.copy(cid[keep_fs]),
                np.copy(actual_mass[keep_fs]),
                np.copy(eval_time[keep_fs]),
                np.copy(form_time[keep_fs]),
                np.copy(a_v[keep_fs]),
                np.copy(phot_neb_ex[:, idx_cat][keep_fs]),
                list(np.array(cat["filters"])[idx]),
                list(np.array(filter_units[: len(idx_cat)])),
            ]
            lib_type = namedtuple(
                "cluster_data",
                ["id", "actual_mass", "time", "form_time", "A_V", "phot_neb_ex", "filter_names", "filter_units"],
            )
            lib = lib_type(*fields)
            cs = cluster_slug(
                lib=lib,
                sample_density=lib_den.sample_den,
                reltol=args.tol,
                bw_phot=args.bwphot,
                bw_phys=args.bwphys,
            )
            cs.add_filters(lib.filter_names, pobs=cat["libcomp"][i][keep_fs])
            cs.make_cache(range(3), filters=lib.filter_names)
            cat["cs"].append(cs)

    # fixed comparison parameter vector
    if args.params is not None:
        p = np.asarray(args.params, dtype=float)
        expected = 4 + args.nav
        if p.size != expected:
            raise ValueError(f"--params expected {expected} values, got {p.size}")
    else:
        p = np.zeros(4 + args.nav, dtype=float)
        p[0] = -2.0
        p[1] = 6.0
        p[2] = 0.5 if args.mdd else -1.0
        p[3] = 5.0 if args.mdd else 7.0
        p[4:] = np.log10(np.full(args.nav, 1.0 / 3.0))

    logl = evaluate_logl(catalogs, p, nav=args.nav, mdd=args.mdd)

    print("=== Ablation summary ===")
    print(f"clean_mode={args.clean_mode}")
    print(f"libcomp_mode={args.libcomp_mode}")
    print(f"pobs_mode={args.pobs_mode}")
    print(f"comp_threshold={args.comp_threshold:.4f}")
    print("params=" + ",".join(f"{x:.10g}" for x in p))
    print(f"catalog clusters: before_clean={before}, after_clean={after_clean}")
    print(f"library rows: before={ncl_init}, after_prune={after_lib}")
    for cat in catalogs:
        print(f"{cat['basename']}: n_obs={len(cat['phot'])}, n_filtersets={len(cat['filtersets'])}")
    print(f"logL_at_fixed_params={logl:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
