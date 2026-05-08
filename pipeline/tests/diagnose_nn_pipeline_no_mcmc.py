#!/usr/bin/env python3
"""NN-only LEGUS pipeline diagnostics without fitting MCMC."""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from slugpy import read_cluster

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bundled_pipeline"))

from catalog_readers import reader_register  # noqa: E402
from clean_legus import clean_legus  # noqa: E402
from completeness_io import _nn_scaler_model_paths, legus_nn_missing_uv_u_fills  # noqa: E402
from hybrid_libcomp import (  # noqa: E402
    HybridLegusLibCompletenessCalculator,
    format_hybrid_libcomp_summary,
    legus_catalog_abs_bounds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("libdir", help="cluster_slug library path passed to read_cluster().")
    p.add_argument("--galaxy-names", nargs="+", required=True)
    p.add_argument("--catalog-glob", default="hlsp_legus*{galaxy_name}*.tab")
    p.add_argument("--legus-cct-root", default="/g/data/jh2/jt4478/make_LEGUS_CCT")
    p.add_argument("--legus-tab-dir", default="/g/data/jh2/jt4478/Tang26B/cluster_data")
    p.add_argument("--nn-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--photsystem", default="Vega")
    p.add_argument("--comp-threshold", type=float, default=0.01)
    p.add_argument("--lib-vmag-max", type=float, default=-6.0)
    p.add_argument("--hybrid-range-margin", type=float, default=0.5)
    p.add_argument("--hybrid-min-bands-nonzero", type=int, default=4)
    p.add_argument("--hybrid-nn-batch-rows", type=int, default=65536)
    p.add_argument("--max-lib-rows", type=int, default=250000, help="0 means all rows.")
    p.add_argument("--plot-sample", type=int, default=200000, help="0 means plot all rows.")
    p.add_argument("--skip-library", action="store_true", help="Stop after clean_legus diagnostics.")
    return p.parse_args()


def filt_wave_key(filt: str) -> tuple[int, int, str]:
    match = re.search(r"F(\d+)W", str(filt))
    if match:
        return (0, int(match.group(1)), str(filt))
    return (1, 9999, str(filt))


def discover_catalogs(args: argparse.Namespace) -> list[str]:
    catalogs: list[str] = []
    for galaxy_name in args.galaxy_names:
        pattern = args.catalog_glob.format(galaxy_name=galaxy_name)
        matches = sorted(glob.glob(os.path.join(args.legus_tab_dir, pattern)))
        if not matches:
            matches = sorted(
                glob.glob(os.path.join(args.legus_cct_root, galaxy_name, pattern))
            )
        if not matches:
            raise FileNotFoundError(f"No catalog matched {pattern!r} for {galaxy_name}")
        catalogs.extend(matches)
    return catalogs


def add_filtersets(cat: dict) -> None:
    filtersets: list[list[str]] = []
    filtersets_detect: list[np.ndarray] = []
    fset = np.zeros(len(cat["phot"]), dtype=int)
    filters = np.asarray(cat["filters"])
    for i, detect_row in enumerate(cat["detect"]):
        f = list(filters[detect_row])
        if f not in filtersets:
            filtersets.append(f)
            filtersets_detect.append(np.array(detect_row, dtype=bool))
        fset[i] = filtersets.index(f)

    cat["filtersets"] = filtersets
    cat["filtersets_detect"] = filtersets_detect
    cat["filtersets_index"] = fset
    cat["cid_filterset"] = []
    cat["phot_filterset"] = []
    cat["photerr_filterset"] = []
    cat["detect_filterset"] = []
    for i, d in enumerate(filtersets_detect):
        idx = fset == i
        cat["cid_filterset"].append(cat["cid"][idx])
        cat["phot_filterset"].append(cat["phot"][idx][:, d])
        cat["photerr_filterset"].append(cat["photerr"][idx][:, d])
        cat["detect_filterset"].append(cat["detect"][idx][:, d])


def load_catalogs(paths: Iterable[str]) -> list[dict]:
    catalogs = []
    for path in paths:
        cat = reader_register["LEGUS"].read(path)
        add_filtersets(cat)
        catalogs.append(cat)
    return catalogs


def write_filterset_csv(path: Path, catalogs: list[dict], stage: str) -> None:
    rows = []
    for cat in catalogs:
        all_filters = [str(f) for f in cat["filters"]]
        for i, fs in enumerate(cat["filtersets"]):
            fs = [str(f) for f in fs]
            missing = [f for f in all_filters if f not in fs]
            rows.append(
                {
                    "stage": stage,
                    "catalog": cat["basename"],
                    "galaxy": cat.get("galaxy_fullname", cat["basename"]),
                    "filterset_index": i,
                    "n_filters": len(fs),
                    "n_clusters": len(cat["phot_filterset"][i]),
                    "filters": " ".join(fs),
                    "missing_filters": " ".join(missing),
                }
            )
    exists = path.exists()
    with path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def v_index(filter_names: list[str]) -> int:
    for token in ("F555W", "F606W"):
        for i, f in enumerate(filter_names):
            if token in str(f):
                return i
    raise ValueError(f"No V-like filter in {filter_names}")


def plot_vmag_comp(path: Path, v_app: np.ndarray, comp: np.ndarray, title: str, sample: int) -> None:
    finite = np.isfinite(v_app) & np.isfinite(comp)
    x = v_app[finite]
    y = comp[finite]
    if sample > 0 and len(x) > sample:
        rng = np.random.default_rng(8675309)
        idx = rng.choice(len(x), size=sample, replace=False)
        x = x[idx]
        y = y[idx]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    if len(x) > 1000:
        hb = ax.hexbin(x, y, gridsize=80, mincnt=1, cmap="viridis", bins="log")
        fig.colorbar(hb, ax=ax, label="log10(N)")
    else:
        ax.scatter(x, y, s=6, alpha=0.45, linewidths=0)
    ax.axvline(24.0, color="tab:red", lw=1.2, ls=":", label="m_V = 24")
    ax.set_xlabel("Library apparent V magnitude")
    ax.set_ylabel("NN hybrid library completeness")
    ax.set_xlim(15.0, 28.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def observed_nn_completeness(cat: dict, nn_dir: str) -> np.ndarray:
    from completeness_io import predict_catalog_completeness_with_nn

    comp_all = np.full(len(cat["phot"]), np.nan, dtype=float)
    dmod = float(cat["dmod"])
    full_order = [str(f) for f in cat["filters"]]
    galaxy = str(cat.get("galaxy_fullname", cat["basename"]))
    cid_to_row = {int(cid): i for i, cid in enumerate(cat["cid"])}

    for i, subset in enumerate(cat["filtersets"]):
        subset = [str(f) for f in subset]
        phot_app = np.asarray(cat["phot_filterset"][i], dtype=float) + dmod
        fills = legus_nn_missing_uv_u_fills(
            cat["filters"],
            np.asarray(cat["phot"], dtype=float),
            np.asarray(cat["detect"], dtype=bool),
            dmod,
            full_order,
            subset,
            use_apparent_magnitude=True,
        )
        comp = predict_catalog_completeness_with_nn(
            phot_app,
            galaxy_fullname=galaxy,
            nn_dir=nn_dir,
            subset_filters=subset,
            full_filter_order=full_order,
            missing_band_fills=fills if fills else None,
        )
        for cid, val in zip(cat["cid_filterset"][i], comp):
            row = cid_to_row.get(int(cid))
            if row is not None:
                comp_all[row] = float(val)
    return comp_all


def plot_observed_catalog_nn(outdir: Path, catalogs: list[dict], nn_dir: str, stage: str) -> None:
    csv_path = outdir / f"observed_nn_completeness_{stage}.csv"
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for cat in catalogs:
        filters = [str(f) for f in cat["filters"]]
        vi = v_index(filters)
        v_abs = np.asarray(cat["phot"], dtype=float)[:, vi]
        v_app = v_abs + float(cat["dmod"])
        comp = observed_nn_completeness(cat, nn_dir)
        ok = np.isfinite(v_app) & np.isfinite(comp)
        label = str(cat.get("galaxy_fullname", cat["basename"]))
        ax.scatter(v_app[ok], comp[ok], s=12, alpha=0.55, linewidths=0, label=label)
        for cid, vv_abs, vv_app, cc, fidx in zip(
            cat["cid"], v_abs, v_app, comp, cat["filtersets_index"]
        ):
            rows.append(
                {
                    "stage": stage,
                    "catalog": cat["basename"],
                    "galaxy": label,
                    "cid": int(cid),
                    "filterset_index": int(fidx),
                    "v_abs_mag": float(vv_abs),
                    "v_app_mag": float(vv_app),
                    "nn_completeness": float(cc),
                }
            )
    ax.axvline(24.0, color="tab:red", lw=1.2, ls=":", label="m_V = 24")
    ax.set_xlabel("Observed cluster apparent V magnitude")
    ax.set_ylabel("NN predicted completeness")
    ax.set_xlim(15.0, 28.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Observed catalog NN completeness ({stage})")
    ax.legend(loc="best")
    png_path = outdir / f"observed_nn_completeness_vs_vmag_{stage}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[observed-plot] {png_path}")
    print(f"[observed-csv] {csv_path}")


def main() -> int:
    args = parse_args()
    os.environ["LEGUS_CCT_ROOT"] = args.legus_cct_root
    os.environ["LEGUS_TAB_DIR"] = args.legus_tab_dir
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    catalog_paths = discover_catalogs(args)
    print("Catalogs:")
    for p in catalog_paths:
        print(f"  {p}")

    catalogs = load_catalogs(catalog_paths)
    filter_csv = outdir / "clean_legus_filtersets.csv"
    if filter_csv.exists():
        filter_csv.unlink()
    write_filterset_csv(filter_csv, catalogs, "before_clean")
    plot_observed_catalog_nn(outdir, catalogs, args.nn_dir, "before_clean")

    clean_catalogs = copy.deepcopy(catalogs)
    before = sum(len(c["phot"]) for c in clean_catalogs)
    after = clean_legus(
        clean_catalogs,
        verbose=True,
        nn_dir=args.nn_dir,
        comp_threshold=args.comp_threshold,
        enforce_hybrid_criteria=True,
        lib_vmag_max=args.lib_vmag_max,
        min_bands_nonzero=args.hybrid_min_bands_nonzero,
    )
    write_filterset_csv(filter_csv, clean_catalogs, "after_clean")
    plot_observed_catalog_nn(outdir, clean_catalogs, args.nn_dir, "after_clean")
    print(f"[clean_legus_nn] total before={before} after={after} removed={before - after}")
    print(f"[clean_legus_nn] filterset CSV: {filter_csv}")
    if args.skip_library:
        print("[skip] --skip-library set; not reading cluster_slug library or plotting libcomp.")
        return 0

    allfilters: list[str] = []
    for cat in clean_catalogs:
        for f in cat["filters"]:
            if f not in allfilters:
                allfilters.append(f)
    allfilters = sorted([str(f) for f in allfilters], key=filt_wave_key)

    print(f"Loading library with filters={allfilters}")
    lib = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=allfilters)
    phot_neb_ex = np.asarray(lib.phot_neb_ex, dtype=float)
    lib_filter_names = [str(f) for f in lib.filter_names]
    if args.max_lib_rows > 0 and len(phot_neb_ex) > args.max_lib_rows:
        phot_neb_ex = phot_neb_ex[: args.max_lib_rows]
        print(f"[library] using first {len(phot_neb_ex)} rows (--max-lib-rows)")
    v_abs = phot_neb_ex[:, v_index(lib_filter_names)]

    summary_path = outdir / "nn_libcomp_summary.txt"
    with summary_path.open("w") as sfp:
        sfp.write(f"NN-only pipeline diagnostic; no MCMC fit\n")
        sfp.write(f"library_rows={len(phot_neb_ex)} comp_threshold={args.comp_threshold}\n\n")

        for cat in clean_catalogs:
            galaxy = str(cat.get("galaxy_fullname", cat["basename"]))
            scaler, model = _nn_scaler_model_paths(args.nn_dir, galaxy, None, None)
            sfp.write(f"Catalog: {cat['basename']} galaxy={galaxy}\n")
            sfp.write(f"  scaler={scaler}\n")
            sfp.write(f"  model={model}\n")
            print(f"[nn] {galaxy}: scaler={scaler}")
            print(f"[nn] {galaxy}: model={model}")

            for i, d in enumerate(cat["filtersets_detect"]):
                subset_filters = [str(f) for f in np.asarray(cat["filters"])[d]]
                subset_filters = sorted(subset_filters, key=filt_wave_key)
                min_bands = min(args.hybrid_min_bands_nonzero, len(subset_filters))
                phot_sub = np.asarray(cat["phot"], dtype=float)[:, d]
                detect_sub = np.asarray(cat["detect"], dtype=bool)[:, d]
                bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                    phot_sub,
                    detect_sub,
                    subset_filters,
                    range_margin=args.hybrid_range_margin,
                )
                nn_fills = legus_nn_missing_uv_u_fills(
                    cat["filters"],
                    np.asarray(cat["phot"], dtype=float),
                    np.asarray(cat["detect"], dtype=bool),
                    float(cat["dmod"]),
                    [str(f) for f in cat["filters"]],
                    subset_filters,
                    use_apparent_magnitude=True,
                )
                calc = HybridLegusLibCompletenessCalculator(
                    subset_filters,
                    bounds_lo,
                    bounds_hi,
                    lib_filter_names,
                    scaler,
                    model,
                    lib_vmag_max=args.lib_vmag_max,
                    min_bands_nonzero=min_bands,
                    nn_batch_rows=args.hybrid_nn_batch_rows,
                    nn_full_filter_order=[str(f) for f in cat["filters"]],
                    nn_missing_band_fills=nn_fills,
                )
                result = calc.compute(phot_neb_ex, float(cat["dmod"]), galaxy_fullname=galaxy)
                comp = result["comp_hybrid"]
                missing = [f for f in cat["filters"] if str(f) not in subset_filters]
                label = f"{cat['basename']} filterset={i} nband={len(subset_filters)} missing={missing}"
                sfp.write(label + "\n")
                sfp.write(format_hybrid_libcomp_summary(result["summary"]) + "\n\n")
                png = outdir / f"{galaxy}_filterset{i}_{len(subset_filters)}band_vapp_vs_nn_libcomp.png"
                plot_vmag_comp(png, v_abs + float(cat["dmod"]), comp, label, args.plot_sample)
                print(f"[plot] {png}")

    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
