#!/usr/bin/env python3
"""Compare NGC628 observed luminosity functions from old and Tang26B readers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/g/data/jh2/jt4478/Tang26B")
OLD_ROOT = Path("/g/data/jh2/jt4478/legus_slug23")
OLD_COMP = OLD_ROOT / "completeness_data"
OUTDIR = ROOT / "output_io"

OLD_CATALOGS = [
    OLD_ROOT / "cluster_catalogs/hlsp_628c.tab",
    OLD_ROOT / "cluster_catalogs/hlsp_628e.tab",
]

TANG_CATALOGS = [
    Path(
        "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-c/"
        "hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab"
    ),
    Path(
        "/g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-e/"
        "hlsp_legus_hst_acs-wfc3_ngc628-e_multiband_v1_padagb-mwext-avgapcor.tab"
    ),
]

BANDS = ["F275W", "F336W", "F435W", "F555W", "F814W"]
BINS = np.arange(-13.0, -4.0 + 0.25, 0.25)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_filtersets(cat: dict) -> None:
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(cat["phot"]), dtype=int)
    filters = np.asarray(cat["filters"])
    for i, detected in enumerate(cat["detect"]):
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

        # Tang26B clean_legus updates these fields if present. Ensure each
        # optional per-filterset field exists when the per-catalog field exists.
        for base in ("mass", "age", "mass_min", "mass_max", "age_min", "age_max"):
            if base in cat:
                cat.setdefault(f"{base}_filterset", []).append(cat[base][idx])


def remove_cids(cat: dict, cids_to_delete: np.ndarray) -> None:
    if len(cids_to_delete) == 0:
        return
    keep = ~np.isin(cat["cid"], cids_to_delete)
    for key in (
        "cid",
        "phot",
        "photerr",
        "detect",
        "filtersets_index",
        "ra",
        "dec",
        "class",
        "mass",
        "age",
        "mass_min",
        "mass_max",
        "age_min",
        "age_max",
    ):
        if key in cat:
            cat[key] = cat[key][keep]

    for cid in cids_to_delete:
        for i in range(len(cat["filtersets"])):
            idx_keep = cat["cid_filterset"][i] != cid
            for key in (
                "cid_filterset",
                "phot_filterset",
                "photerr_filterset",
                "detect_filterset",
                "mass_filterset",
                "age_filterset",
                "mass_min_filterset",
                "mass_max_filterset",
                "age_min_filterset",
                "age_max_filterset",
            ):
                if key in cat:
                    cat[key][i] = cat[key][i][idx_keep]


def clean_old_legacy(catalogs: list[dict]) -> dict:
    """Replicate old clean_legus for NGC628 using local completeness arrays."""
    removed_comp = {}
    for cat in catalogs:
        n_removed = 0
        for i, phot in enumerate(cat["phot_filterset"]):
            if len(phot) == 0:
                continue
            five_band = phot.shape[1] == 5
            if "628c" in cat["basename"]:
                comp_name = "LEGUSngc628c_comp.npy" if five_band else "LEGUSngc628cnoUV_comp.npy"
            elif "628e" in cat["basename"]:
                comp_name = "LEGUSngc628e_comp.npy" if five_band else "LEGUSngc628enoUV_comp.npy"
            else:
                raise ValueError(f"Unknown old catalog basename {cat['basename']}")
            comp = np.load(OLD_COMP / comp_name)
            if len(comp) != len(phot):
                raise ValueError(
                    f"{comp_name} length {len(comp)} does not match "
                    f"{cat['basename']} filterset {i} length {len(phot)}"
                )
            cids_to_delete = cat["cid_filterset"][i][comp == 0.0]
            n_removed += len(cids_to_delete)
            remove_cids(cat, cids_to_delete)
        removed_comp[cat["basename"]] = int(n_removed)

    removed_dup = {}
    for i, cat in enumerate(catalogs):
        dup = np.zeros(len(cat["ra"]), dtype=bool)
        for other in catalogs[i + 1 :]:
            ra_match = np.equal.outer(cat["ra"], other["ra"])
            dec_match = np.equal.outer(cat["dec"], other["dec"])
            dup |= np.logical_or.reduce(ra_match & dec_match, axis=1)
        cids_to_delete = cat["cid"][dup]
        removed_dup[cat["basename"]] = int(len(cids_to_delete))
        remove_cids(cat, cids_to_delete)
    return {"removed_completeness_zero": removed_comp, "removed_duplicates": removed_dup}


def read_old_catalogs() -> tuple[list[dict], dict]:
    old_readers = load_module("old_catalog_readers_for_lf", OLD_ROOT / "catalog_readers.py")
    catalogs = []
    for path in OLD_CATALOGS:
        cat = old_readers.reader_register["LEGUS"].read(str(path))
        prepare_filtersets(cat)
        catalogs.append(cat)
    clean_info = clean_old_legacy(catalogs)
    return catalogs, clean_info


def read_old_catalogs_nn_legacy_rule() -> tuple[list[dict], dict]:
    old_readers = load_module("old_catalog_readers_for_nn_legacy_lf", OLD_ROOT / "catalog_readers.py")
    catalogs = []
    for path in OLD_CATALOGS:
        cat = old_readers.reader_register["LEGUS"].read(str(path))
        galaxy = "ngc628-c" if "628c" in cat["basename"] else "ngc628-e"
        cat["galaxy_fullname"] = galaxy
        prepare_filtersets(cat)
        catalogs.append(cat)
    before = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
    clean = clean_nn_legacy_rule(catalogs, nn_dir=str(ROOT / "nn_models"))
    clean["before_clean"] = before
    clean["after_clean"] = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
    return catalogs, clean


def read_tang_catalogs() -> tuple[list[dict], dict]:
    bundled = ROOT / "pipeline/bundled_pipeline"
    sys.path.insert(0, str(bundled))
    try:
        os.environ["LEGUS_CCT_ROOT"] = "/g/data/jh2/jt4478/make_LEGUS_CCT"
        os.environ["LEGUS_TAB_DIR"] = str(ROOT / "cluster_data")
        readers = load_module("tang_catalog_readers_for_lf", bundled / "catalog_readers.py")
        clean_mod = load_module("tang_clean_legus_for_lf", bundled / "clean_legus.py")
        catalogs = []
        for path in TANG_CATALOGS:
            cat = readers.reader_register["LEGUS"].read(str(path))
            prepare_filtersets(cat)
            catalogs.append(cat)
        before = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
        clean_mod.clean_legus(
            catalogs,
            verbose=False,
            nn_dir=str(ROOT / "nn_models"),
            comp_threshold=0.01,
            enforce_hybrid_criteria=True,
            lib_vmag_max=-6.0,
            min_bands_nonzero=4,
        )
        after = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
        return catalogs, {"before_clean": before, "after_clean": after}
    finally:
        try:
            sys.path.remove(str(bundled))
        except ValueError:
            pass


def clean_nn_legacy_rule(catalogs: list[dict], nn_dir: str) -> dict:
    """Old clean_legus rule, but observed completeness is predicted by NN.

    This intentionally mirrors the old cleaner's behavior:
    - clean each existing filterset independently
    - remove only rows with comp <= 0
    - remove duplicated RA/DEC across catalogs

    It does not apply Tang26B's hybrid V/B-I/Nband/Vmag criteria and does not
    apply a positive comp_threshold such as 0.01.
    """
    bundled = ROOT / "pipeline/bundled_pipeline"
    sys.path.insert(0, str(bundled))
    try:
        comp_io = load_module("tang_completeness_io_for_legacy_lf", bundled / "completeness_io.py")
        removed_comp = {}
        comp_summary = {}
        for cat in catalogs:
            n_removed = 0
            cat_summary = []
            for i in range(len(cat["phot_filterset"])):
                phot_abs = np.asarray(cat["phot_filterset"][i], dtype=float)
                phot_app = phot_abs + float(cat["dmod"])
                subset_filters = [str(f) for f in cat["filtersets"][i]]
                full_filters = [str(f) for f in cat["filters"]]
                fills = comp_io.legus_nn_missing_uv_u_fills(
                    full_filters,
                    np.asarray(cat["phot"], dtype=float),
                    np.asarray(cat["detect"], dtype=bool),
                    float(cat["dmod"]),
                    full_filters,
                    subset_filters,
                    use_apparent_magnitude=True,
                )
                comp = comp_io.predict_catalog_completeness_with_nn(
                    phot_app,
                    galaxy_fullname=cat.get("galaxy_fullname", cat["basename"]),
                    nn_dir=nn_dir,
                    subset_filters=subset_filters,
                    full_filter_order=full_filters,
                    missing_band_fills=fills if fills else None,
                )
                idx_del = ~(np.isfinite(comp) & (comp > 0.0))
                cids_to_delete = cat["cid_filterset"][i][idx_del]
                cat_summary.append(
                    {
                        "filters": subset_filters,
                        "n_before": int(len(comp)),
                        "n_removed_comp_le_0": int(len(cids_to_delete)),
                        "comp_min": float(np.nanmin(comp)) if len(comp) else None,
                        "comp_median": float(np.nanmedian(comp)) if len(comp) else None,
                    }
                )
                n_removed += len(cids_to_delete)
                remove_cids(cat, cids_to_delete)
            removed_comp[cat["basename"]] = int(n_removed)
            comp_summary[cat["basename"]] = cat_summary

        removed_dup = {}
        for i, cat in enumerate(catalogs):
            dup = np.zeros(len(cat["ra"]), dtype=bool)
            for other in catalogs[i + 1 :]:
                ra_match = np.equal.outer(cat["ra"], other["ra"])
                dec_match = np.equal.outer(cat["dec"], other["dec"])
                dup |= np.logical_or.reduce(ra_match & dec_match, axis=1)
            cids_to_delete = cat["cid"][dup]
            removed_dup[cat["basename"]] = int(len(cids_to_delete))
            remove_cids(cat, cids_to_delete)
        return {
            "removed_nn_comp_le_0": removed_comp,
            "removed_duplicates": removed_dup,
            "comp_summary": comp_summary,
        }
    finally:
        try:
            sys.path.remove(str(bundled))
        except ValueError:
            pass


def read_tang_catalogs_nn_legacy_rule() -> tuple[list[dict], dict]:
    bundled = ROOT / "pipeline/bundled_pipeline"
    sys.path.insert(0, str(bundled))
    try:
        os.environ["LEGUS_CCT_ROOT"] = "/g/data/jh2/jt4478/make_LEGUS_CCT"
        os.environ["LEGUS_TAB_DIR"] = str(ROOT / "cluster_data")
        readers = load_module("tang_catalog_readers_for_nn_legacy_lf", bundled / "catalog_readers.py")
        catalogs = []
        for path in TANG_CATALOGS:
            cat = readers.reader_register["LEGUS"].read(str(path))
            prepare_filtersets(cat)
            catalogs.append(cat)
        before = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
    finally:
        try:
            sys.path.remove(str(bundled))
        except ValueError:
            pass

    clean = clean_nn_legacy_rule(catalogs, nn_dir=str(ROOT / "nn_models"))
    after = {cat["basename"]: int(len(cat["cid"])) for cat in catalogs}
    clean["before_clean"] = before
    clean["after_clean"] = after
    return catalogs, clean


def band_token(filter_name: str) -> str | None:
    for band in BANDS:
        if band in str(filter_name):
            return band
    return None


def flatten_by_band(catalogs: list[dict]) -> dict[str, np.ndarray]:
    out = {band: [] for band in BANDS}
    for cat in catalogs:
        phot = np.asarray(cat["phot"], dtype=float)
        detect = np.asarray(cat["detect"], dtype=bool)
        for i, filt in enumerate(cat["filters"]):
            band = band_token(str(filt))
            if band is None:
                continue
            vals = phot[:, i]
            ok = detect[:, i] & np.isfinite(vals)
            out[band].append(vals[ok])
    return {
        band: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for band, chunks in out.items()
    }


def class_counts(catalogs: list[dict]) -> dict[str, dict[str, int]]:
    out = {}
    for cat in catalogs:
        cls = np.asarray(cat.get("class", []), dtype=int)
        vals, counts = np.unique(cls, return_counts=True)
        out[cat["basename"]] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
    return out


def filterset_counts(catalogs: list[dict]) -> dict[str, list[dict]]:
    out = {}
    for cat in catalogs:
        rows = []
        for filt, phot in zip(cat["filtersets"], cat["phot_filterset"]):
            rows.append({"filters": [str(f) for f in filt], "n": int(len(phot))})
        out[cat["basename"]] = rows
    return out


def summary_rows(label: str, band_data: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for band, vals in band_data.items():
        row = {"sample": label, "band": band, "n": int(len(vals))}
        if len(vals):
            qs = np.percentile(vals, [5, 16, 50, 84, 95])
            row.update(
                {
                    "min": float(np.min(vals)),
                    "p05": float(qs[0]),
                    "p16": float(qs[1]),
                    "median": float(qs[2]),
                    "p84": float(qs[3]),
                    "p95": float(qs[4]),
                    "max": float(np.max(vals)),
                    "n_bright_m8": int(np.sum(vals < -8.0)),
                    "n_faint_m6_to_m7": int(np.sum((vals >= -7.0) & (vals < -6.0))),
                }
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = [
        "sample",
        "band",
        "n",
        "min",
        "p05",
        "p16",
        "median",
        "p84",
        "p95",
        "max",
        "n_bright_m8",
        "n_faint_m6_to_m7",
    ]
    with path.open("w") as fp:
        fp.write(",".join(keys) + "\n")
        for row in rows:
            fp.write(",".join("" if row.get(k) is None else str(row.get(k, "")) for k in keys) + "\n")


def plot_lfs(
    old_data: dict[str, np.ndarray],
    old_nn_legacy_data: dict[str, np.ndarray],
    tang_data: dict[str, np.ndarray],
    tang_nn_legacy_data: dict[str, np.ndarray],
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
    axes = axes.ravel()
    for ax, band in zip(axes, BANDS):
        old = old_data[band]
        tang = tang_data[band]
        ax.hist(
            old,
            bins=BINS,
            histtype="step",
            linewidth=1.7,
            color="tab:blue",
            label=f"old reader + legacy clean (N={len(old)})",
        )
        old_nn = old_nn_legacy_data[band]
        ax.hist(
            old_nn,
            bins=BINS,
            histtype="step",
            linewidth=1.7,
            color="tab:red",
            label=f"old reader + NN old-rule clean (N={len(old_nn)})",
        )
        ax.hist(
            tang,
            bins=BINS,
            histtype="step",
            linewidth=1.7,
            color="tab:orange",
            label=f"Tang26B reader + clean (N={len(tang)})",
        )
        nn_legacy = tang_nn_legacy_data[band]
        ax.hist(
            nn_legacy,
            bins=BINS,
            histtype="step",
            linewidth=1.7,
            color="tab:green",
            label=f"Tang26B reader + NN old-rule clean (N={len(nn_legacy)})",
        )
        ax.axvline(-6.0, color="0.35", lw=0.9, ls=":")
        ax.set_title(band)
        ax.set_ylabel("clusters")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()
    axes[0].legend(fontsize=8, frameon=False)
    axes[-1].axis("off")
    for ax in axes[:5]:
        ax.set_xlabel("absolute magnitude")
    fig.suptitle("NGC628-c/e observed luminosity functions after reader-specific cleaning")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    old_catalogs, old_clean = read_old_catalogs()
    old_nn_legacy_catalogs, old_nn_legacy_clean = read_old_catalogs_nn_legacy_rule()
    tang_catalogs, tang_clean = read_tang_catalogs()
    tang_nn_legacy_catalogs, tang_nn_legacy_clean = read_tang_catalogs_nn_legacy_rule()

    old_data = flatten_by_band(old_catalogs)
    old_nn_legacy_data = flatten_by_band(old_nn_legacy_catalogs)
    tang_data = flatten_by_band(tang_catalogs)
    tang_nn_legacy_data = flatten_by_band(tang_nn_legacy_catalogs)
    rows = summary_rows("old_reader_legacy_clean", old_data)
    rows += summary_rows("old_reader_nn_legacy_rule_clean", old_nn_legacy_data)
    rows += summary_rows("tang26b_reader_nn_hybrid_clean", tang_data)
    rows += summary_rows("tang26b_reader_nn_legacy_rule_clean", tang_nn_legacy_data)

    stem = OUTDIR / "ngc628_reader_lf_compare"
    write_csv(stem.with_suffix(".csv"), rows)
    plot_lfs(old_data, old_nn_legacy_data, tang_data, tang_nn_legacy_data, stem.with_suffix(".png"))

    meta = {
        "old_catalogs": [str(p) for p in OLD_CATALOGS],
        "tang_catalogs": [str(p) for p in TANG_CATALOGS],
        "old_clean": old_clean,
        "old_nn_legacy_clean": old_nn_legacy_clean,
        "tang_clean": tang_clean,
        "tang_nn_legacy_clean": tang_nn_legacy_clean,
        "old_n_clusters": {cat["basename"]: int(len(cat["cid"])) for cat in old_catalogs},
        "old_nn_legacy_n_clusters": {
            cat["basename"]: int(len(cat["cid"])) for cat in old_nn_legacy_catalogs
        },
        "tang_n_clusters": {cat["basename"]: int(len(cat["cid"])) for cat in tang_catalogs},
        "tang_nn_legacy_n_clusters": {
            cat["basename"]: int(len(cat["cid"])) for cat in tang_nn_legacy_catalogs
        },
        "old_class_counts": class_counts(old_catalogs),
        "old_nn_legacy_class_counts": class_counts(old_nn_legacy_catalogs),
        "tang_class_counts": class_counts(tang_catalogs),
        "tang_nn_legacy_class_counts": class_counts(tang_nn_legacy_catalogs),
        "old_filterset_counts": filterset_counts(old_catalogs),
        "old_nn_legacy_filterset_counts": filterset_counts(old_nn_legacy_catalogs),
        "tang_filterset_counts": filterset_counts(tang_catalogs),
        "tang_nn_legacy_filterset_counts": filterset_counts(tang_nn_legacy_catalogs),
        "csv": str(stem.with_suffix(".csv")),
        "png": str(stem.with_suffix(".png")),
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print("\nSummary:")
    for row in rows:
        if row["band"] == "F555W":
            print(row)


if __name__ == "__main__":
    main()
