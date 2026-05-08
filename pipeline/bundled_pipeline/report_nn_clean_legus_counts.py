#!/usr/bin/env python3
"""Report observed-cluster counts after LEGUS reader + clean_legus.

Scans galaxies available in ``nn_models/``. If multiple NN model directories
share the same base prefix before the final ``-`` (for example
``ngc1313-e``/``ngc1313-w``), they are combined into one galaxy-level run.
"""

import argparse
import glob
import os
import os.path as osp
import warnings
from collections import defaultdict

import numpy as np

from catalog_readers import reader_register
from clean_legus import clean_legus

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = None

if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count observed clusters after catalog reader + clean_legus "
        "for galaxies with NN models."
    )
    parser.add_argument(
        "--legus-cct-root",
        default="/g/data/jh2/jt4478/make_LEGUS_CCT",
        help="root for LEGUS CCT ancillary files and per-galaxy catalogs",
    )
    parser.add_argument(
        "--legus-tab-dir",
        default="/g/data/jh2/jt4478/Tang26B/cluster_data",
        help="directory for legacy LEGUS readme fallbacks",
    )
    parser.add_argument(
        "--cluster-slug-lib-dir",
        default="/g/data/jh2/jt4478/cluster_slug",
        help="cluster_slug parent directory for env CLUSTER_SLUG_LIB_DIR",
    )
    parser.add_argument(
        "--nn-dir",
        default="/g/data/jh2/jt4478/Tang26B/nn_models",
        help="directory with per-galaxy NN completeness artifacts",
    )
    parser.add_argument(
        "--catalog-glob",
        default="hlsp_legus*{galaxy_name}*.tab",
        help="glob pattern used to auto-discover LEGUS catalogs",
    )
    parser.add_argument(
        "--comp-threshold",
        type=float,
        default=0.01,
        help="minimum expected completeness for keeping observed clusters",
    )
    parser.add_argument(
        "--lib-vmag-max",
        type=float,
        default=-6.0,
        help="hybrid-aligned V-band cut used inside clean_legus",
    )
    parser.add_argument(
        "--min-bands-nonzero",
        type=int,
        default=4,
        help="hybrid-aligned minimum number of detected bands",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-catalog clean_legus messages",
    )
    parser.add_argument(
        "--output",
        default="/g/data/jh2/jt4478/Tang26B/output_io/nn_clean_legus_counts.tsv",
        help="output TSV path",
    )
    return parser.parse_args()


def group_nn_galaxies(nn_dir):
    names = sorted(
        entry
        for entry in os.listdir(nn_dir)
        if osp.isdir(osp.join(nn_dir, entry))
    )
    grouped = defaultdict(list)
    for name in names:
        base = name.rsplit("-", 1)[0] if "-" in name else name
        grouped[base].append(name)

    ordered = []
    used = set()
    for name in names:
        if name in used:
            continue
        base = name.rsplit("-", 1)[0] if "-" in name else name
        members = sorted(grouped[base])
        if len(members) > 1:
            ordered.append((base, members))
            used.update(members)
        else:
            ordered.append((name, [name]))
            used.add(name)
    return ordered


def discover_catalogs(galaxy_names, legus_cct_root, catalog_glob):
    catalogs = []
    for gal in galaxy_names:
        gal_key = gal.split("_")[0]
        cat_dir = osp.join(legus_cct_root, gal_key)
        pattern = catalog_glob.replace("{galaxy_name}", gal_key)
        matches = sorted(glob.glob(osp.join(cat_dir, pattern)))
        if not matches:
            raise FileNotFoundError(
                f"No LEGUS catalog found for galaxy '{gal}' in {cat_dir} "
                f"with pattern {pattern}"
            )
        catalogs.append(matches[0])
    return catalogs


def read_catalogs(catalog_paths):
    catalogs = []
    for cat in catalog_paths:
        data = reader_register["LEGUS"].read(cat)

        filtersets = []
        filtersets_detect = []
        fset = np.zeros(len(data["phot"]))
        for i, detect in enumerate(data["detect"]):
            filt = list(np.array(data["filters"])[detect])
            if filt not in filtersets:
                filtersets.append(filt)
                filtersets_detect.append(np.copy(detect))
            fset[i] = filtersets.index(filt)
        data["filtersets"] = filtersets
        data["filtersets_index"] = fset
        data["filtersets_detect"] = filtersets_detect

        cid_filterset = []
        phot_filterset = []
        photerr_filterset = []
        for i, detect in enumerate(data["filtersets_detect"]):
            idx = data["filtersets_index"] == i
            cid_filterset.append(data["cid"][idx])
            phot_filterset.append(data["phot"][idx][:, detect])
            photerr_filterset.append(data["photerr"][idx][:, detect])
        data["cid_filterset"] = cid_filterset
        data["phot_filterset"] = phot_filterset
        data["photerr_filterset"] = photerr_filterset
        catalogs.append(data)
    return catalogs


def main():
    args = parse_args()

    os.environ["LEGUS_CCT_ROOT"] = args.legus_cct_root
    os.environ["LEGUS_TAB_DIR"] = args.legus_tab_dir
    os.environ["CLUSTER_SLUG_LIB_DIR"] = args.cluster_slug_lib_dir

    groups = group_nn_galaxies(args.nn_dir)

    rows = ["galaxy_group\tgalaxy_names\tstatus\tn_catalogs\tn_before\tn_after"]
    for group_name, galaxy_names in groups:
        try:
            catalog_paths = discover_catalogs(
                galaxy_names, args.legus_cct_root, args.catalog_glob
            )
            catalogs = read_catalogs(catalog_paths)
            n_before = sum(len(cat["phot"]) for cat in catalogs)
            n_after = clean_legus(
                catalogs,
                args.verbose,
                nn_dir=args.nn_dir,
                comp_threshold=args.comp_threshold,
                enforce_hybrid_criteria=True,
                lib_vmag_max=args.lib_vmag_max,
                min_bands_nonzero=args.min_bands_nonzero,
            )
            rows.append(
                f"{group_name}\t{' '.join(galaxy_names)}\tok\t{len(catalogs)}\t"
                f"{n_before}\t{n_after}"
            )
        except FileNotFoundError:
            continue

    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="ascii") as fp:
        fp.write("\n".join(rows) + "\n")

    print(f"Wrote {len(rows) - 1} rows to {args.output}")


if __name__ == "__main__":
    main()
