#!/usr/bin/env python3
"""
For one LEGUS galaxy (e.g. ngc628-c): build a 5D axis-aligned box in *absolute*
magnitude from observed cluster photometry (same space as slug ``phot_neb_ex``),
drop slug-library rows outside that box, then run NN ``libcomp`` on survivors.

Verbose diagnostics at each step: catalog ranges, library counts, M_V > -6 checks.

Example (HPC):
  cd .../new_pipeline_slug_legus/bundled_pipeline
  python ../scripts/diagnose_obs_domain_libcomp.py \\
    /g/data/jh2/jt4478/cluster_slug/tang \\
    /g/data/jh2/jt4478/make_LEGUS_CCT/ngc628-c/hlsp_legus_hst_acs-wfc3_ngc628-c_multi_v1_fixedvis.tab \\
    --nn-scaler /g/data/jh2/jt4478/Tang26B/nn_models/scaler_phot_ngc628-c.pkl \\
    --nn-model /g/data/jh2/jt4478/Tang26B/nn_models/best_model_phot_ngc628-c.pt \\
    --legus-cct-root /g/data/jh2/jt4478/make_LEGUS_CCT

Notes:
  - Catalog reader ``LEGUS`` (hlsp) applies the same quality cuts as the main
    pipeline; min/max per band are taken over *those* rows (detected bands only).
  - Use ``--range-margin`` to widen each edge slightly (mag).
  - NN is evaluated in chunks to avoid OOM on multi-million-row libraries.

Optional extras (HPC; can be slow):

  --analyze-faint-full-lib
      On the *entire* library (no 5D cut), select rows with absolute M_V > --lib-vmag-max,
      run the *joint* 5-band NN, and print libcomp histograms (near-zero bins).

  --analyze-fourband-perband
      Define "in observed mag range" per band using the same [lo,hi] as [2]. Count library
      rows with >=4 bands in range. On a random subsample, run *single-band* NN calls
      (other bands imputed via scaler means — diagnostic only) and flag rows where at least
      one in-range band has per-band comp < --per-band-small-threshold.

  --analyze-interp-range
      [9] Count library clusters in an "NN interpolation" region and joint libcomp, especially
      for M_V > --lib-vmag-max. Use --interp-range-mode catalog_minmax (same 5D box as [4])
      or scaler_sigma (|(m_app-mean)/scale| <= --sigma-clip per band in scaler space).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

read_cluster = pytest.importorskip("slugpy").read_cluster

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "bundled_pipeline"))

from catalog_readers import reader_register  # noqa: E402
from completeness_io import _load_pickle, predict_catalog_completeness_with_nn  # noqa: E402


def _filt_wave_key(lib_filter_names: list[str], filt: str):
    m = re.search(r"F(\d+)W", str(filt))
    if m is not None:
        return (0, int(m.group(1)), str(filt))
    return (1, lib_filter_names.index(str(filt)), str(filt))


def _v_abs_index(lib_filter_names: list[str]) -> int | None:
    for cand in ("ACS_F555W", "WFC3_UVIS_F555W"):
        if cand in lib_filter_names:
            return lib_filter_names.index(cand)
    for i, fn in enumerate(lib_filter_names):
        if str(fn).endswith("F555W"):
            return i
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "libdir",
        help="cluster_slug library path for read_cluster()",
    )
    p.add_argument(
        "catalog",
        help="LEGUS HLSP .tab catalog path (paired readme for metadata)",
    )
    p.add_argument(
        "--nn-scaler",
        required=True,
        help="StandardScaler pickle for photometry",
    )
    p.add_argument(
        "--nn-model",
        required=True,
        help="Torch completeness model .pt",
    )
    p.add_argument(
        "--legus-cct-root",
        default=os.environ.get("LEGUS_CCT_ROOT", "/g/data/jh2/jt4478/make_LEGUS_CCT"),
    )
    p.add_argument(
        "--legus-tab-dir",
        default=os.environ.get("LEGUS_TAB_DIR", "/home/100/jt4478/slugfiles/LEGUS_cat"),
    )
    p.add_argument(
        "--cattype",
        default="LEGUS",
        help="must be LEGUS (hlsp reader) for this script",
    )
    p.add_argument(
        "--photsystem",
        default="Vega",
        help="photsystem for read_cluster",
    )
    p.add_argument(
        "--range-margin",
        type=float,
        default=0.0,
        help="add this (mag) to each side of [min,max] per band before pruning",
    )
    p.add_argument(
        "--lib-vmag-max",
        type=float,
        default=-6.0,
        help="report clusters with absolute M_V > this (same sign convention as pipeline)",
    )
    p.add_argument(
        "--nn-batch-rows",
        type=int,
        default=65536,
        help="rows per NN chunk (reduce if OOM)",
    )
    p.add_argument(
        "--max-rows-after-mask",
        type=int,
        default=-1,
        help="if >0, only process first N surviving rows (debug / laptop)",
    )
    p.add_argument(
        "--analyze-faint-full-lib",
        action="store_true",
        help="after main steps, run joint NN on all M_V > --lib-vmag-max rows (full lib, no 5D cut)",
    )
    p.add_argument(
        "--analyze-fourband-perband",
        action="store_true",
        help="diagnostic: mag-in-range>=4 bands + per-band NN on a subsample (see --per-band-sample)",
    )
    p.add_argument(
        "--per-band-sample",
        type=int,
        default=300_000,
        help="max rows (from n_in_mag>=4) for per-band NN subsample",
    )
    p.add_argument(
        "--per-band-small-threshold",
        type=float,
        default=0.001,
        help="flag per-band comp below this as 'small' in [8]",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for subsampling in [8]",
    )
    p.add_argument(
        "--analyze-interp-range",
        action="store_true",
        help="[9] count libs in interpolation region; libcomp for all & for M_V > --lib-vmag-max",
    )
    p.add_argument(
        "--interp-range-mode",
        choices=("catalog_minmax", "scaler_sigma"),
        default="scaler_sigma",
        help="catalog_minmax = same 5D ABS box as [4]; scaler_sigma = |z|<=sigma in scaler space",
    )
    p.add_argument(
        "--sigma-clip",
        type=float,
        default=3.0,
        help="for scaler_sigma: max |(m_app-mean_j)/scale_j| per band (apparent mags)",
    )
    return p.parse_args()


def _batched_joint_comp(
    phot_neb_ex: np.ndarray,
    idx_rows: np.ndarray,
    lib_indices_nn: list[int],
    dmod: float,
    subset_filters: list[str],
    nn_full_order: list[str],
    galaxy_fullname: str,
    nn_scaler: str,
    nn_model: str,
    bs: int,
) -> np.ndarray:
    """Joint 5-band NN; idx_rows indexes rows of phot_neb_ex."""
    n = int(idx_rows.size)
    out = np.zeros(n, dtype=float)
    bs = max(1024, bs)
    n_batches = (n + bs - 1) // bs
    for b in range(n_batches):
        lo = b * bs
        hi = min((b + 1) * bs, n)
        rows = idx_rows[lo:hi]
        sub_abs = phot_neb_ex[np.ix_(rows, lib_indices_nn)]
        sub = sub_abs + dmod
        fr = np.all(np.isfinite(sub), axis=1)
        chunk = np.zeros(hi - lo, dtype=float)
        if np.any(fr):
            chunk[fr] = predict_catalog_completeness_with_nn(
                sub[fr],
                galaxy_fullname=galaxy_fullname,
                nn_dir=None,
                subset_filters=subset_filters,
                full_filter_order=nn_full_order,
                nn_scaler_path=nn_scaler,
                nn_model_path=nn_model,
            )
        out[lo:hi] = chunk
        if b == 0 or b == n_batches - 1 or (b + 1) % max(1, n_batches // 10) == 0:
            print(f"      joint batch {b + 1}/{n_batches}")
    return out


def _print_comp_histogram(name: str, comp: np.ndarray) -> None:
    comp = np.asarray(comp, dtype=float)
    finite = comp[np.isfinite(comp)]
    if finite.size == 0:
        print(f"    {name}: no finite values")
        return
    print(f"    {name}: N={finite.size}")
    print(
        f"      min/median/max: {np.min(finite):.6g} / {np.median(finite):.6g} / {np.max(finite):.6g}"
    )
    edges = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    hist, _ = np.histogram(finite, bins=edges)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        print(f"      [{lo:g}, {hi:g}): {hist[i]:d}")
    n_pos = int(np.sum(finite > 0))
    n_tiny = int(np.sum((finite > 0) & (finite <= 1e-6)))
    n_near = int(np.sum((finite > 1e-6) & (finite <= 1e-3)))
    print(f"      count comp>0: {n_pos}  ;  (0,1e-6]: {n_tiny}  ;  (1e-6,1e-3]: {n_near}")


def main() -> None:
    args = _parse_args()
    os.environ["LEGUS_CCT_ROOT"] = args.legus_cct_root
    os.environ["LEGUS_TAB_DIR"] = args.legus_tab_dir

    if args.cattype != "LEGUS":
        raise SystemExit("This script expects --cattype LEGUS (hlsp reader).")

    print("=" * 72)
    print("[1] Read LEGUS catalog (pipeline reader; includes LEGUS quality cuts)")
    print(f"    path: {args.catalog}")
    data = reader_register["LEGUS"].read(args.catalog)
    filters_cat = [str(f) for f in data["filters"]]
    phot = np.asarray(data["phot"], dtype=float)
    detect = np.asarray(data["detect"], dtype=bool)
    dmod = float(data["dmod"])
    galaxy_fullname = data.get("galaxy_fullname", "")
    print(f"    galaxy_fullname: {galaxy_fullname!r}")
    print(f"    dmod: {dmod:.4f}")
    print(f"    n_clusters (after reader cuts): {phot.shape[0]}")
    print(f"    filters ({len(filters_cat)}): {filters_cat}")

    print("\n" + "=" * 72)
    print("[2] Per-band observed absolute magnitude range (catalog)")
    print("    (min/max over rows with detect=True in that band; phot is ABS mag)")
    bounds_lo = []
    bounds_hi = []
    for i, fname in enumerate(filters_cat):
        m = detect[:, i] & np.isfinite(phot[:, i])
        n = int(np.sum(m))
        if n == 0:
            print(f"    *** {fname}: NO valid detections — cannot define range ***")
            bounds_lo.append(np.nan)
            bounds_hi.append(np.nan)
            continue
        lo = float(np.min(phot[m, i]))
        hi = float(np.max(phot[m, i]))
        p1, p50, p99 = np.percentile(phot[m, i], [1, 50, 99])
        print(
            f"    {fname}: n={n:6d}  min={lo:8.4f}  max={hi:8.4f}  "
            f"p01={p1:8.4f} p50={p50:8.4f} p99={p99:8.4f}"
        )
        lo -= args.range_margin
        hi += args.range_margin
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        print(f"         -> box edges after margin {args.range_margin:+.3f}: [{lo:.4f}, {hi:.4f}]")

    if any(np.isnan(bounds_lo[j]) for j in range(len(filters_cat))):
        raise SystemExit("Abort: missing per-band range.")

    print("\n" + "=" * 72)
    print("[3] Load slug library (read_cluster)")
    lib = read_cluster(args.libdir, photsystem=args.photsystem, read_filters=filters_cat)
    lib_filter_names = [str(f) for f in lib.filter_names]
    phot_neb_ex = np.asarray(lib.phot_neb_ex, dtype=float)
    n_lib_total = phot_neb_ex.shape[0]
    print(f"    libdir: {args.libdir}")
    print(f"    filter_names from lib: {lib_filter_names}")
    print(f"    phot_neb_ex shape: {phot_neb_ex.shape} (rows x bands)")

    # Align catalog filter names to library columns (same strings as read_filters)
    lib_cols = []
    for f in filters_cat:
        if f not in lib_filter_names:
            raise SystemExit(f"Catalog filter {f!r} not in lib.filter_names {lib_filter_names}")
        lib_cols.append(lib_filter_names.index(f))
    lib_cols = np.array(lib_cols, dtype=int)

    # NN order: wavelength-sorted (same as analyze_catalog_mid_mdd.py); used everywhere below
    subset_filters = sorted(filters_cat, key=lambda f: _filt_wave_key(lib_filter_names, f))
    nn_full_order = subset_filters
    lib_indices_nn = [lib_filter_names.index(f) for f in subset_filters]

    print("\n" + "=" * 72)
    print("[4] 5D domain mask: keep library row iff ALL bands inside [lo,hi] (ABS mag)")
    inside = np.ones(n_lib_total, dtype=bool)
    for j, fname in enumerate(filters_cat):
        x = phot_neb_ex[:, lib_cols[j]]
        lo, hi = bounds_lo[j], bounds_hi[j]
        band_ok = np.isfinite(x) & (x >= lo) & (x <= hi)
        n_out = int(np.sum(~band_ok))
        print(
            f"    {fname}: lib inside box {int(np.sum(band_ok))} / {n_lib_total} "
            f"(outside or non-finite: {n_out})"
        )
        inside &= band_ok

    n_inside = int(np.sum(inside))
    print(f"\n    *** SURVIVORS after full 5D intersection: {n_inside} / {n_lib_total} ***")
    print(f"    fraction kept: {n_inside / max(n_lib_total, 1):.6e}")

    v_idx = _v_abs_index(lib_filter_names)
    if v_idx is None:
        print("    [warn] Could not resolve V band (F555W) in library — skip M_V diagnostics.")
    else:
        v_all = phot_neb_ex[:, v_idx]
        n_faint_all = int(np.sum(np.isfinite(v_all) & (v_all > args.lib_vmag_max)))
        v_in = v_all[inside]
        n_faint_in = int(np.sum(np.isfinite(v_in) & (v_in > args.lib_vmag_max)))
        print(
            f"\n    Absolute M_V (lib column {lib_filter_names[v_idx]}): "
            f"M_V > {args.lib_vmag_max} in FULL lib: {n_faint_all} rows"
        )
        print(
            f"    Absolute M_V: M_V > {args.lib_vmag_max} among 5D survivors: {n_faint_in} rows"
        )

    idx_keep = None
    comp = None
    if n_inside == 0:
        print("\n    No rows inside 5D domain — skipping [5]/[6] NN on survivors.")
    else:
        idx_keep = np.flatnonzero(inside)
        if args.max_rows_after_mask > 0:
            idx_keep = idx_keep[: args.max_rows_after_mask]
            print(
                f"\n    [debug] truncating to first {len(idx_keep)} survivors "
                "(--max-rows-after-mask)"
            )

        n_surv = int(idx_keep.size)
        print("\n" + "=" * 72)
        print("[5] NN libcomp on survivors (apparent mag = ABS + dmod)")
        print(f"    subset_filters (NN order): {subset_filters}")
        print(f"    rows for NN: {n_surv} (batched)")
        comp = _batched_joint_comp(
            phot_neb_ex,
            idx_keep,
            lib_indices_nn,
            dmod,
            subset_filters,
            nn_full_order,
            galaxy_fullname,
            args.nn_scaler,
            args.nn_model,
            args.nn_batch_rows,
        )
        fr = np.isfinite(comp)
        print(
            f"    rows finite comp: {int(np.sum(fr))} / {n_surv}  |  "
            f"min/median/max: {np.min(comp):.6g} / {np.median(comp):.6g} / {np.max(comp):.6g}"
        )

        print("\n" + "=" * 72)
        print("[6] M_V > cutoff vs NN output (5D survivors only)")
        if v_idx is not None:
            v_surv = phot_neb_ex[idx_keep, v_idx]
            faint = np.isfinite(v_surv) & (v_surv > args.lib_vmag_max)
            n_faint = int(np.sum(faint))
            eps = 1e-12
            bad = faint & (comp > eps)
            n_bad = int(np.sum(bad))
            print(f"    Survivors with M_V > {args.lib_vmag_max}: {n_faint}")
            print(f"    Of those, with libcomp > {eps:g}: {n_bad}")
            if n_bad > 0:
                print(
                    "    [!!] Some faint (M_V > cutoff) rows still have positive NN comp "
                    "(expected if you do NOT apply physical zeroing)."
                )
                w = np.where(bad)[0]
                show = w[: min(10, len(w))]
                print(f"    First indices (into survivor array): {show.tolist()}")
                print(f"    Example: M_V={v_surv[show[0]]:.4f}  comp={comp[show[0]]:.6g}")
            else:
                print("    No faint survivors with comp > eps.")

    # --- [7] Full library, no 5D cut: faint (M_V > cutoff) joint libcomp ---
    if args.analyze_faint_full_lib:
        print("\n" + "=" * 72)
        print("[7] FULL library (no observed 5D cut): M_V > cutoff → joint NN libcomp")
        print("    (Uses all rows with finite V and M_V > --lib-vmag-max; can be millions.)")
        if v_idx is None:
            print("    [skip] No V band index.")
        else:
            v_all = phot_neb_ex[:, v_idx]
            idx_faint = np.flatnonzero(np.isfinite(v_all) & (v_all > args.lib_vmag_max))
            print(f"    N rows with M_V > {args.lib_vmag_max}: {idx_faint.size}")
            if idx_faint.size == 0:
                print("    Nothing to evaluate.")
            else:
                comp_f = _batched_joint_comp(
                    phot_neb_ex,
                    idx_faint,
                    lib_indices_nn,
                    dmod,
                    subset_filters,
                    nn_full_order,
                    galaxy_fullname,
                    args.nn_scaler,
                    args.nn_model,
                    args.nn_batch_rows,
                )
                _print_comp_histogram("joint libcomp (faint-only, full lib)", comp_f)

    # --- [8] Mag-in-range >=4 bands (proxy); per-band NN on subsample ---
    if args.analyze_fourband_perband:
        print("\n" + "=" * 72)
        print("[8] Proxy: >=4 bands with ABS mag in catalog [min,max] per band; per-band NN")
        print(
            "    WARNING: single-band NN uses imputed means in other bands — "
            "diagnostic only, not a physical marginal P(det|m_b)."
        )
        in_band = np.zeros((n_lib_total, len(filters_cat)), dtype=bool)
        for j, fname in enumerate(filters_cat):
            x = phot_neb_ex[:, lib_cols[j]]
            lo, hi = bounds_lo[j], bounds_hi[j]
            in_band[:, j] = np.isfinite(x) & (x >= lo) & (x <= hi)
        n_in_mag = np.sum(in_band, axis=1)
        mask4 = n_in_mag >= 4
        n4 = int(np.sum(mask4))
        print(f"    Library rows with >=4 bands in observed mag range: {n4} / {n_lib_total}")
        idx4 = np.flatnonzero(mask4)
        if n4 == 0:
            print("    Nothing to subsample.")
        else:
            rng = np.random.default_rng(args.random_seed)
            n_take = min(int(args.per_band_sample), n4)
            choice = rng.choice(idx4, size=n_take, replace=False)
            print(f"    Subsample for per-band NN: {n_take} rows (seed={args.random_seed})")

            comp_pb = np.zeros((n_take, len(filters_cat)), dtype=float)
            for jb in range(len(filters_cat)):
                fname = filters_cat[jb]
                lib_col = lib_cols[jb]
                bs = max(1024, args.nn_batch_rows)
                nbat = (n_take + bs - 1) // bs
                for bb in range(nbat):
                    lo = bb * bs
                    hi = min((bb + 1) * bs, n_take)
                    rows = choice[lo:hi]
                    sub_abs = phot_neb_ex[rows, lib_col].reshape(-1, 1)
                    sub = sub_abs + dmod
                    fr = np.isfinite(sub[:, 0])
                    chunk = np.zeros(hi - lo, dtype=float)
                    if np.any(fr):
                        chunk[fr] = predict_catalog_completeness_with_nn(
                            sub[fr],
                            galaxy_fullname=galaxy_fullname,
                            nn_dir=None,
                            subset_filters=[fname],
                            full_filter_order=nn_full_order,
                            nn_scaler_path=args.nn_scaler,
                            nn_model_path=args.nn_model,
                        )
                    comp_pb[lo:hi, jb] = chunk
                print(f"      per-band column {jb} ({fname}): done")

            thr = float(args.per_band_small_threshold)
            n_any_small = 0
            min_inrange = []
            for r in range(n_take):
                ri = int(choice[r])
                bands_in = np.where(in_band[ri])[0]
                if bands_in.size < 4:
                    continue
                vals = comp_pb[r, bands_in]
                if np.any(vals < thr):
                    n_any_small += 1
                min_inrange.append(float(np.min(vals)))

            print(
                f"    Among subsample: rows with ANY in-range band per-band comp < {thr}: "
                f"{n_any_small} / {n_take} ({100.0 * n_any_small / max(n_take, 1):.4f}%)"
            )
            if min_inrange:
                mi = np.array(min_inrange, dtype=float)
                print(
                    f"    Distribution of min(comp | in-range bands) over subsample: "
                    f"median={np.median(mi):.6g} min={np.min(mi):.6g}"
                )

            comp_joint_s = _batched_joint_comp(
                phot_neb_ex,
                choice,
                lib_indices_nn,
                dmod,
                subset_filters,
                nn_full_order,
                galaxy_fullname,
                args.nn_scaler,
                args.nn_model,
                args.nn_batch_rows,
            )
            print("    Joint NN on *same* subsample (for comparison):")
            _print_comp_histogram("joint libcomp (>=4 mag-bands subsample)", comp_joint_s)

            # Rows: >=4 in-range AND some in-range per-band comp tiny
            tricky = 0
            for r in range(n_take):
                ri = int(choice[r])
                bands_in = np.where(in_band[ri])[0]
                if bands_in.size < 4:
                    continue
                vals = comp_pb[r, bands_in]
                if np.any(vals < thr) and np.any(vals >= thr):
                    tricky += 1
            print(
                f"    Subsample rows with >=4 in-range bands AND mix of "
                f"comp>={thr} and comp<{thr} among those bands: {tricky} / {n_take}"
            )

    # --- [9] Interpolation region: in-range count, faint count, joint libcomp ---
    if args.analyze_interp_range:
        print("\n" + "=" * 72)
        print("[9] NN interpolation region: N in range, N faint (M_V > cutoff), libcomp")
        print(f"    --interp-range-mode {args.interp_range_mode!r}", end="")
        if args.interp_range_mode == "scaler_sigma":
            print(f", --sigma-clip {args.sigma_clip:g}")
        else:
            print(" (same ABS mag box as [4])")

        if v_idx is None:
            print("    [skip] No V band index.")
        else:
            v_all = phot_neb_ex[:, v_idx]
            if args.interp_range_mode == "catalog_minmax":
                interp_mask = np.asarray(inside, dtype=bool)
            else:
                scaler = _load_pickle(args.nn_scaler)
                mean = np.asarray(getattr(scaler, "mean_", np.zeros(5)), dtype=np.float64).ravel()
                scale = np.asarray(getattr(scaler, "scale_", np.ones(5)), dtype=np.float64).ravel()
                scale = np.where(scale > 1e-20, scale, 1.0)
                if mean.size != len(nn_full_order) or scale.size != len(nn_full_order):
                    raise SystemExit(
                        f"Scaler dim {mean.size} != {len(nn_full_order)} (check scaler vs filter order)."
                    )
                print(f"    scaler mean ({nn_full_order}): {mean}")
                print(f"    scaler scale: {scale}")
                x_app = phot_neb_ex[:, lib_indices_nn] + dmod
                finite_all = np.all(np.isfinite(x_app), axis=1)
                z = (x_app - mean) / scale
                interp_mask = finite_all & np.all(np.abs(z) <= args.sigma_clip, axis=1)

            n_ir = int(np.sum(interp_mask))
            faint_ir = interp_mask & np.isfinite(v_all) & (v_all > args.lib_vmag_max)
            n_faint_ir = int(np.sum(faint_ir))
            print(f"    N library rows in interpolation region: {n_ir} / {n_lib_total}")
            print(
                f"    Of those, M_V > {args.lib_vmag_max} (faint): {n_faint_ir}"
            )
            if args.interp_range_mode == "catalog_minmax" and n_faint_ir == 0:
                print(
                    "    [note] catalog_minmax usually has V max ~-6; faint-in-box is often 0. "
                    "Try --interp-range-mode scaler_sigma."
                )

            if n_ir == 0:
                print("    Nothing to evaluate with joint NN.")
            else:
                idx_ir = np.flatnonzero(interp_mask)
                comp_ir = _batched_joint_comp(
                    phot_neb_ex,
                    idx_ir,
                    lib_indices_nn,
                    dmod,
                    subset_filters,
                    nn_full_order,
                    galaxy_fullname,
                    args.nn_scaler,
                    args.nn_model,
                    args.nn_batch_rows,
                )
                _print_comp_histogram("joint libcomp | in interpolation region", comp_ir)

                v_ir = v_all[idx_ir]
                faint_loc = np.isfinite(v_ir) & (v_ir > args.lib_vmag_max)
                if np.any(faint_loc):
                    print(
                        f"    Faint (M_V > {args.lib_vmag_max}) inside region — "
                        "libcomp near 0?"
                    )
                    _print_comp_histogram(
                        "joint libcomp | faint & in interpolation region",
                        comp_ir[faint_loc],
                    )
                    print(
                        "    (Phot NN does not force comp→0 for faint; pipeline uses faint mask.)"
                    )
                else:
                    print(
                        f"    No faint rows inside this interpolation region — "
                        "skip faint-only histogram."
                    )

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
