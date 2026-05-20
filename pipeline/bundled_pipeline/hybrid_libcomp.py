"""
Hybrid LEGUS library completeness: NN inside 5D observed ABS mag box + hard per-band gates.

Main entry: :class:`HybridLegusLibCompletenessCalculator` — intended for use from
``analyze_catalog_mid_mdd.py`` (or scripts) after ``read_cluster`` and catalog bounds are known.

Rules (per-band hard flags ``pband[j]`` in {0,1}):
    * Band j: 1 if finite ABS mag <= catalog faint limit ``bounds_hi[j]`` (max observed in band),
      else 0.
    * V (F555W): additionally force 0 if absolute M_V > ``lib_vmag_max`` (default -6).

Joint rule ``rule_ok`` (all must hold to keep NN output):
    * V per-band == 1
    * **B or I**: not (**B==0 and I==0**) — i.e. reject only when **both** B (F435W/F438W)
      and I (F814W) are 0; if either is 1, this gate passes.
    * At least ``min_bands_nonzero`` bands have per-band flag 1 (default 4).

Output: ``comp_hybrid = comp_nn * rule_ok`` with ``comp_nn == 0`` outside the 5D intersection box.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from completeness_io import predict_catalog_completeness_with_nn


def legus_catalog_abs_bounds(
    phot: np.ndarray,
    detect: np.ndarray,
    filters_cat: Sequence[str],
    range_margin: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """
    Per-band [lo, hi] in **absolute** magnitude from LEGUS-like ``phot`` + ``detect`` arrays
    (same convention as ``catalog_reader_legus_hlsp``).
    """
    phot = np.asarray(phot, dtype=float)
    detect = np.asarray(detect, dtype=bool)
    bounds_lo: List[float] = []
    bounds_hi: List[float] = []
    for i in range(len(filters_cat)):
        m = detect[:, i] & np.isfinite(phot[:, i])
        if not np.any(m):
            raise ValueError(f"No valid detections for band {filters_cat[i]}")
        lo = float(np.min(phot[m, i])) - float(range_margin)
        hi = float(np.max(phot[m, i])) + float(range_margin)
        bounds_lo.append(lo)
        bounds_hi.append(hi)
    return bounds_lo, bounds_hi


def library_filter_alias_candidates(filt: str) -> List[str]:
    """
    Candidate library filter names for one catalog filter.

    Some LEGUS catalogs use F435W where the SLUG library contains the nearby
    F438W band, or vice versa. Treat these as library-column aliases while
    leaving the observed catalog filter names unchanged for NN metadata.
    """
    name = str(filt)
    candidates = [name]
    if "F435W" in name:
        candidates.append(name.replace("F435W", "F438W"))
    if "F438W" in name:
        candidates.append(name.replace("F438W", "F435W"))
    return candidates


def resolve_lib_filter_name(lib_filter_names: Sequence[str], filt: str) -> str:
    """Return the first library filter matching ``filt`` or its aliases."""
    names = [str(f) for f in lib_filter_names]
    for cand in library_filter_alias_candidates(str(filt)):
        if cand in names:
            return cand
    raise ValueError(
        f"Could not resolve catalog filter {filt!r} in library filters {names!r}; "
        f"tried {library_filter_alias_candidates(str(filt))!r}"
    )


def _filt_wave_key(lib_filter_names: List[str], filt: str) -> Tuple[int, int, str]:
    match = re.search(r"F(\d+)W", str(filt))
    if match is not None:
        return (0, int(match.group(1)), str(filt))
    return (1, lib_filter_names.index(resolve_lib_filter_name(lib_filter_names, str(filt))), str(filt))


def lib_column_indices(lib_filter_names: Sequence[str], filters_cat: Sequence[str]) -> np.ndarray:
    names = [str(f) for f in lib_filter_names]
    return np.array([names.index(resolve_lib_filter_name(names, str(f))) for f in filters_cat], dtype=int)


def v_band_lib_index(lib_filter_names: Sequence[str]) -> Optional[int]:
    names = [str(f) for f in lib_filter_names]
    for cand in ("ACS_F555W", "WFC3_UVIS_F555W", "ACS_F606W", "WFC3_UVIS_F606W"):
        if cand in names:
            return names.index(cand)
    for i, fn in enumerate(names):
        if str(fn).endswith("F555W") or str(fn).endswith("F606W"):
            return i
    return None


def v_band_catalog_index(filters_cat: Sequence[str]) -> Optional[int]:
    for token in ("F555W", "F606W"):
        for i, f in enumerate(filters_cat):
            if token in str(f):
                return i
    return None


def b_band_catalog_index(filters_cat: Sequence[str]) -> Optional[int]:
    for i, f in enumerate(filters_cat):
        if "F435W" in f or "F438W" in f:
            return i
    return None


def i_band_catalog_index(filters_cat: Sequence[str]) -> Optional[int]:
    for i, f in enumerate(filters_cat):
        if "F814W" in f:
            return i
    return None


def hybrid_per_band_hard_flags(
    phot_neb_ex: np.ndarray,
    lib_cols: np.ndarray,
    bounds_hi: Sequence[float],
    v_lib_idx: int,
    v_cat_idx: int,
    lib_vmag_max: float,
) -> np.ndarray:
    """
    Per-band hard flags, shape (N, nband) in **catalog band order**:
    1 if ABS mag <= bounds_hi[j], else 0; V catalog column forced to 0 where
    absolute M_V (from ``phot_neb_ex[:, v_lib_idx]``) > lib_vmag_max.
    """
    phot_neb_ex = np.asarray(phot_neb_ex, dtype=float)
    ntot, nband = phot_neb_ex.shape[0], len(lib_cols)
    pband = np.zeros((ntot, nband), dtype=np.float32)
    hi = np.asarray(bounds_hi, dtype=float)
    for j in range(nband):
        mcol = phot_neb_ex[:, int(lib_cols[j])]
        pband[:, j] = (np.isfinite(mcol) & (mcol <= hi[j])).astype(np.float32)
    v_abs = phot_neb_ex[:, v_lib_idx]
    faint_v = np.isfinite(v_abs) & (v_abs > lib_vmag_max)
    pband[faint_v, v_cat_idx] = 0.0
    return pband


def hybrid_rule_ok_mask(
    pband: np.ndarray,
    v_cat_idx: int,
    b_cat_idx: Optional[int],
    i_cat_idx: Optional[int],
    min_bands_nonzero: int,
) -> np.ndarray:
    """
    LEGUS-style gate: V==1, (B==1 or I==1), and at least ``min_bands_nonzero`` bands are 1.
    """
    ok_v = pband[:, v_cat_idx] > 0.5
    ok_b = pband[:, b_cat_idx] > 0.5 if b_cat_idx is not None else np.zeros(len(pband), dtype=bool)
    ok_i = pband[:, i_cat_idx] > 0.5 if i_cat_idx is not None else np.zeros(len(pband), dtype=bool)
    ok_b_or_i = ok_b | ok_i
    n_ok = np.sum(pband > 0.5, axis=1)
    return ok_v & ok_b_or_i & (n_ok >= int(min_bands_nonzero))


def inside_5d_abs_box(
    phot_neb_ex: np.ndarray,
    lib_cols: np.ndarray,
    bounds_lo: Sequence[float],
    bounds_hi: Sequence[float],
) -> np.ndarray:
    """Boolean mask: all bands finite and in [lo_j, hi_j] (ABS mag)."""
    phot_neb_ex = np.asarray(phot_neb_ex, dtype=float)
    ntot = phot_neb_ex.shape[0]
    nband = len(lib_cols)
    inside = np.ones(ntot, dtype=bool)
    lo = np.asarray(bounds_lo, dtype=float)
    hi = np.asarray(bounds_hi, dtype=float)
    for j in range(nband):
        x = phot_neb_ex[:, int(lib_cols[j])]
        inside &= np.isfinite(x) & (x >= lo[j]) & (x <= hi[j])
    return inside


def batched_joint_nn_libcomp(
    phot_neb_ex: np.ndarray,
    idx_rows: np.ndarray,
    lib_indices_nn: List[int],
    dmod: float,
    subset_filters: List[str],
    nn_full_order: List[str],
    galaxy_fullname: str,
    nn_scaler_path: str,
    nn_model_path: str,
    batch_rows: int,
    missing_band_fills: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Joint NN completeness for indexed rows; returns array of shape (idx_rows.size,)."""
    n = int(idx_rows.size)
    out = np.zeros(n, dtype=float)
    bs = max(1024, int(batch_rows))
    n_batches = (n + bs - 1) // bs
    progress = os.environ.get("HYBRID_LIBCOMP_PROGRESS", "")
    for b in range(n_batches):
        if progress:
            print(f"[hybrid-libcomp] NN batch {b + 1}/{n_batches}", flush=True)
        lo = b * bs
        hi = min((b + 1) * bs, n)
        rows = idx_rows[lo:hi]
        sub_abs = phot_neb_ex[np.ix_(rows, lib_indices_nn)]
        sub = sub_abs + float(dmod)
        fr = np.all(np.isfinite(sub), axis=1)
        chunk = np.zeros(hi - lo, dtype=float)
        if np.any(fr):
            chunk[fr] = predict_catalog_completeness_with_nn(
                sub[fr],
                galaxy_fullname=galaxy_fullname,
                nn_dir=None,
                subset_filters=subset_filters,
                full_filter_order=nn_full_order,
                nn_scaler_path=nn_scaler_path,
                nn_model_path=nn_model_path,
                missing_band_fills=missing_band_fills,
            )
        out[lo:hi] = chunk
    return out


def hybrid_libcomp_summary(
    n_total: int,
    inside_5d: np.ndarray,
    rule_ok: np.ndarray,
    comp_nn: np.ndarray,
    comp_hybrid: np.ndarray,
    *,
    eps: float = 1e-15,
) -> Dict[str, Any]:
    """
    Aggregate counts and completeness stats for hybrid libcomp output.

    * **Kept**: ``comp_hybrid > eps`` (finite).
    * **Dropped** breakdown (mutually exclusive): outside 5D box; inside but ``rule_ok`` false;
      inside with rules but ``comp_nn <= eps``.
    """
    inside = np.asarray(inside_5d, dtype=bool).ravel()
    rule = np.asarray(rule_ok, dtype=bool).ravel()
    cnn = np.asarray(comp_nn, dtype=float).ravel()
    ch = np.asarray(comp_hybrid, dtype=float).ravel()
    if not (inside.size == rule.size == cnn.size == ch.size == int(n_total)):
        raise ValueError("inside_5d, rule_ok, comp_nn, comp_hybrid must match n_total")

    kept = (ch > eps) & np.isfinite(ch)
    n_kept = int(np.sum(kept))
    n_drop = int(n_total - n_kept)

    n_in = int(np.sum(inside))
    n_out = int(n_total - n_in)
    n_rule = int(np.sum(rule))

    drop_outside = int(np.sum(~inside))
    drop_rule = int(np.sum(inside & ~rule))
    drop_nn = int(np.sum(inside & rule & (cnn <= eps)))

    out: Dict[str, Any] = {
        "n_total": int(n_total),
        "n_kept": n_kept,
        "n_dropped": n_drop,
        "n_inside_5d": n_in,
        "n_outside_5d": n_out,
        "n_rule_ok": n_rule,
        "dropped_outside_5d": drop_outside,
        "dropped_inside_rule_fail": drop_rule,
        "dropped_inside_nn_zero": drop_nn,
    }
    if n_kept > 0:
        hk = ch[kept]
        nk = cnn[kept]
        out["kept_comp_hybrid_min"] = float(np.min(hk))
        out["kept_comp_hybrid_max"] = float(np.max(hk))
        out["kept_comp_hybrid_mean"] = float(np.mean(hk))
        out["kept_comp_hybrid_median"] = float(np.median(hk))
        out["kept_comp_nn_mean"] = float(np.mean(nk))
        out["kept_comp_nn_median"] = float(np.median(nk))
    else:
        out["kept_comp_hybrid_min"] = None
        out["kept_comp_hybrid_max"] = None
        out["kept_comp_hybrid_mean"] = None
        out["kept_comp_hybrid_median"] = None
        out["kept_comp_nn_mean"] = None
        out["kept_comp_nn_median"] = None
    return out


def format_hybrid_libcomp_summary(summary: Dict[str, Any]) -> str:
    """Human-readable block (CLI / logs)."""
    lines = [
        f"Total clusters: {summary['n_total']}",
        f"  Kept (comp_hybrid > 0): {summary['n_kept']}",
        f"  Dropped: {summary['n_dropped']}",
        "  Dropped breakdown (exclusive):",
        f"    outside 5D observed box: {summary['dropped_outside_5d']}",
        f"    inside 5D, rules failed: {summary['dropped_inside_rule_fail']}",
        f"    inside 5D, rules OK, NN ~0: {summary['dropped_inside_nn_zero']}",
        f"  Rows inside 5D box: {summary['n_inside_5d']}; rule_ok (all rows): {summary['n_rule_ok']}",
    ]
    if summary["n_kept"] > 0 and summary["kept_comp_hybrid_mean"] is not None:
        lines.extend(
            [
                "  Kept rows — comp_hybrid: "
                f"min={summary['kept_comp_hybrid_min']:.6g}, "
                f"median={summary['kept_comp_hybrid_median']:.6g}, "
                f"mean={summary['kept_comp_hybrid_mean']:.6g}, "
                f"max={summary['kept_comp_hybrid_max']:.6g}",
                "  Kept rows — comp_nn (same rows): "
                f"median={summary['kept_comp_nn_median']:.6g}, "
                f"mean={summary['kept_comp_nn_mean']:.6g}",
            ]
        )
    else:
        lines.append("  Kept rows — no survivors; completeness stats N/A.")
    return "\n".join(lines)


class HybridLegusLibCompletenessCalculator:
    """
    Callable from the main pipeline once catalog bounds and filter names are fixed.

    Parameters
    ----------
    filters_cat :
        Catalog filter names in catalog column order (same as ``read_cluster(..., read_filters=...)``).
    bounds_lo, bounds_hi :
        Per-band absolute magnitude limits (same length as ``filters_cat``).
    lib_filter_names :
        ``lib.filter_names`` strings from ``read_cluster`` (defines ``phot_neb_ex`` column order).
    nn_full_filter_order :
        If set, columns passed to the NN scaler use this full band list (e.g. all 5 LEGUS bands);
        missing UV/U (F275W / F336W) bands use ``nn_missing_band_fills`` when provided, else
        scaler training means for those columns.
        Required when ``filters_cat`` is a strict subset but the scaler was trained on more bands.
    """

    def __init__(
        self,
        filters_cat: Sequence[str],
        bounds_lo: Sequence[float],
        bounds_hi: Sequence[float],
        lib_filter_names: Sequence[str],
        nn_scaler_path: str,
        nn_model_path: str,
        *,
        lib_vmag_max: float = -6.0,
        min_bands_nonzero: int = 4,
        nn_batch_rows: int = 65536,
        nn_full_filter_order: Optional[Sequence[str]] = None,
        nn_missing_band_fills: Optional[Dict[str, float]] = None,
    ) -> None:
        self.filters_cat = [str(f) for f in filters_cat]
        self.bounds_lo = [float(x) for x in bounds_lo]
        self.bounds_hi = [float(x) for x in bounds_hi]
        self.lib_filter_names = [str(f) for f in lib_filter_names]
        self.nn_scaler_path = str(nn_scaler_path)
        self.nn_model_path = str(nn_model_path)
        self.lib_vmag_max = float(lib_vmag_max)
        self.min_bands_nonzero = int(min_bands_nonzero)
        self.nn_batch_rows = int(nn_batch_rows)
        self._nn_missing_band_fills: Optional[Dict[str, float]] = (
            dict(nn_missing_band_fills) if nn_missing_band_fills else None
        )

        if len(self.filters_cat) != len(self.bounds_lo) or len(self.filters_cat) != len(
            self.bounds_hi
        ):
            raise ValueError("filters_cat and bounds must have same length")

        self._lib_cols = lib_column_indices(self.lib_filter_names, self.filters_cat)
        self._b_cat_idx = b_band_catalog_index(self.filters_cat)
        self._i_cat_idx = i_band_catalog_index(self.filters_cat)
        self._v_cat_idx = v_band_catalog_index(self.filters_cat)
        if self._v_cat_idx is None:
            raise ValueError(f"Could not resolve V (F555W/F606W) in filters_cat={self.filters_cat!r}")
        self._v_lib_idx = int(self._lib_cols[self._v_cat_idx])
        if self._b_cat_idx is None and self._i_cat_idx is None:
            raise ValueError(
                f"Need at least one of B (F435W/F438W) or I (F814W) in filters_cat; got {self.filters_cat}"
            )

        subset = sorted(self.filters_cat, key=lambda ff: _filt_wave_key(self.lib_filter_names, ff))
        self._subset_filters = subset
        if nn_full_filter_order is None:
            self._nn_full_order = subset
        else:
            full = [str(f) for f in nn_full_filter_order]
            for f in subset:
                if f not in full:
                    raise ValueError(
                        "nn_full_filter_order must list every band in filters_cat; "
                        f"missing {f!r} (filters_cat={subset!r}, nn_full={full!r})"
                    )
            self._nn_full_order = sorted(
                full, key=lambda ff: _filt_wave_key(self.lib_filter_names, ff)
            )
        self._lib_indices_nn = [
            self.lib_filter_names.index(resolve_lib_filter_name(self.lib_filter_names, f))
            for f in subset
        ]

    def compute(
        self,
        phot_neb_ex: np.ndarray,
        dmod: float,
        galaxy_fullname: str = "",
    ) -> Dict[str, Any]:
        """
        Returns dict with ``comp_hybrid``, ``comp_nn``, ``pband``, ``inside_5d``, ``rule_ok``,
        and ``summary`` (:func:`hybrid_libcomp_summary` counts + kept-row completeness stats).
        """
        phot_neb_ex = np.asarray(phot_neb_ex, dtype=float)
        ntot = phot_neb_ex.shape[0]
        inside = inside_5d_abs_box(phot_neb_ex, self._lib_cols, self.bounds_lo, self.bounds_hi)

        comp_nn = np.zeros(ntot, dtype=float)
        idx_in = np.flatnonzero(inside)
        if idx_in.size > 0:
            comp_nn[idx_in] = batched_joint_nn_libcomp(
                phot_neb_ex,
                idx_in,
                self._lib_indices_nn,
                dmod,
                self._subset_filters,
                self._nn_full_order,
                galaxy_fullname,
                self.nn_scaler_path,
                self.nn_model_path,
                self.nn_batch_rows,
                missing_band_fills=self._nn_missing_band_fills,
            )

        pband = hybrid_per_band_hard_flags(
            phot_neb_ex,
            self._lib_cols,
            self.bounds_hi,
            self._v_lib_idx,
            self._v_cat_idx,
            self.lib_vmag_max,
        )
        rule_ok = hybrid_rule_ok_mask(
            pband,
            self._v_cat_idx,
            self._b_cat_idx,
            self._i_cat_idx,
            self.min_bands_nonzero,
        )
        comp_hybrid = comp_nn * rule_ok.astype(np.float64)

        summary = hybrid_libcomp_summary(ntot, inside, rule_ok, comp_nn, comp_hybrid)

        return {
            "comp_hybrid": comp_hybrid,
            "comp_nn": comp_nn,
            "pband": pband,
            "inside_5d": inside,
            "rule_ok": rule_ok,
            "summary": summary,
        }

    @classmethod
    def from_legus_catalog_dict(
        cls,
        data: Dict[str, Any],
        lib_filter_names: Sequence[str],
        nn_scaler_path: str,
        nn_model_path: str,
        *,
        range_margin: float = 0.0,
        lib_vmag_max: float = -6.0,
        min_bands_nonzero: int = 4,
        nn_batch_rows: int = 65536,
    ) -> HybridLegusLibCompletenessCalculator:
        """Build bounds from a ``reader_register['LEGUS'].read(...)`` dict."""
        filters_cat = [str(f) for f in data["filters"]]
        phot = np.asarray(data["phot"], dtype=float)
        detect = np.asarray(data["detect"], dtype=bool)
        lo, hi = legus_catalog_abs_bounds(phot, detect, filters_cat, range_margin=range_margin)
        return cls(
            filters_cat,
            lo,
            hi,
            lib_filter_names,
            nn_scaler_path,
            nn_model_path,
            lib_vmag_max=lib_vmag_max,
            min_bands_nonzero=min_bands_nonzero,
            nn_batch_rows=nn_batch_rows,
        )
