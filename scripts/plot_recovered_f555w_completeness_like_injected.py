#!/usr/bin/env python3
"""Plot recovered F555W completeness by reff in the injected-F555W plot style."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V_BAND = "F555W"


def _discover_pairs(cat_dir: Path, outname: str) -> list[tuple[int, float]]:
    pat = re.compile(
        rf"catalogue_frame(\d+)_{re.escape(outname)}_reff([0-9]+\.[0-9]+)\.parquet$"
    )
    pairs: list[tuple[int, float]] = []
    for p in sorted(cat_dir.glob(f"catalogue_frame*_{outname}_reff*.parquet")):
        m = pat.match(p.name)
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))
    return pairs


def _recovered_f555w(phot: pd.DataFrame) -> pd.DataFrame:
    p = phot.copy()
    p["_filter"] = p["filter_name"].astype(str).str.upper()
    v = p[p["_filter"] == V_BAND][["cluster_id", "mag"]].copy()
    v = v.dropna(subset=["cluster_id", "mag"])
    v = v.groupby("cluster_id", as_index=False)["mag"].first()
    return v.rename(columns={"mag": "recovered_f555w"})


def _binned_fraction(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok].astype(float)
    y = y[ok].astype(float)
    rows: list[dict[str, float | int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        n = int(np.sum(m))
        n_rec = int(np.sum(y[m] > 0.5))
        comp = n_rec / n if n else np.nan
        err = np.sqrt(comp * (1.0 - comp) / n) if n else np.nan
        rows.append(
            {
                "mag_center": 0.5 * (float(lo) + float(hi)),
                "mag_lo": float(lo),
                "mag_hi": float(hi),
                "n_recovered_f555w": n,
                "n_in_catalogue": n_rec,
                "completeness": comp,
                "binom_err": err,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--galaxy", required=True)
    ap.add_argument("--outname", required=True)
    ap.add_argument("--root", type=Path, default=Path("/scratch/jh2/jt4478/comp_pipeline_restructure"))
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    gal_dir = root / args.galaxy
    cat_dir = gal_dir / "white" / "catalogue"
    outdir = args.output_dir or (gal_dir / "white" / "diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(cat_dir, args.outname)
    if not pairs:
        raise SystemExit(f"No catalogue_frame parquet found for outname={args.outname!r}")

    reffs = sorted({r for _, r in pairs})
    reff_to_xy: dict[float, tuple[list[np.ndarray], list[np.ndarray]]] = {
        r: ([], []) for r in reffs
    }

    for frame, reff in pairs:
        cpath = cat_dir / f"catalogue_frame{frame}_{args.outname}_reff{reff:.2f}.parquet"
        ppath = cat_dir / f"photometry_frame{frame}_{args.outname}_reff{reff:.2f}.parquet"
        if not (cpath.exists() and ppath.exists()):
            continue

        cat = pd.read_parquet(cpath)
        phot = pd.read_parquet(ppath)
        if "cluster_id" not in cat.columns or "in_catalogue" not in cat.columns:
            continue

        merged = cat.merge(_recovered_f555w(phot), on="cluster_id", how="left")
        found = (
            pd.to_numeric(merged["passes_found"], errors="coerce").fillna(0).to_numpy(dtype=int)
            if "passes_found" in merged.columns
            else np.ones(len(merged), dtype=int)
        )
        x = pd.to_numeric(merged["recovered_f555w"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merged["in_catalogue"], errors="coerce").fillna(0).to_numpy(dtype=float)
        valid = np.isfinite(x) & (found == 1)
        if np.any(valid):
            reff_to_xy[reff][0].append(x[valid])
            reff_to_xy[reff][1].append(y[valid])

    all_x = [
        arr
        for x_chunks, _y_chunks in reff_to_xy.values()
        for arr in x_chunks
        if len(arr)
    ]
    if not all_x:
        raise SystemExit("No finite recovered F555W values found")

    x_all = np.concatenate(all_x)
    vmin = args.vmin if args.vmin is not None else float(np.floor(np.nanmin(x_all) * 2.0) / 2.0)
    vmax = args.vmax if args.vmax is not None else float(np.ceil(np.nanmax(x_all) * 2.0) / 2.0)
    bins = np.arange(vmin, vmax + args.bin_width, args.bin_width)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(reffs))]
    parts: list[pd.DataFrame] = []

    for ri, reff in enumerate(reffs):
        x_chunks, y_chunks = reff_to_xy[reff]
        if not x_chunks:
            continue
        x = np.concatenate(x_chunks)
        y = np.concatenate(y_chunks)
        df = _binned_fraction(x, y, bins)
        df.insert(0, "reff_pc", reff)
        df.insert(0, "x_band", V_BAND)
        parts.append(df)

        sub = df[df["n_recovered_f555w"] > 0]
        if len(sub):
            ax.errorbar(
                sub["mag_center"],
                sub["completeness"] * 100.0,
                yerr=sub["binom_err"] * 100.0,
                fmt="-o",
                ms=3,
                lw=1.2,
                capsize=2,
                color=colors[ri],
                label=f"reff={reff:.0f} pc",
            )

    ax.set_title(f"Completeness vs recovered {V_BAND} (by reff, frames concatenated)")
    ax.set_xlabel(f"Recovered {V_BAND} mag")
    ax.set_ylabel("N in_catalogue / N recovered per bin [%]")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle(f"{args.galaxy} {args.outname}", fontsize=11, y=1.02)
    fig.tight_layout()

    stem = f"{args.outname}_completeness_vs_recovered_f555w_mag_by_reff"
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    csv = outdir / f"{stem}.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(csv, index=False)

    print("Wrote", png)
    print("Wrote", pdf)
    print("Wrote", csv)


if __name__ == "__main__":
    main()
