#!/usr/bin/env python
"""Generate convergence diagnostics for an emcee HDFBackend chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MID_LABELS = [
    r"$\alpha_M$",
    r"$\log M_{\mathrm{break}}$",
    r"$\alpha_T$",
    r"$\log (T_{\mathrm{MID}} / \mathrm{yr})$",
]

MDD_LABELS = [
    r"$\alpha_M$",
    r"$\log M_{\mathrm{break}}$",
    r"$\gamma_{\mathrm{MDD}}$",
    r"$\log (T_{\mathrm{MDD,min}} / \mathrm{yr})$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot emcee trace, log-probability, autocorrelation, and posterior diagnostics."
    )
    parser.add_argument(
        "chain",
        nargs="?",
        default="output_chains/ngc628_mid_pobs_hybrid_mv6.h5",
        help="Path to emcee HDFBackend .h5 chain.",
    )
    parser.add_argument(
        "--outdir",
        default="output_io/diagnostics_ngc628_mid_pobs_hybrid_mv6",
        help="Directory for diagnostic plots and summary text.",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=300,
        help="Burn-in samples to discard for posterior summaries.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Thinning factor for flattened posterior samples.",
    )
    parser.add_argument(
        "--model",
        choices=["mid", "mdd"],
        default="mid",
        help="Controls labels for the first four demographic parameters.",
    )
    parser.add_argument(
        "--max-walkers",
        type=int,
        default=30,
        help="Maximum number of walkers to show in trace plots.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=10,
        help="Maximum number of dimensions to plot in trace/autocorr figures.",
    )
    return parser.parse_args()


def running_mean_by_walker(chain: np.ndarray) -> np.ndarray:
    steps = np.arange(1, chain.shape[0] + 1, dtype=float)[:, None, None]
    return np.cumsum(chain, axis=0) / steps


def finite_autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.full(max_lag + 1, np.nan)
    x = x - np.mean(x)
    var = np.dot(x, x)
    if var <= 0:
        return np.full(max_lag + 1, np.nan)
    lags = np.arange(max_lag + 1)
    acf = np.empty(max_lag + 1)
    for lag in lags:
        acf[lag] = np.dot(x[: x.size - lag], x[lag:]) / var
    return acf


def make_labels(ndim: int, model: str) -> list[str]:
    labels = MID_LABELS if model == "mid" else MDD_LABELS
    if ndim <= 4:
        return labels[:ndim]
    return labels + [rf"$\log p(A_V)_{{{i}}}$" for i in range(ndim - 4)]


def plot_traces(chain: np.ndarray, labels: list[str], outpath: Path, max_walkers: int) -> None:
    nstep, nwalker, ndim = chain.shape
    walker_idx = np.linspace(0, nwalker - 1, min(max_walkers, nwalker), dtype=int)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, max(2.0 * ndim, 3.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        ax.plot(chain[:, walker_idx, dim], lw=0.45, alpha=0.45)
        ax.set_ylabel(labels[dim])
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Trace plot ({nstep} steps, showing {len(walker_idx)}/{nwalker} walkers)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_running_means(chain: np.ndarray, labels: list[str], outpath: Path, max_walkers: int) -> None:
    nstep, nwalker, ndim = chain.shape
    walker_idx = np.linspace(0, nwalker - 1, min(max_walkers, nwalker), dtype=int)
    running = running_mean_by_walker(chain[:, walker_idx, :])
    ensemble = np.mean(chain, axis=1)
    ensemble_running = np.cumsum(ensemble, axis=0) / np.arange(1, nstep + 1)[:, None]

    fig, axes = plt.subplots(ndim, 1, figsize=(9, max(2.0 * ndim, 3.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        ax.plot(running[:, :, dim], lw=0.45, alpha=0.25, color="tab:blue")
        ax.plot(ensemble_running[:, dim], lw=1.5, color="black", label="ensemble")
        ax.set_ylabel(labels[dim])
    axes[0].legend(loc="best", frameon=False)
    axes[-1].set_xlabel("step")
    fig.suptitle("Running mean diagnostic")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_log_prob(log_prob: np.ndarray, outpath: Path, max_walkers: int) -> None:
    nstep, nwalker = log_prob.shape
    walker_idx = np.linspace(0, nwalker - 1, min(max_walkers, nwalker), dtype=int)
    q16, q50, q84 = np.nanpercentile(log_prob, [16, 50, 84], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(log_prob[:, walker_idx], lw=0.5, alpha=0.45)
    axes[0].set_ylabel(r"$\log p$")
    axes[0].set_title(f"Walker log probability (showing {len(walker_idx)}/{nwalker})")
    axes[1].fill_between(np.arange(nstep), q16, q84, alpha=0.25)
    axes[1].plot(q50, color="black", lw=1.2)
    axes[1].set_ylabel(r"$\log p$ percentiles")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_acceptance(accepted: np.ndarray, iteration: int, outpath: Path) -> np.ndarray:
    acceptance = accepted / float(iteration)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(acceptance, bins=20, color="tab:blue", alpha=0.8)
    ax.axvline(np.mean(acceptance), color="black", lw=1.5, label=f"mean={np.mean(acceptance):.3f}")
    ax.set_xlabel("acceptance fraction")
    ax.set_ylabel("number of walkers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return acceptance


def plot_autocorr(chain: np.ndarray, labels: list[str], outpath: Path) -> None:
    nstep, _, ndim = chain.shape
    max_lag = min(200, max(1, nstep // 2))
    fig, ax = plt.subplots(figsize=(9, 5))
    for dim in range(ndim):
        acfs = [finite_autocorr_1d(chain[:, walker, dim], max_lag) for walker in range(chain.shape[1])]
        acf = np.nanmean(acfs, axis=0)
        ax.plot(np.arange(max_lag + 1), acf, lw=1.1, label=labels[dim])
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("mean walker autocorrelation")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_posterior_matrix(samples: np.ndarray, labels: list[str], outpath: Path) -> None:
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.1 * ndim, 2.1 * ndim))
    axes = np.asarray(axes)
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row == col:
                ax.hist(samples[:, col], bins=35, color="0.2", histtype="step")
            elif row > col:
                ax.hist2d(samples[:, col], samples[:, row], bins=35, cmap="Blues")
            else:
                ax.axis("off")
                continue
            if row == ndim - 1:
                ax.set_xlabel(labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=9)
            elif col != 0:
                ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_derived_alpha4(samples: np.ndarray, outpath: Path) -> tuple[float, float, float] | None:
    if samples.shape[1] < 2:
        return None
    alpha4 = samples[:, 0] - 4.0 / samples[:, 1]
    q16, q50, q84 = np.percentile(alpha4[np.isfinite(alpha4)], [16, 50, 84])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alpha4, bins=40, color="tab:green", alpha=0.8)
    ax.axvline(q50, color="black", lw=1.4, label=f"median={q50:.3f}")
    ax.axvspan(q16, q84, color="black", alpha=0.12, label="16-84%")
    ax.set_xlabel(r"$\alpha_M - 4 / \log M_{\mathrm{break}}$")
    ax.set_ylabel("posterior samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return q16, q50, q84


def write_summary(
    outpath: Path,
    chain_path: Path,
    chain: np.ndarray,
    log_prob: np.ndarray,
    labels: list[str],
    discard: int,
    samples: np.ndarray,
    tau: np.ndarray | None,
    acceptance: np.ndarray,
    alpha4_summary: tuple[float, float, float] | None,
) -> None:
    nstep, nwalker, ndim = chain.shape
    lines = [
        f"chain: {chain_path}",
        f"shape: steps={nstep}, walkers={nwalker}, ndim={ndim}",
        f"discard: {discard}",
        f"posterior samples after discard/thin: {samples.shape[0]}",
        f"log_prob max: {np.nanmax(log_prob):.6g}",
        f"log_prob final median: {np.nanmedian(log_prob[-1]):.6g}",
        "",
        "acceptance_fraction:",
        f"  mean={np.mean(acceptance):.4f}, min={np.min(acceptance):.4f}, max={np.max(acceptance):.4f}",
    ]
    low_accept = int(np.sum(acceptance < 0.15))
    high_accept = int(np.sum(acceptance > 0.60))
    lines.append(f"  walkers_below_0.15={low_accept}, walkers_above_0.60={high_accept}")

    lines.extend(["", "autocorr_time:"])
    if tau is None:
        lines.append("  unavailable")
    else:
        for label, value in zip(labels, tau):
            ratio = nstep / value if np.isfinite(value) and value > 0 else np.nan
            lines.append(f"  {label}: tau={value:.3f}, steps/tau={ratio:.2f}, recommended_steps_50tau={50*value:.0f}")

    lines.extend(["", "posterior_quantiles_after_discard:"])
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    for label, lo, med, hi in zip(labels, q16, q50, q84):
        lines.append(f"  {label}: {med:.6g} -{med-lo:.3g} +{hi-med:.3g}")

    if alpha4_summary is not None:
        lo, med, hi = alpha4_summary
        lines.extend(
            [
                "",
                "derived_alpha_at_1e4:",
                f"  alpha_M - 4/logM_break = {med:.6g} -{med-lo:.3g} +{hi-med:.3g}",
            ]
        )

    lines.extend(
        [
            "",
            "quick_read:",
            "  Convergence is usually weak if steps/tau is much below 50.",
            "  Acceptance fractions near 0.2-0.5 are often usable for ensemble samplers, but trace stability matters.",
            "  If discard is close to total steps, posterior summaries are based on very little post-burn-in chain.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    chain_path = Path(args.chain)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reader = emcee.backends.HDFBackend(str(chain_path), read_only=True)
    chain = reader.get_chain()
    log_prob = reader.get_log_prob()
    nstep, _, ndim_total = chain.shape
    ndim = min(args.max_dim, ndim_total)
    labels = make_labels(ndim_total, args.model)[:ndim]

    if args.discard >= nstep:
        raise ValueError(f"--discard={args.discard} is >= chain length ({nstep})")

    chain_plot = chain[:, :, :ndim]
    samples = reader.get_chain(discard=args.discard, thin=args.thin, flat=True)[:, :ndim]

    try:
        tau = reader.get_autocorr_time(tol=0)[:ndim]
    except Exception:
        tau = None

    plot_traces(chain_plot, labels, outdir / "trace.png", args.max_walkers)
    plot_running_means(chain_plot, labels, outdir / "running_mean.png", args.max_walkers)
    plot_log_prob(log_prob, outdir / "log_probability.png", args.max_walkers)
    acceptance = plot_acceptance(reader.accepted, reader.iteration, outdir / "acceptance_fraction.png")
    plot_autocorr(chain_plot, labels, outdir / "autocorrelation.png")
    plot_posterior_matrix(samples, labels, outdir / "posterior_matrix.png")
    alpha4_summary = plot_derived_alpha4(samples, outdir / "derived_alpha_at_1e4.png")
    write_summary(
        outdir / "summary.txt",
        chain_path,
        chain,
        log_prob,
        labels,
        args.discard,
        samples,
        tau,
        acceptance,
        alpha4_summary,
    )
    print(f"Wrote diagnostics to {outdir}")
    print((outdir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
