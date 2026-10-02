"""Rebuild paper figures from saved summaries, without new simulation."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73"]


def finish(fig, path, quick):
    if quick:
        fig.suptitle("Smoke test — not paper results", fontsize=10, color="#777777")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup(ax, xlabel, ylabel):
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, which="major", alpha=0.22)
    ax.tick_params(which="both", direction="in")


def curve(ax, rows, xkey, label, color, style, marker):
    rows = sorted(rows, key=lambda r: r["L"])
    x = np.array([r[xkey] for r in rows])
    y = np.array([r["error_q90"] for r in rows])
    bad = ~np.isfinite(y) | (y <= 0)
    if np.any(bad):
        warnings.warn(f"{label}: {bad.sum()} undefined/nonpositive quantiles; gaps shown in figure")
    y[bad] = np.nan
    ax.plot(x, y, color=color, linestyle=style, marker=marker, markersize=4, label=label)
    return x, y, int(bad.sum())


def guide(ax, x, y, exponent, label):
    good = np.flatnonzero(np.isfinite(y) & (y > 0) & (x > 0))
    if len(good) >= 2:
        k = good[0]
        ax.plot(x, y[k] * (x / x[k]) ** exponent, color="#777777", ls=":", label=label)


def error_plots(name, rows, cfg, directory, quick):
    ylabel = (r"Eigenvalue error $\varepsilon_{0.90}$" if name == "eigenvalue"
              else r"Observable error $\varepsilon_{O,0.90}$")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    setup(ax, r"Initial states per side $L=R$", ylabel)
    missing = 0
    for i, scheme in enumerate(["independent", "shared"]):
        selected = [r for r in rows if r["scheme"] == scheme and r["N"] == cfg["fixed_N"]]
        x, y, count = curve(ax, selected, "L", scheme.capitalize(), COLORS[i],
                            "-" if i == 0 else "--", "o" if i == 0 else "s")
        missing += count
        if i == 0:
            guide(ax, x, y, -1, r"$L^{-1}$ slope guide")
    ax.set_title(f"Fixed N = {cfg['fixed_N']}")
    if missing:
        ax.text(0.02, 0.02, f"{missing} undefined quantiles (gaps); see trials.csv", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=9)
    finish(fig, directory / f"{name}_error_vs_state_count", quick)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    setup(ax, r"Mean sampled total evolution time $T_{\mathrm{total}}$", ylabel)
    missing = 0
    for i, N in enumerate(cfg["N_values"]):
        for scheme in ["independent", "shared"]:
            selected = [r for r in rows if r["scheme"] == scheme and r["N"] == N]
            x, y, count = curve(ax, selected, "mean_total_time", f"{scheme.capitalize()}, N={N}",
                                COLORS[i], "-" if scheme == "independent" else "--",
                                "o" if scheme == "independent" else "s")
            missing += count
            if N == cfg["fixed_N"] and scheme == "independent":
                guide(ax, x, y, -0.5, r"$T_{\mathrm{total}}^{-1/2}$ slope guide")
    if missing:
        ax.text(0.02, 0.02, f"{missing} undefined quantiles (gaps); see trials.csv", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    finish(fig, directory / f"{name}_error_vs_total_time", quick)


def multiplicity_plot(rows, cfg, directory, quick):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    rows = sorted(rows, key=lambda r: r["L"])
    L = np.array([r["L"] for r in rows])
    for cluster, ax in enumerate(axes):
        setup(ax, r"Initial states per side $L=R$", r"Normalized singular value $s_k/L$")
        for k, color in enumerate(COLORS + ["#CC79A7"], 1):
            y = np.array([r[f"sv_{cluster}_{k}"] for r in rows])
            ax.plot(L, y, "o-", markersize=4, color=color, label=rf"$s_{k}/L$")
        # Use the first noise singular value, not an arbitrarily selected tail value.
        noise_index = 3 if cluster == 0 else 2
        y = np.array([r[f"sv_{cluster}_{noise_index}"] for r in rows])
        guide(ax, L, y, -0.5, r"$L^{-1/2}$ slope guide")
        center = np.unique(cfg["eigenvalues"])[cluster]
        ax.set_title(rf"Peak near $\lambda={center:g}$; multiplicity {2 if cluster == 0 else 1}")
        ax.legend(fontsize=8)
    failed = sum(r["failed"] for r in rows)
    if failed:
        axes[0].text(0.02, 0.02, f"{int(failed)} failed trials excluded from SV means", transform=axes[0].transAxes, fontsize=8)
    finish(fig, directory / "multiplicity_singular_values_vs_state_count", quick)


def plot_run(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    with (path / "summary.csv").open(newline="") as stream:
        rows = [{k: v if k == "scheme" else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("No completed results to plot")
    directory = path / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    if meta["experiment"] == "multiplicity":
        multiplicity_plot(rows, meta["config"], directory, meta["quick"])
    else:
        error_plots(meta["experiment"], rows, meta["config"], directory, meta["quick"])
    print(f"Figures saved to {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    plot_run(parser.parse_args().run_directory)
