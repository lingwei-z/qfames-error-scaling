"""Rebuild paper figures from saved summaries, without new simulation."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullFormatter
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
SCHEMES = ("independent", "shared")
PAPER_SIZE = (3.35, 3.25)


def style(scheme):
    return dict(linestyle="-" if scheme == "independent" else "--",
                marker="o" if scheme == "independent" else "s",
                markerfacecolor=None if scheme == "independent" else "none",
                markersize=4.5, linewidth=1.5, markeredgewidth=1.0)


def scheme_handles(schemes=SCHEMES):
    return [Line2D([], [], color="#333333", label=s.capitalize(), **style(s)) for s in schemes]


def legend(fig, handles, y, ncol, x=.55):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x, y), ncol=ncol,
               columnspacing=.8, handlelength=1.6, handletextpad=.4, frameon=False, fontsize=9)


def finish(fig, path, quick):
    # Short log ranges otherwise acquire long, densely packed minor labels.
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            axis.set_minor_formatter(NullFormatter())
            if limits[1] / limits[0] < 10:
                axis.set_major_locator(MaxNLocator(nbins=3, integer=axis is ax.xaxis))
                axis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.3g}"))
    if quick:
        fig.text(.5, .01, "Smoke test: not paper results", ha="center", va="bottom", fontsize=9, color="#777777")
    # Keep the physical publication size; tight bounding boxes change the canvas.
    for extension in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=300)
    plt.close(fig)


def setup(ax, xlabel, ylabel):
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, which="major", alpha=0.22)
    ax.tick_params(which="both", direction="in", labelsize=9)


def single_panel():
    fig, ax = plt.subplots(figsize=PAPER_SIZE)
    fig.subplots_adjust(left=.22, right=.97, bottom=.19, top=.73)
    return fig, ax


def curve(ax, rows, xkey, label, color, scheme, ykey="error_q90"):
    rows = sorted(rows, key=lambda r: r["L"])
    x = np.array([r[xkey] for r in rows])
    y = np.array([r[ykey] for r in rows])
    bad = ~np.isfinite(y) | (y <= 0)
    if np.any(bad):
        warnings.warn(f"{label}: {bad.sum()} undefined/nonpositive values; gaps shown")
    y[bad] = np.nan
    ax.plot(x, y, color=color, label=label, **style(scheme))
    return x, y


def guide(ax, x, y, exponent, label):
    """Prescribed reference slope; log-mean vertical placement, no slope regression."""
    good = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x > 0)
    if np.count_nonzero(good) >= 2:
        amplitude = np.mean(np.log(y[good]) - exponent * np.log(x[good]))
        xx = np.sort(x[good])
        reference, = ax.plot(xx, np.exp(amplitude + exponent * np.log(xx)),
                             color="#777777", ls=":", lw=1.5)
        ax.legend(handles=[reference], labels=[label + " ref."], markerfirst=False,
                  loc="lower left", bbox_to_anchor=(.03, .06), borderaxespad=0,
                  handlelength=2.4, handletextpad=.5, fontsize=9,
                  facecolor="white", edgecolor="none", framealpha=.8, borderpad=.2)


def error_plots(name, rows, cfg, directory, quick):
    ylabel = (r"Eigenvalue error $\varepsilon_{0.90}$" if name == "eigenvalue"
              else r"Observable error $\varepsilon_{O,0.90}$")
    fig, ax = single_panel()
    setup(ax, r"Initial states per side $L=R$", ylabel)
    for i, scheme in enumerate(SCHEMES):
        selected = [r for r in rows if r["scheme"] == scheme and r["N"] == cfg["fixed_N"]]
        x, y = curve(ax, selected, "L", scheme.capitalize(), COLORS[i], scheme)
        if scheme == "independent":
            guide(ax, x, y, -1, r"$L^{-1}$")
    ax.set_title(f"Fixed N = {cfg['fixed_N']}", fontsize=10, pad=7)
    legend(fig, ax.get_legend_handles_labels()[0], .96, 1)
    finish(fig, directory / f"{name}_error_vs_state_count", quick)

    fig, ax = single_panel()
    setup(ax, r"Total evolution time $T_{\mathrm{total}}$", ylabel)
    for i, N in enumerate(cfg["N_values"]):
        for scheme in SCHEMES:
            selected = [r for r in rows if r["scheme"] == scheme and r["N"] == N]
            x, y = curve(ax, selected, "mean_total_time", f"{scheme}, N={N}", COLORS[i], scheme)
            if N == cfg["fixed_N"] and scheme == "independent":
                guide(ax, x, y, -.5, r"$T_{\mathrm{total}}^{-1/2}$")
    handles = [Line2D([], [], color=COLORS[i], lw=2, label=rf"$N={N}$") for i, N in enumerate(cfg["N_values"])]
    legend(fig, handles, .995, 3, x=.52)
    legend(fig, scheme_handles(), .90, 2)
    finish(fig, directory / f"{name}_error_vs_total_time", quick)


def threshold_range(ax, x, lower, upper, tau=None):
    """Show an inset illustrative tau band, not the full admissible interval.

    Leave 15% of the logarithmic gap at either singular-value boundary.
    Bounds come from mean spectra, not confidence or per-trial guarantees.
    """
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (upper > lower)
    inset_lower = np.full_like(lower, np.nan, dtype=float)
    inset_upper = np.full_like(upper, np.nan, dtype=float)
    gap = np.log(upper[valid]) - np.log(lower[valid])
    inset_lower[valid] = np.exp(np.log(lower[valid]) + .15 * gap)
    inset_upper[valid] = np.exp(np.log(upper[valid]) - .15 * gap)
    color = "#52687A"
    boundary, = ax.plot(x, inset_lower, color=color, ls="--", lw=1.1,
                        label=r"Illustrative $\tau$ band", zorder=3)
    ax.plot(x, inset_upper, color=color, ls="--", lw=1.1, zorder=3)
    ax.fill_between(x, inset_lower, inset_upper, where=valid,
                    color=color, alpha=.10, linewidth=0, zorder=1)
    handles = [boundary]
    if tau is not None:
        chosen = np.full_like(inset_lower, tau)
        fixed, = ax.plot(x, chosen, color="black", ls="-.", lw=1,
                         label=rf"$\tau={tau:g}$", zorder=4)
        handles.append(fixed)
        # A value outside the inset band can still be admissible: test the full gap.
        outside = valid & ((tau < lower) | (tau >= upper))
        ax.plot(x[outside], chosen[outside], ls="none", marker="x", color="red", markersize=6,
                zorder=5)
    return handles


def multiplicity_plot(rows, cfg, directory, quick, tau=None):
    """One identical-layout 1x2 figure per scheme; never mix scheme rows."""
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError("tau must be finite and positive")
    for scheme in SCHEMES:
        selected = sorted([r for r in rows if r.get("scheme", "independent") == scheme],
                          key=lambda r: r["L"])
        if not selected:
            warnings.warn(f"No {scheme} multiplicity data; skipping its singular-value figure.")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True)
        fig.subplots_adjust(left=.105, right=.98, bottom=.19, top=.74, wspace=.13)
        for cluster, ax in enumerate(axes):
            setup(ax, r"Initial states per side $L=R$", r"Normalized singular value $s_k/L$" if cluster == 0 else "")
            ax.set_ylim(2e-3, .4)
            for k, color in enumerate(COLORS, 1):
                x, y = curve(ax, selected, "L", f"s{k}, {scheme}", color, scheme, f"sv_{cluster}_{k}")
                if k == 4:
                    sem = np.array([r.get(f"sv_{cluster}_{k}_sem", np.nan) for r in selected])
                    valid = np.isfinite(y) & np.isfinite(sem)
                    if np.any(valid):
                        ax.errorbar(x[valid], y[valid], yerr=sem[valid], fmt="none",
                                    ecolor=color, elinewidth=.7, capsize=1.5, capthick=.7)
                if k == 4 and scheme == "independent":
                    guide(ax, x[x >= 24], y[x >= 24], -.5, r"$L^{-1/2}$")
            # Older saved runs predate explicit evaluation centers.
            center = cfg.get("evaluation_centers", np.unique(cfg["eigenvalues"]))[cluster]
            multiplicity = np.count_nonzero(np.asarray(cfg["eigenvalues"]) == center)
            lower = np.array([r[f"sv_{cluster}_{multiplicity + 1}"] for r in selected])
            upper = np.array([r[f"sv_{cluster}_{multiplicity}"] for r in selected])
            tau_handles = threshold_range(ax, x, lower, upper, tau)
            ax.set_title(rf"Target ${center:g}$; multiplicity {multiplicity}", fontsize=10)
        handles = [Line2D([], [], color=c, lw=2, label=rf"$s_{k}/L$") for k, c in enumerate(COLORS, 1)]
        legend(fig, handles, 1.0, 5, x=.53)
        legend(fig, scheme_handles([scheme]) + tau_handles, .91, 1 + len(tau_handles), x=.53)
        finish(fig, directory / f"multiplicity_singular_values_vs_state_count_{scheme}", quick)


def multiplicity_standard_errors(path, rows):
    """Attach sample SD / sqrt(n) from saved trials; preserve summary means."""
    fields = [f"sv_{cluster}_{k}" for cluster in range(2) for k in range(1, 6)]
    groups = {}
    with (path / "trials.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError("Saved multiplicity trials lack five singular values; rerun in a new output directory.")
        for trial in reader:
            key = (trial["scheme"], float(trial["N"]), float(trial["L"]))
            groups.setdefault(key, []).append([float(trial[field]) if trial[field] else np.nan for field in fields])
    for row in rows:
        key = (row["scheme"], row["N"], row["L"])
        values = np.asarray(groups.get(key, []), dtype=float)
        if values.size:
            values = values[np.all(np.isfinite(values), axis=1)]
        for i, field in enumerate(fields):
            # Do not silently drop invalid trials or imply zero uncertainty for n=1.
            row[field + "_sem"] = (np.std(values[:, i], ddof=1) / np.sqrt(len(values))
                                   if len(values) >= 2 and np.all(np.isfinite(values[:, i]))
                                   else np.nan)


def read_run(path):
    meta = json.loads((path / "metadata.json").read_text())
    with (path / "summary.csv").open(newline="") as stream:
        rows = [{k: v if k in {"scheme", "evaluation_mode"} else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    return meta, rows


def plot_run(path, tau=None):
    path = Path(path)
    meta, rows = read_run(path)
    if not rows:
        raise ValueError("No completed results to plot")
    directory = path / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    if meta["experiment"] == "multiplicity":
        if meta.get("evaluation_mode") not in {"exact_cluster_centers", "estimated_cluster_centers"}:
            raise ValueError("Unsupported multiplicity evaluation mode; use a new run directory.")
        if any(f"sv_{c}_5" not in row for row in rows for c in range(2)):
            raise ValueError("Saved multiplicity summary lacks s5/L; rerun in a new output directory.")
        multiplicity_standard_errors(path, rows)
        multiplicity_plot(rows, meta["config"], directory, meta["quick"], tau=tau)
    else:
        error_plots(meta["experiment"], rows, meta["config"], directory, meta["quick"])
    print(f"Figures saved to {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--tau", type=float, help="Normalized singular-value threshold guide (multiplicity only)")
    args = parser.parse_args()
    plot_run(args.run_directory, tau=args.tau)
