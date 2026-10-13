# QFAMES: error and cost scaling

Numerical experiments for **Quantum Spectral Estimation with Multiple Initial States: Improved Error and Cost Scaling**, by Lingwei Zhang and Zhiyan Ding.

This repository compares independent and shared time sampling in QFAMES using controlled toy Hamiltonians. It covers eigenvalue estimation, multiplicity structure, and observable estimation within a degenerate eigenspace.

The underlying framework is introduced in Zhiyan Ding, Lin Lin, Yilun Yang, and Ruizhe Zhang, *Quantum Filtering and Analysis of Multiplicities in Eigenvalue Spectra*, PRX Quantum **7**, 020318 (2026), [doi:10.1103/jch7-734h](https://doi.org/10.1103/jch7-734h).

## Installation

Use Python 3.10 or later. Run all commands from the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The experiments require NumPy, SciPy, and Matplotlib. No quantum hardware or LaTeX installation is needed.

## Quick check

```bash
python -m unittest discover -s tests -v
python -m experiments.eigenvalue_scaling --quick
python -m experiments.multiplicity_scaling --quick
python -m experiments.observable_scaling --quick
```

Quick runs use small sweeps and four trials per point. Their figures are marked as smoke tests; they check the computation and output pipeline, not asymptotic scaling. To repeat an existing run, add `--resume` or choose a new `--output` directory.

## Notebook

Install the notebook dependencies and launch JupyterLab:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab run_experiments.ipynb
```

Select a Python environment with the dependencies installed, then run the cells in order. The notebook calls the same experiment modules as the command-line interface and displays summary tables and six figures.

- `MODE` selects `"paper"` (JSON configurations) or `"quick"` (smoke tests).
- Each experiment has an action selector: `"run"` computes or resumes, `"plot"` regenerates figures from saved data, and `"view"` displays existing figures. All three currently use `"plot"`. Set all three to `"plot"` to replot without simulation.
- `RESUME = True` reuses completed compatible trials. For changed settings or a fresh run, choose a new `RUN_NAME`.
- `OUTPUTS` selects the experiment directories. Missing data is skipped for `"plot"` and `"view"`; eigenvalue and multiplicity also skip results whose saved configuration differs from the requested settings.

Default full-run outputs are `results/eigenvalue/`, `results/multiplicity/`, and `results/observable/`. Quick runs still use `results/quick/`. Full sweeps can take considerable time and memory. Avoid running the notebook and a terminal job against the same output directory simultaneously.

## Command-line experiments

```bash
python -m experiments.eigenvalue_scaling --workers 2 --resume
python -m experiments.multiplicity_scaling --workers 2 --resume
python -m experiments.observable_scaling --workers 2 --resume
```

These commands compute missing trials or resume compatible saved runs. The default is one worker; use a modest count because each worker holds a spectral data tensor. Completed trials are saved incrementally. Resume requires unchanged settings, source hashes, and NumPy/SciPy versions.

Settings are defined in `configs/eigenvalue.json`, `configs/multiplicity.json`, and `configs/observable.json`. For a separate run:

```bash
python -m experiments.eigenvalue_scaling --config configs/eigenvalue.json --output results/my-run --trials 100 --seed 123
```

| Experiment | Diagnostic | Figures |
| --- | --- | --- |
| Eigenvalue | Errors for the two dominant eigenvalues, with a small spectral tail | Error vs. state count and total evolution time |
| Multiplicity | Mean normalized singular values s1/L through s5/L at estimated peaks targeting 0 and 0.02, with multiplicities 2 and 1 | Separate two-panel figures for independent and shared sampling |
| Observable | Errors for observable eigenvalues 1 and 2 in the multiplicity-two target eigenspace | Error vs. state count and total evolution time |

All sweeps use L=R. Eigenvalue and observable state-count figures use `fixed_N`; each total-time figure compares three N values for both sampling schemes. Multiplicity first searches for two peaks in `search_bounds`, then evaluates SVD at those estimated eigenvalues using the same sampled data. `evaluation_centers` labels the targets and measures peak error; it does not supply the SVD positions. Failed peak searches are counted and excluded from singular-value means.

Each multiplicity panel shows two dashed threshold boundaries with pale shading between them. They lie inside the mean-spectrum admissible interval `mean(s_(m+1)/L) <= tau < mean(s_m/L)`, leaving 15% of the logarithmic gap at each end so neither boundary touches a singular-value curve. The shaded band is an illustrative interior selection, not the full admissible range, a confidence interval, or a per-trial recovery guarantee. A threshold shared by both targets must lie in their full interval intersection at the chosen L. Tau uses normalized singular values (the raw SVD cutoff is L*tau), and is separate from the peak-search `threshold`. Add `--tau 0.1` to the multiplicity replot command, or set `MULTIPLICITY_TAU` in the notebook, to overlay a fixed threshold; red crosses mark values outside the full admissible interval.

The original three experiment runs remain archived under `results/old/{eigenvalue,multiplicity,observable}/`. The subsequent eigenvalue and multiplicity runs with p_tail=0.1 are archived separately under `results/old/2026-10-12-ptail-0.1/`. Their default output directories are empty and ready for fresh runs: keep `RUN_NAME = None` and set the desired experiment actions to `"run"`. Archived results cannot be resumed with changed settings or source code.

## Saved results and replotting

New results use the three experiment directories directly under `results/`; `results/old/` remains ignored by Git. Each completed experiment directory contains:

- `metadata.json`: resolved configuration, seed, software versions, and source hashes.
- `trials.csv`: individual trial results, sampled evolution costs, and diagnostics.
- `summary.csv`: 90th-percentile errors, mean costs, failure counts, and mean normalized singular values as applicable.
- `figures/`: PDF and 300-dpi PNG exports.

Regenerate figures without simulation:

```bash
python -m qfames.plotting results/eigenvalue
python -m qfames.plotting results/multiplicity
python -m qfames.plotting results/observable
```

The command-line plotter reads saved metadata and summaries rather than current configurations. Multiplicity also reads `trials.csv` for the s4/L standard-error bars. Reference lines show prescribed slopes, not fitted regressions. Figure styling is defined in `qfames/plotting.py`.

The archived observable run in `results/old/observable/` retains metadata and CSV files from an earlier code version. It supports replotting, but `--resume` rejects its source-hash mismatch; use a new output directory for fresh simulations.

## Sampling and cost conventions

Times follow a truncated Gaussian density

$$
a_T^{\mathrm{trunc}}(t) \propto \exp\left(-\frac{t^2}{4T^2}\right), \qquad |t| \leq \sigma T.
$$

$T$ is the filter scale; the sampled maximum time is recorded separately. Correlations receive independent real and imaginary Gaussian noise, including when times are shared across state pairs.

Costs sum the absolute sampled evolution times for every state pair, including repeated shared times. Observable costs include spectral-data acquisition and both observable evolution legs. Cost plots use mean sampled cost and the empirical 90th-percentile error at each configuration.

Eigenvalue and multiplicity use a small coherent tail with normalized weight 0.05. Observable uses deterministic initial states and a known target multiplicity. These are controlled Gaussian-noise simulations at fixed T; quick checks and reference slopes do not establish asymptotic scaling.

For eigenvalue estimation at T=200 with the tail at 0.01, the untruncated Gaussian population-filter approximation predicts a lower-peak shift of about 2.0e-5 for p_tail=0.05, compared with 4.2e-5 at 0.1 and 3.7e-7 at the original 0.001. Thus 0.05 reduces tail bias, but can still flatten the large-L independent-sampling curves; it is not expected to fully reproduce the original small-tail accuracy. These are deterministic bias estimates, not new Monte Carlo results.

## Repository layout

```text
qfames/                       # Sampling, estimators, and plotting
experiments/                  # Experiment entry points and trial execution
configs/                      # JSON experiment settings
tests/                        # Numerical and experiment regression checks
results/eigenvalue/           # New eigenvalue results
results/multiplicity/         # New multiplicity results
results/observable/           # New observable results
results/old/                  # Archived previous runs (ignored)
run_experiments.ipynb         # Notebook entry point
requirements.txt              # Core dependencies
requirements-notebook.txt     # Notebook dependencies
```
