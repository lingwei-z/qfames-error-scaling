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
- Each experiment has an action selector: `"run"` computes or resumes, `"plot"` regenerates figures from saved data, and `"view"` displays existing figures. Currently eigenvalue and observable use `"plot"`; multiplicity uses `"run"`. Set all three to `"plot"` to replot without simulation.
- `RESUME = True` reuses completed compatible trials. For changed settings or a fresh run, choose a new `RUN_NAME`.
- `OUTPUTS` selects the experiment directories. Missing data is skipped for `"plot"` and `"view"`; eigenvalue and multiplicity also skip results whose saved configuration differs from the requested settings.

Default full-run outputs are under `results/paper/`. Full sweeps can take considerable time and memory. Avoid running the notebook and a terminal job against the same output directory simultaneously.

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
| Multiplicity | Mean normalized singular values s1/L through s5/L at exact centers 0 and 0.02, with multiplicities 2 and 1 | Separate two-panel figures for independent and shared sampling |
| Observable | Errors for observable eigenvalues 1 and 2 in the multiplicity-two target eigenspace | Error vs. state count and total evolution time |

All sweeps use L=R. Eigenvalue and observable state-count figures use `fixed_N`; each total-time figure compares three N values for both sampling schemes. Multiplicity evaluates known centers without peak search or threshold recovery.

## Saved results and replotting

Tracked results are under `results/paper/`. Each experiment directory contains:

- `metadata.json`: resolved configuration, seed, software versions, and source hashes.
- `trials.csv`: individual trial results, sampled evolution costs, and diagnostics.
- `summary.csv`: 90th-percentile errors, mean costs, failure counts, and mean normalized singular values as applicable.
- `figures/`: PDF and 300-dpi PNG exports.

Regenerate figures without simulation:

```bash
python -m qfames.plotting results/paper/eigenvalue
python -m qfames.plotting results/paper/multiplicity
python -m qfames.plotting results/paper/observable
```

The command-line plotter reads saved metadata and summaries rather than current configurations. Multiplicity also reads `trials.csv` for the s4/L standard-error bars. Reference lines show prescribed slopes, not fitted regressions. Figure styling is defined in `qfames/plotting.py`.

## Sampling and cost conventions

Times follow a truncated Gaussian density proportional to exp(-t²/(4T²)), with |t| ≤ sigma T. T is the filter scale; the sampled maximum time is recorded separately. Correlations receive independent real and imaginary Gaussian noise, including when times are shared across state pairs.

Costs sum the absolute sampled evolution times for every state pair, including repeated shared times. Observable costs include spectral-data acquisition and both observable evolution legs. Cost plots use mean sampled cost and the empirical 90th-percentile error at each configuration.

Eigenvalue and multiplicity use a small coherent tail with normalized weight 0.001. Observable uses deterministic initial states and a known target multiplicity. These are controlled Gaussian-noise simulations at fixed T; quick checks and reference slopes do not establish asymptotic scaling.

## Repository layout

```text
qfames/                       # Sampling, estimators, and plotting
experiments/                  # Experiment entry points and trial execution
configs/                      # JSON experiment settings
tests/                        # Numerical and experiment regression checks
results/paper/                # Saved experiment data and figures
run_experiments.ipynb          # Notebook entry point
requirements.txt              # Core dependencies
requirements-notebook.txt     # Notebook dependencies
```
