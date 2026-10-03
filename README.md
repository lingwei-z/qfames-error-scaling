# QFAMES: error and cost scaling

Numerical experiments for **Quantum Spectral Estimation with Multiple Initial States: Improved Error and Cost Scaling**, by Lingwei Zhang and Zhiyan Ding.

This repository compares independent and shared time sampling in QFAMES, using controlled toy Hamiltonians. It contains experiments for eigenvalue estimation, multiplicity structure, and observable estimation within a degenerate eigenspace. The code follows the revised experiment plan in the manuscript draft; it is not a reproduction of every figure in version 1.0.

The underlying framework is introduced in Zhiyan Ding, Lin Lin, Yilun Yang, and Ruizhe Zhang, *Quantum Filtering and Analysis of Multiplicities in Eigenvalue Spectra*, PRX Quantum **7**, 020318 (2026), [doi:10.1103/jch7-734h](https://doi.org/10.1103/jch7-734h).

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Only NumPy, SciPy, and Matplotlib are required. No quantum hardware or LaTeX installation is needed. Commands below should be run from the repository root.

## Quick check

```bash
python -m unittest discover -s tests -v
python -m experiments.eigenvalue_scaling --quick
python -m experiments.multiplicity_scaling --quick
python -m experiments.observable_scaling --quick
```

Quick runs use three small state counts, N in {40, 80, 160}, and four trials per point. Their figures are explicitly marked as smoke tests. They check the computation and output pipeline, not asymptotic scaling or publication-quality quantiles.

## Run all experiments in a notebook

The notebook `run_experiments.ipynb` provides a single entry point
for running all three experiments and inspecting their summary
tables and figures.

Install the notebook dependencies:

```bash
python -m pip install -r requirements-notebook.txt
```

Open the notebook in JupyterLab, Jupyter Notebook, or VS Code,
and select a Python environment with the project dependencies installed.
For example, to launch JupyterLab:

```bash
jupyter lab run_experiments.ipynb
```

Run the cells in order, or select **Run All**.

- `MODE = "paper"` uses the full experiment configurations.
- `MODE = "quick"` runs small smoke tests.
- Each experiment cell has an independent selector: `EIGENVALUE_ACTION`,
  `MULTIPLICITY_ACTION`, or `OBSERVABLE_ACTION`. All default to `"plot"`,
  which reads saved summaries and metadata, regenerates figures, and displays
  results without running simulations.
- Set a selector to `"run"` to compute or resume that experiment, or `"view"`
  to display its existing results without regenerating figures.
- Missing saved data is reported and skipped; it does not trigger computation.
  `OUTPUTS` selects each experiment directory independently. Multiplicity defaults
  to `results/paper/multiplicity/`; explicitly select `"run"` for it.
- `RESUME = True` reuses completed trials when settings and versions match.
  To rerun from scratch, set a new `RUN_NAME` (for example `"paper-v2"`)
  and select `"run"` for the desired experiments.

Full experiments can be computationally expensive.
Default paper outputs are saved under `results/paper/`.
Previous multiplicity and observable results are archived in Git-ignored
`results/old/2026-10-02/`; their complete settings and methods are recorded in
[Archived experiment settings](docs/archived-settings-2026-10-02.md).
Avoid running multiple jobs against the same output directory.

The notebook calls the same experiment modules as the command-line
interface. Experimental settings are defined in `configs/`.

## Run the experiments

```bash
python -m experiments.eigenvalue_scaling --workers 2
python -m experiments.multiplicity_scaling --workers 2
python -m experiments.observable_scaling --workers 2
```

The default is one worker. Use a modest worker count: each worker holds an entire spectral data tensor, and the large observable configurations can use over 1 GB per worker. Full runs involve thousands of trials and can be expensive, especially the global peak searches. Run the quick checks first. The scripts print progress, checkpoint each completed trial, and do not overwrite previous runs.

| Experiment | Spectrum | Fixed-N state-count figure | N values in total-time figure | Figures |
| --- | --- | --- | --- | --- |
| Eigenvalue | {0, 0.25} | N = 200 | 50, 200, 800 | Error vs L=R; error vs sampled total time |
| Multiplicity | {0, 0, 0.02} | N = 500 | Not applicable | Separate independent/shared 1x2 singular-value figures at exact centers |
| Observable | {0, 0, 0.1} | N = 500 | 200, 500, 1000 | Error vs L=R; error vs sampled total time |

The eigenvalue and observable total-time figures each contain **six measured curves**: three independent and three shared, with N fixed along each curve and L=R varied. The state-count figure reuses the corresponding fixed-N trials from the same sweep. It compares both sampling schemes.

Edit `configs/eigenvalue.json`, `configs/multiplicity.json`, or `configs/observable.json` to change the settings. For a separate run:

```bash
python -m experiments.eigenvalue_scaling --config configs/eigenvalue.json --output results/my-run --trials 100 --seed 123
```

Resume interrupted computation with the same arguments plus `--resume`:

```bash
python -m experiments.observable_scaling --workers 2 --resume
```

Resuming checks the resolved configuration, source hashes, and NumPy/SciPy versions. Trial seeds depend on experiment, N, L, trial index, and sampling scheme, so changing the worker count does not change numerical results. States are paired between the schemes at each (N,L,trial); time and noise streams are separate. Changing chunk size changes the observable RNG draw order and therefore requires a separate run.

## Replot without simulation

```bash
python -m qfames.plotting results/paper/eigenvalue
python -m qfames.plotting results/paper/multiplicity
python -m qfames.plotting results/paper/observable
```

Equivalently, add `--plot-only` to an experiment command. This reads the saved metadata and summary, not the current configuration file. Each figure is exported as both vector PDF and 300-dpi PNG. Reference lines use a prescribed slope, with log-mean vertical placement; they are not slope regressions. Multiplicity singular-value guides appear only in the independent figure and use its s4/L points with L >= 24 in each panel; guides are skipped if fewer than two such points exist. The shared multiplicity figure has no reference guide. Each reference label shows a matching dotted line sample after `ref.`. All data points remain visible. Eigenvalue and observable canvases are 3.35 by 3.25 inches with identical axes bounds, 9 pt ticks/legends and 10 pt labels; PDF exports preserve the physical canvas and vector content. Cost-plot color denotes N; solid circles denote independent, dashed open squares denote shared. Cost coordinates remain trial means.

Output names are:

```text
results/paper/eigenvalue/figures/eigenvalue_error_vs_state_count.{pdf,png}
results/paper/eigenvalue/figures/eigenvalue_error_vs_total_time.{pdf,png}
results/paper/multiplicity/figures/multiplicity_singular_values_vs_state_count_independent.{pdf,png}
results/paper/multiplicity/figures/multiplicity_singular_values_vs_state_count_shared.{pdf,png}
results/paper/observable/figures/observable_error_vs_state_count.{pdf,png}
results/paper/observable/figures/observable_error_vs_total_time.{pdf,png}
```

Each experiment directory also contains:

- `metadata.json`: resolved settings, seed, software versions, and source hashes.
- `trials.csv`: individual errors, estimated peaks, actual evolution costs, failure status, and relevant numerical diagnostics. For multiplicity, `sv_<cluster>_<k>` stores **s_k/L evaluated at the exact cluster centers**, not s_k. `evaluation_mode=exact_cluster_centers` and `evaluation_theta_0=0`, `evaluation_theta_1=0.02` identify the evaluation locations in trials and summaries; metadata also records these locations. Legacy `peak_0` and `peak_1` stay empty; `error`, `peak_error`, and summary `error_q90` are NaN (not applicable).
- `summary.csv`: 90th-percentile errors, mean sampled costs, failure counts, and mean normalized singular values.

Time tensors are not written to disk: their realized absolute sums and maxima are recorded per trial, and the seed plus configuration and code identify the generating computation. Keeping every L x R x N tensor would create unnecessarily large data files. Temporary runs are ignored by Git. Paper results under results/paper/ are tracked together with the corresponding code and configurations.

## Time distributions and resource accounting

Spectral data use the truncated density

$$a_T^{\mathrm{trunc}}(t) \propto e^{-t^2/(4T^2)}, \qquad |t|\leq\sigma T.$$

Thus the untruncated standard deviation is sqrt(2)T and the ideal spectral filter is exp[-T²(theta-lambda)²]. **T is a filter scale, not the maximum evolution time.** The sampled maximum is recorded separately.

For eigenvalue and multiplicity experiments, every trial computes

$$T_{\mathrm{spectral}}=\sum_{l,r,n}|t_{l,r,n}|.$$

For shared sampling this is LR times the sum of the absolute sampled times. Reusing a time value does not eliminate the evolution for each state pair. No plotted cost is replaced by LRNT, and signed times are never summed as a cost.

The observable numerator uses independent pairs (t,t') at scale T/sqrt(2), with each leg truncated at sigma*T/sqrt(2). The shared scheme shares each pair across state pairs, while measurement noise remains independent. Its complete cost is

$$T_{\mathrm{total}}=T_{\mathrm{spectral}}+\sum_{l,r,n}(|t_{l,r,n}|+|t'_{l,r,n}|).$$

The first term includes peak localization and the denominator matrix; the second includes both observable evolution legs. The code records both contributions separately. `max_time` is the maximum spectral |t| or observable |t|+|t'| in the trial.

Costs use the manuscript's unit-per-complex-datum convention. A real/imaginary circuit repetition factor, shot count, or observable block-encoding overhead is not added to this Gaussian-noise simulation. For literal hardware resource counts, those overheads must be specified separately. The plotted horizontal coordinate is the arithmetic mean of the actual costs at a fixed (scheme,N,L); the vertical coordinate is the empirical 90th-percentile error at that same configuration.

## Numerical choices

- **Noise:** exact correlations plus independent real and imaginary Gaussian noise, each with the configured standard deviation. This is a surrogate model, not bounded single-shot Hadamard outcomes and not a direct verification of every theorem assumption.
- **Initial states:** experiment 1 repeats a fixed equal-weight vector to keep spectral weights fixed across L and trials. Experiment 2 uses independent normalized complex Gaussian vectors with every component magnitude greater than 0.5. A feasibility check and attempt limit prevent impossible rejection loops.
- **Truncation:** experiments 1 and 3 use sigma = 6; experiment 2 restores sigma = 4. The omitted Gaussian mass decreases from erfc(2), about 4.7e-3, to erfc(3), about 2.2e-5. The truncated kernel is still not exactly the ideal Gaussian kernel.
- **Peak search:** experiment 1 search the declared spectral interval using the Frobenius norm, then refine locally. The grid step is 0.4/T, equivalent to the original code's 20q/T at q=0.02; there is no hidden factor of 20. The search intervals and number of distinct clusters are supplied for these toy problems. True eigenvalues are used for error evaluation, not as evaluation locations. This is continuous refinement of the grid algorithm, not an exact implementation of a finite-grid theorem.
- **Multiplicity:** explicitly configured evaluation_centers=[0.0,0.02], with target multiplicities 2 and 1 and no peak search or threshold recovery. Both independent and shared schemes use spectrum [0.0,0.0,0.02,0.10], T=50, sigma=4, N=500, L=R in [5,8,12,16,24,32,50,70,100], noise_std=0.1, min_overlap=0.4, 50 trials and seed=42. Initial states retain the random complex Gaussian, normalization, and minimum-overlap rejection procedure; their dimension is len(eigenvalues), now 4. The seed rule is unchanged: state seed [seed,2,L,N,trial,0], sampling/noise seed ending in 1 (independent) or 2 (shared). Noise remains independent across matrix entries even with shared times. Two separate 1x2 figures show independent and shared sampling respectively; each panel shows that scheme's four trial-mean normalized singular values s1/L through s4/L. The added eigenvalue 0.10 contributes to sampling but creates no third evaluation center or panel. The setting aims to preserve the middle singular values' deterministic residual plateaus while introducing shared-time fluctuation through the fourth spectral direction, so shared s4/L may approach a plateau at sufficiently large L while independent s4/L continues to decrease. This is an intended behavior to investigate, not a verified result; no trend is imposed in plotting or data processing. Existing saved results retain their original metadata and have not been regenerated with this setting.
- **Observable:** uses deterministic initial states with fixed cross-cluster correlation: each row is `[1, exp(2*pi*j*i/n), 1]/sqrt(3)`. Each row has unit norm; the first two columns have Gram matrix `(n/3)I`, so the target-cluster frame has condition number 1; the third column equals the first. Both sampling schemes use exactly the same frame for a given L,R, while their time-sharing patterns differ. The target multiplicity 2 is a known input. Peak localization reads `search_bounds` from the configuration (default [-0.04, 0.04]), maximizing the Frobenius norm with bounded `minimize_scalar` and xatol=1e-10. There is no global scan or second-peak requirement. The estimated theta_star (saved as `peak_0`) is used in both denominator and observable numerator; `peak_1` is unused, and `peak_error` measures the target location error. Optimizer success, evaluation count, and message are retained in the trial CSV; failures remain in error quantiles. This is an observable error scaling experiment, not end-to-end recovery of a completely unknown spectrum. The spectrum is {0,0,0.1}, the block-supported observable is [[1.5,0.5,0],[0.5,1.5,0],[0,0,0]], and targets are {1,2}. T=50, sigma=6, and both noise standard deviations are 0.015. N is 200, 500, or 1000, with fixed_N=500 and 100 trials per point for the formal run. A separate 3-trial check precedes the formal run; seed 42 and all data points are retained. Outputs use the formal `observable/` directory; previous results are archived under `results/old/2026-10-02/observable/`. Local search reduces classical computation only: costs still include all sampled spectral and two-leg observable evolution times, with LR weighting for shared times.
- **Generalized eigenvalues:** the projected problem is solved directly, without an explicit inverse. Errors use bottleneck matching in the complex plane, retaining imaginary parts rather than silently dropping them. `max_imaginary` and `projected_condition` are saved for diagnosis. This can produce larger reported errors than the old real-part-only metric.
- **Failures:** failed peak searches or singular projected denominators receive infinite error. Quantiles use NumPy's `inverted_cdf` definition and include failures. Infinite quantiles appear as gaps with a warning, not fabricated finite points. Multiplicity singular-value means include all trials and do not depend on peak-search success. A peak-search failure stops before observable acquisition, so its recorded time is the actual partial cost.
- **Reference lines:** dotted curves show prescribed reference slopes, not fitted convergence rates. Their vertical positions use the log-mean offset of valid independent values; multiplicity restricts this placement and line to L >= 24. Only the independent singular-value figure shows the L^(-1/2) guide; the shared figure has no guide. A matching dotted line sample follows each `ref.` label. All measured points remain plotted.

At fixed T, varying L changes the statistical budget. The cost guide is therefore T_total^(-1/2), **not** a numerical test of Heisenberg scaling obtained by varying T. Shared-sampling error need not be flat for every finite-size toy model: independent measurement noise still averages across entries, and the block-supported observable may suppress some shared-time fluctuations. The code does not force saturation or curve collapse. Publication claims should be based on the full runs and their diagnostics.

## Repository layout

```text
qfames/
  core.py                         # Sampling, filters, peak search, observable estimator
  plotting.py                     # All figure generation
experiments/
  common.py                       # Trial execution, seeds, checkpointing, summaries
  eigenvalue_scaling.py            # Experiment 1
  multiplicity_scaling.py          # Experiment 2
  observable_scaling.py            # Experiment 3
configs/                          # Editable experiment settings
tests/test_numerics.py             # Numerical identity and cost checks
tests/test_multiplicity.py         # Exact-center evaluation and plot regression checks
tests/test_observable.py           # Observable state, peak, and trial regression checks
requirements.txt
```

This refactors `QFAMES.py`, `Experiment_3.py`, `Experiment_4.py`, and `Experiment_5.py`. Obsolete N-only, tail, observable-norm, and first multiplicity panels are omitted from the current experiment suite. No unpublished manuscript PDF or inherited plot is bundled.

## Citation and release

Use the manuscript's final bibliographic entry when available. The title and authors above identify the associated work; no publication venue or preprint identifier is assumed here. Before the public release, select a code license with the coauthor and add its LICENSE file. This bundle intentionally does not assign a license on the authors' behalf.

## Regression tests

The experiment-specific tests complement the numerical checks in `tests/test_numerics.py`:

- `tests/test_multiplicity.py` checks exact-center evaluation, paired states, distinct sampling streams, shared noiseless rank, scheme-separated plotting, independent-only tail reference placement, and absence of shared reference guides.
- `tests/test_observable.py` checks deterministic initial-state properties, shared and independent time-sampling shapes and costs, local peak localization, reuse of the same states across schemes, use of the estimated peak in both the denominator and numerator, and preservation of spectral cost when localization fails.

Run all tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

To run either experiment-specific file separately:

```bash
python -m unittest discover -s tests -p "test_multiplicity.py" -v
python -m unittest discover -s tests -p "test_observable.py" -v
```

These small regression checks do not run the full experiment sweeps or validate asymptotic scaling.
