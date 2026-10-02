# Validation of the repository preparation

Checks run on 2026-10-02 using Python 3.12, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.

- Eight numerical checks passed: shared-cost multiplicity, shared/independent filter broadcasting, the truncated sampler, complex projected observable recovery, two-leg observable cost and phase cancellation, noiseless peak location, failed-trial quantiles, and infeasible-overlap rejection.
- All three quick experiments completed: 72 eigenvalue trials, 12 multiplicity trials, and 72 observable trials. No algorithmic failures occurred in these 156 smoke-test trials.
- All five requested figure types were exported as PDF and PNG. Observable cost and multiplicity figure layouts were visually inspected.
- The multiplicity quick run was repeated with one worker. Its trial CSV was byte-identical to the two-worker output. Resuming the completed run scheduled zero further trials and rebuilt its summary and figure.

The full default experiment sweeps have not been executed. These checks establish numerical identities and pipeline operation, not convergence rates, stable 90th-percentile estimates, or paper-ready experimental conclusions. Smoke-test outputs are deliberately excluded from the source archive.
