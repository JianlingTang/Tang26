# `analyze_catalog_mid_mdd_test_pobs.py`

This script is a test-only copy of `analyze_catalog_mid_mdd.py` for checking
whether the NGC3344 MID mass-function fit is driven by faint-end pobs /
completeness assumptions.

The default behavior is intended to match the production analyzer:

```bash
python analyze_catalog_mid_mdd_test_pobs.py ... --test-pobs-mode none
```

The extra options only activate when `--test-pobs-mode` is set.

## What The Analyzer Does

The analyzer fits cluster population parameters with MCMC using the SLUG
library and LEGUS observed catalog photometry. It builds one `cluster_slug`
object per observed filterset, attaches library observational probabilities
(`pobs` / `libcomp`), and samples the MID or MDD model parameters.

For the current NGC3344 tests, the relevant MID parameters include:

```text
alpha_M      mass function slope
log M_break  exponential mass cutoff
alpha_T      MID age/disruption slope
log T_MID    MID timescale
log p(A_V)   extinction distribution control points
```

## Why This Copy Exists

The production analyzer should remain stable. This copy adds controlled
faint-end pobs experiments without changing the standard pipeline path.

The goal is to test whether the baseline preference for a flatter mass slope
(`alpha_M ~ -1`) instead of the expected power-law value (`alpha_M ~ -2`) is
caused by overestimated faint-end completeness near the V-band detection limit.

## Added Test Modes

### `--test-pobs-mode bright-v-cut`

Keeps only bright observed clusters and removes faint library rows from the
observable model.

```text
observed catalog: keep rows with absolute M_V < --test-v-abs-cut
library pobs: force comp = 0 where absolute M_V > --test-v-abs-cut
```

Default cut:

```text
--test-v-abs-cut -6.5
```

This tests whether clusters close to the faint boundary are driving the fit.

### `--test-pobs-mode shift-completeness-minus1`

Shifts the completeness calculation one magnitude brighter.

Default shift:

```text
--test-completeness-mag-shift -1.0
```

Implementation:

```text
NN apparent-magnitude inputs: m -> m - 1
absolute library magnitudes used by observed-box / hybrid gates: M -> M - 1
```

Example interpretation:

```text
an object that previously entered the NN as apparent magnitude 25
now enters as apparent magnitude 24
```

This approximates a completeness curve shifted to the bright side, so the
effective faint-end pobs is lower at the original observed boundary.

## Current NGC3344 Jobs

The qsub wrappers live in:

```text
pipeline/scripts/nn_mid_mdd_jobs/
```

Bright V cut:

```bash
qsub /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_test_bright_vcut_m65.sh
```

Completeness shift:

```bash
qsub /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_test_pobs_shift_minus1.sh
```

Expected output chains:

```text
output_chains/ngc3344_mid_test_bright_vcut_m65.h5
output_chains/ngc3344_mid_test_pobs_shift_minus1.h5
```

## How To Interpret The Results

Compare each test chain against the baseline observed-box chain:

```text
output_chains/ngc3344_mid_no_hybrid_pobs.h5
```

Useful diagnostics:

```text
best alpha_M
best log M_break
maximum logL
corner plot
observed vs SLUG predicted luminosity plots
```

If either test makes `alpha_M` move significantly closer to `-2`, that supports
the hypothesis that faint-end pobs / completeness is overestimated in the
baseline fit.

If `alpha_M` remains near `-1`, then the flat slope is less likely to be caused
only by the faint-end completeness shape, and may instead be driven by the
NGC3344 luminosity/color distribution, small-number statistics, or another
modeling assumption.
