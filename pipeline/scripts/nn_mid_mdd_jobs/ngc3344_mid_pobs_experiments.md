# NGC3344 MID pobs / completeness experiments

This note documents the four NGC3344 MID jobs used to test whether the fitted
mass-function slope is driven by the faint-end selection function.

## Goal

The baseline expectation for the cluster mass function is a power-law slope near
`alpha_M = -2`. The current NGC3344 MID fits prefer a flatter slope around
`alpha_M ~ -1`. These jobs test whether that preference is caused by the
faint-end completeness / pobs treatment near the V-band limit.

## Jobs

### 1. Baseline observed-box pobs

Script:

```bash
pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid.sh
```

Output chain:

```text
output_chains/ngc3344_mid_no_hybrid_pobs.h5
```

Uses the standard runner with:

```text
--pobs-mode observed-box
```

This is the main comparison baseline. Despite the output name containing
`no_hybrid_pobs`, this is not no-pobs; it is NN pobs multiplied by the observed
absolute-magnitude box.

### 2. Hybrid pobs with M_V = -6

Script:

```bash
pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_pobs_hybrid_mv6.sh
```

Output chain:

```text
output_chains/ngc3344_mid_pobs_hybrid_mv6.h5
```

Uses:

```text
--pobs-mode hybrid
--lib-vmag-max -6.0
```

This tests the legacy hybrid selection: observed box plus per-band gates,
minimum-band rule, B/I rule, and the V-band hard cutoff.

### 3. Bright observed/library V cut at M_V = -6.5

Script:

```bash
pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_test_bright_vcut_m65.sh
```

Output chain:

```text
output_chains/ngc3344_mid_test_bright_vcut_m65.h5
```

Uses the test analyzer:

```text
analyze_catalog_mid_mdd_test_pobs.py
--test-pobs-mode bright-v-cut
--test-v-abs-cut -6.5
--pobs-mode observed-box
```

Behavior:

```text
observed catalog: keep only absolute M_V < -6.5
library pobs: force comp = 0 for library rows with absolute M_V > -6.5
```

This removes clusters near the faint boundary from both the observed catalog and
the observable library.

### 4. Completeness shifted brighter by 1 mag

Script:

```bash
pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_test_pobs_shift_minus1.sh
```

Output chain:

```text
output_chains/ngc3344_mid_test_pobs_shift_minus1.h5
```

Uses the test analyzer:

```text
analyze_catalog_mid_mdd_test_pobs.py
--test-pobs-mode shift-completeness-minus1
--test-completeness-mag-shift -1.0
--pobs-mode observed-box
```

Behavior:

```text
NN apparent-magnitude inputs are shifted by -1 mag: 25 -> 24
absolute library magnitudes used by observed-box gates are also shifted by -1 mag
```

This approximates a completeness curve shifted to the bright side: an object
that previously looked like `M_V = -6` to the completeness model is tested as if
it were `M_V = -7`.

## Interpretation

After the chains finish, compare:

```text
best alpha_M
best log M_break
max logL
luminosity plots
corner plots
```

Expected diagnostic logic:

```text
If alpha_M becomes more negative in the bright-cut or shifted-completeness runs,
then faint-end pobs / completeness overestimate is likely contributing to the
flat alpha_M in the baseline fit.

If alpha_M remains near -1, then the flat slope is less likely to be caused only
by the faint-end pobs shape; it may be driven by the NGC3344 luminosity/color
distribution and small-number statistics.
```
