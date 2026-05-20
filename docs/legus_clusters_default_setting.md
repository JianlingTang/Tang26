# LEGUS Clusters Default Setting

This note describes the default observed-catalog and NN/hybrid completeness
settings used by the two NGC3344 jobs submitted on 2026-05-19:

- `pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mid_newread_maxuvfill_priorcache_hybrid.sh`
  - PBS job: `168766619.gadi-pbs`
- `pipeline/scripts/nn_mid_mdd_jobs/ngc3344_mdd_newread_maxuvfill_priorcache_hybrid.sh`
  - PBS job: `168767102.gadi-pbs`

Both jobs call:

```bash
/g/data/jh2/jt4478/Tang26B/pipeline/bundled_pipeline/analyze_mid_mdd_new_read.py
```

The MID and MDD jobs use the same catalog reader and observational
completeness definition. The only model difference is that the MDD job passes
`--mdd`; the MID job does not.

## Shared Run Settings

- Galaxy: `ngc3344`
- Catalog discovery root: `/g/data/jh2/jt4478/make_LEGUS_CCT`
- Catalog glob: `hlsp_legus*{galaxy_name}*.tab`
- Library: `/g/data/jh2/jt4478/cluster_slug/tang_padova`
- NN directory: `/g/data/jh2/jt4478/Tang26B/nn_models`
- pobs mode: `hybrid`
- Library V-band limit: `--lib-vmag-max -6.0`
- Hybrid observed-box margin: `--hybrid-range-margin 0.05`
- Completeness threshold: `--comp-threshold 1e-12`
- MCMC: `--nwalkers 100`, `--niter 5000`, `--nprocs 52`

Outputs:

- MID chain: `output_chains/ngc3344_mid_newread_maxuvfill_priorcache_tangnn_hybrid.h5`
- MDD chain: `output_chains/ngc3344_mdd_newread_maxuvfill_priorcache_tangnn_hybrid.h5`

## Catalog Reader Criteria

The jobs use the new LEGUS HLSP reader in
`analyze_mid_mdd_new_read.py`, not the old reader that rejected
`photerr > 0.3` detections.

Per-filter detection:

- A filter is treated as detected when the magnitude is finite, the error is
  finite, and the magnitude is not one of the sentinel values
  `44.444`, `66.666`, or `99.999`.
- Large finite photometric errors are retained. There is no `photerr <= 0.3`
  detection cut.
- Apparent magnitudes are converted to absolute magnitudes with
  `M = m - dmod`.

Observed cluster selection:

- LEGUS class satisfies `0 < class < 3.5`.
- A V-like band is detected. The reader checks `F555W` first, then `F606W`.
- Absolute V magnitude satisfies `M_V < -6.0` in the reader.
- At least 4 filters are detected.
- At least one B or I band is detected:
  - B: `F435W` or `F438W`
  - I: `F814W`

After reading, clusters are grouped by their exact detected-filter set. The
cleaning step removes:

- clusters with observed NN completeness below `1e-12`;
- duplicate clusters across catalogs, matched by exact RA/Dec.

The submitted jobs pass `--disable-hybrid-clean-criteria`, so no additional
hybrid hard-mask cleaning is applied to the observed catalog after the reader.
The reader itself already applies the V detection, `M_V`, minimum-band, and
B/I requirements above.

## Missing UV/U Fill For NN Inputs

The NN models can expect the full LEGUS filter order even when an observed
cluster/filterset lacks UV or U. For missing `F275W` or `F336W` NN inputs, the
new script fills the missing value with the faint end of the observed catalog:

```text
fill_mag = max(detected apparent magnitude in that band) + 0.5
```

This is the `maxuvfill` behavior. It treats a missing UV/U detection as a faint
UV/U constraint, rather than accidentally filling it with a bright value.

## NN And Hybrid Library Completeness

For each observed catalog filterset, the code computes library observability as
a hybrid product:

```text
p_obs = p_NN * hard_selection_mask
```

The NN part:

- Uses the galaxy-specific NN scaler/model from `nn_models`.
- Uses library `phot_neb_ex` absolute magnitudes shifted to apparent
  magnitudes with the catalog distance modulus before NN prediction.
- Uses the filterset-specific NN input order sorted by wavelength.
- If the filterset is missing UV/U, the same max-magnitude UV/U fill described
  above is used.

The hard hybrid mask:

- Builds an observed absolute-magnitude box for the selected filterset using
  observed catalog detections in those bands.
- Adds a `0.05` mag margin to the observed box.
- Sets NN completeness to zero outside this observed box.
- Requires the library cluster to be no fainter than the observed faint limit
  in each selected band.
- Forces the V-band flag to zero if the library has `M_V > -6.0`; equivalently,
  the library passes the V cut only when `M_V <= -6.0`.
- Requires V to pass.
- Requires B or I to pass.
- Requires at least 4 bands to pass, capped by the number of bands in the
  filterset.

The final hybrid completeness is:

```text
comp_hybrid = comp_nn * rule_ok
```

Library rows are then pruned if they have `comp_hybrid < 1e-12` for every
catalog/filterset.

## Why Hybrid Is Needed Even With NN

The NN models the smooth recovery probability. It does not by itself guarantee
that the library support obeys the same deterministic catalog and artificial
cluster recovery boundaries as the observed sample.

The hard hybrid gates keep the likelihood consistent with the data definition:

- the observed catalog has a hard V-band sample limit near `M_V = -6`;
- artificial-cluster recovery includes magnitude-consistency criteria to avoid
  counting injected clusters that were blended with or confused with real
  pre-existing clusters;
- these rules create sharp faint-end structure, especially in V band, that a
  pure NN completeness can smooth over.

Therefore the default setting uses NN completeness for the smooth recovery
probability, and explicit hybrid cuts for the deterministic sample/recovery
boundaries.
