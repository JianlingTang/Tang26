# Architecture

## Packaging model

- `**bundled_pipeline/**` holds a **verbatim copy** of the five Python modules that implement the mid-MDD LEGUS + NN completeness driver. They use the same flat imports as `python3_sc/` (`from catalog_readers import …`). At runtime, `**PYTHONPATH` must include `bundled_pipeline/`** (the `scripts/run_analyze_catalog_mid_mdd.sh` helper does this).
- `**fixtures/**` holds **small** HLSP + NN files so pytest can run **without** Gadi paths or a full science dataset.
- `**src/legus_slug_pipeline/`** is a thin **configuration** layer (`PipelinePaths`, `apply_env()`), not a rewrite of the science code.

## Data flow (science)

1. **Catalog ingest** — `catalog_readers.reader_register` reads HLSP tables; ancillary paths use `LEGUS_CCT_ROOT` / `LEGUS_TAB_DIR` when set by the CLI or `PipelinePaths.apply_env()`.
2. **Observation cleaning** — `clean_legus.clean_legus` applies NN-based completeness and duplicate handling.
3. **Library load** — `slugpy.cluster_slug.read_cluster` uses `--cluster-slug-lib-name`.
4. **Library completeness** — `completeness_io.predict_catalog_completeness_with_nn` scores library photometry with the same NN bundle as observations.
5. **MCMC** — `emcee` writes chains under `--output-mcmc-chains-dir` (relative `-o` is resolved there).

## Updating code

After editing sources under the main repo’s `python3_sc/`, run `**scripts/sync_pipeline_from_repo.sh`** so the tarball stays current.