# What ships inside `new_pipeline_slug_legus`

This package is meant to be **self-contained for code**: you can copy or `tar` this folder to Gadi and run the bundled driver **without** the rest of `slugfiles`, as long as you install Python dependencies and bring **your own** cluster_slug library, LEGUS CCT files, NN weights, and PDFs.

## Bundled Python (flat imports)

| File | Role |
|------|------|
| `bundled_pipeline/analyze_catalog_mid_mdd.py` | CLI: MCMC + NN completeness |
| `bundled_pipeline/catalog_readers.py` | HLSP / LEGUS readers |
| `bundled_pipeline/clean_legus.py` | Cleaning + NN obs. completeness |
| `bundled_pipeline/completeness_io.py` | NN inference (Torch / sklearn / joblib) |
| `bundled_pipeline/completeness_calculator.py` | Legacy completeness classes, `comp_register` |

Re-copy from the main repo when sources change:

```bash
./scripts/sync_pipeline_from_repo.sh
```

## Bundled small data (smoke / examples only)

| Path | Role |
|------|------|
| `fixtures/legus_hlsp/*.tab`, `*.readme` | Tiny HLSP table + readme for local tests |
| `fixtures/legus_hlsp/lib_*.pdf` | Placeholder sampling PDFs for argument parsing demos |
| `fixtures/legus_hlsp/nn_comp_smoke/` | Minimal NN scaler + Torch checkpoint for tests |

These are **not** a full science dataset; they exist so tests can run without your Gadi paths.

## You must provide on the target system

- **Python**: 3.13+ recommended (see `pyproject.toml`).
- **PyPI packages**: at minimum those in `pyproject.toml` (`numpy`, `astropy`, `scipy`, `emcee`, `numexpr`, `torch`, `scikit-learn`, …) plus **`joblib`** if your scaler pickles need it.
- **`slugpy`** with `cluster_slug`, `slug_pdf`, `read_cluster` (not declared in this bundle; install from your site’s wheel or source).
- **Data / models** (typical Gadi layout; override with CLI flags):
  - Cluster slug library (`--cluster-slug-lib-name`, etc.)
  - LEGUS CCT ancillary root (`--legus-cct-root`)
  - NN directory or explicit `--nn-scaler` / `--nn-model`
  - Mass / age / A_V PDFs passed as positional arguments to the script
  - Output directory (`--output-mcmc-chains-dir`)

## Layout summary

```
new_pipeline_slug_legus/
  bundled_pipeline/     # Python copies (PYTHONPATH)
  fixtures/             # Small HLSP + NN smoke files
  scripts/              # run_*.sh, sync_*.sh
  src/legus_slug_pipeline/
  tests/
  docs/
```
