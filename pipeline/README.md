# LEGUS cluster_slug pipeline (Gadi-friendly bundle)

This directory is a **self-contained tarball layout**: it includes a **copy** of the Python modules needed to run `analyze_catalog_mid_mdd.py`, small **fixture** data for smoke tests, and **uv** metadata. You do **not** need the rest of `slugfiles` on the machine where you run—only this tree plus your own `slugpy`, cluster_slug library, LEGUS/CCT files, NN weights, and PDF inputs.

## Layout

| Path | Role |
|------|------|
| `bundled_pipeline/` | Copied `python3_sc` modules (flat imports; set `PYTHONPATH` here) |
| `fixtures/legus_hlsp/` | Tiny HLSP table, readme, stub PDFs, minimal NN scaler+`.pt` for tests |
| `scripts/run_analyze_catalog_mid_mdd.sh` | Runs the bundled driver with `PYTHONPATH` set |
| `scripts/sync_pipeline_from_repo.sh` | Refreshes `bundled_pipeline/` from `../python3_sc/` when you have the full repo |
| `src/legus_slug_pipeline/` | Dataclass defaults and `apply_env()` |
| `docs/` | `BUNDLE_MANIFEST.md`, `GADI.md`, architecture notes |
| `tests/` | Pytest smoke + bundled NN fixture test |

See **`docs/BUNDLE_MANIFEST.md`** for the full file list and what you must still provide on Gadi.

## Environment

```bash
cd new_pipeline_slug_legus
uv venv
uv sync --extra dev
source .venv/bin/activate
```

Install **`slugpy`** (and **`joblib`** if your scalers need it) by your site’s method; they are runtime requirements but not always available as PyPI wheels on HPC.

## Run on Gadi (or any host)

```bash
./scripts/run_analyze_catalog_mid_mdd.sh -h
```

Full runbook: **`docs/GADI.md`**.

## Refresh Python copies from the parent repo

When you have `slugfiles` checked out with `python3_sc/` next to this folder:

```bash
./scripts/sync_pipeline_from_repo.sh
```

## Tests

```bash
uv run pytest -q
```
