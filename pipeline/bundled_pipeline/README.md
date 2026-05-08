# Bundled pipeline (offline copy)

These files are a **verbatim copy** of the matching modules under the parent repository `slugfiles/python3_sc/`. They are kept here so you can **tar this directory tree and run on Gadi** without assuming the full `slugfiles` checkout is present.

Imports are **flat** (same as in `python3_sc`): `analyze_catalog_mid_mdd.py` expects `catalog_readers`, `clean_legus`, etc. on `PYTHONPATH`.

**Refresh from upstream** (when you do have the full repo):

```bash
./scripts/sync_pipeline_from_repo.sh
```
