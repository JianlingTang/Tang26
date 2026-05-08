# Running on Gadi (NCI)

## 1. Transfer the bundle

From your laptop (example):

```bash
cd /path/to/slugfiles
tar czvf legus_slug_pipeline.tgz new_pipeline_slug_legus
scp legus_slug_pipeline.tgz USER@gadi.nci.org.au:~/
```

On Gadi:

```bash
tar xzvf legus_slug_pipeline.tgz
cd new_pipeline_slug_legus
```

## 2. Create an environment

Use **uv** or **venv + pip**. Install this project’s dependencies **and** `slugpy` (and `joblib` if needed) from your usual NCI modules or wheels.

Example with uv (if available):

```bash
module load python3/3.13.0  # example; use your project’s module set
uv venv
uv sync --extra dev
source .venv/bin/activate
pip install slugpy joblib   # adjust to your install method
```

## 3. Run the analysis driver

Always prepend `bundled_pipeline` to `PYTHONPATH`, or use the helper script:

```bash
chmod +x scripts/run_analyze_catalog_mid_mdd.sh
./scripts/run_analyze_catalog_mid_mdd.sh \
  --legus-cct-root "${LEGUS_CCT_ROOT}" \
  --legus-tab-dir "${LEGUS_TAB_DIR}" \
  --cluster-slug-lib-name "${CLUSTER_SLUG_LIB}" \
  --nn-dir "${NN_COMP_DIR}" \
  --output-mcmc-chains-dir "${OUT_CHAINS_DIR}" \
  dummy_lib.fits \
  . \
  mass.pdf age.pdf av.pdf \
  /path/to/catalog1.tab
```

Notes:

- The first positional `libname` is **legacy**; the library path used by `read_cluster` is **`--cluster-slug-lib-name`**.
- Replace PDF paths and catalog paths with your real inputs.
- Ensure **NN artifacts** exist under `--nn-dir` (or pass explicit `--nn-scaler` / `--nn-model`).

## 4. Refresh Python sources after editing on another machine

If you later check out the full `slugfiles` repo next to this folder:

```bash
./scripts/sync_pipeline_from_repo.sh
```
