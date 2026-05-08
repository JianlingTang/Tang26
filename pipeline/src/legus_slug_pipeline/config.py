from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem defaults aligned with ``analyze_catalog_mid_mdd.py`` CLI."""

    legus_cct_root: str = "/g/data/jh2/jt4478/make_LEGUS_CCT"
    legus_tab_dir: str = "/home/100/jt4478/slugfiles/LEGUS_cat"
    cluster_slug_lib_dir: str = "/g/data/jh2/jt4478/cluster_slug"
    cluster_slug_lib_name: str = "/g/data/jh2/jt4478/cluster_slug/tang"
    output_mcmc_chains_dir: str = "/g/data/jh2/jt4478/output_mcmc_chains"
    nn_comp_dir: str | None = None

    def apply_env(self) -> None:
        """Export paths consumed by ``catalog_readers`` (and related code) via env."""
        os.environ["LEGUS_CCT_ROOT"] = self.legus_cct_root
        os.environ["LEGUS_TAB_DIR"] = self.legus_tab_dir
        os.environ["CLUSTER_SLUG_LIB_DIR"] = self.cluster_slug_lib_dir
        os.environ["CLUSTER_SLUG_LIB_NAME"] = self.cluster_slug_lib_name
