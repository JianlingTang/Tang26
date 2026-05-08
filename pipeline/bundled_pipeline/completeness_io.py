"""
NN-driven catalog completeness and legacy NGC628 library ``.npy`` loaders.

The production LEGUS + cluster_slug pipeline uses
``predict_catalog_completeness_with_nn`` and optional explicit scaler/model
paths. Functions ``load_library_completeness`` / ``infer_galaxy_key_from_basename``
remain for older call sites that still read precomputed ``libngc628*.npy`` maps.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LEGUS_NGC628_COMPLETENESS_MAP: Dict[str, Dict[int, str]] = {
    "628c": {
        5: "libngc628c_comp.npy",
        4: "libngc628cnoUV_comp.npy",
    },
    "628e": {
        5: "libngc628e_comp.npy",
        4: "libngc628enoUV_comp.npy",
    },
}


def infer_galaxy_key_from_basename(basename: str) -> str:
    if "628c" in basename:
        return "628c"
    if "628e" in basename:
        return "628e"
    raise ValueError(
        f"Cannot infer galaxy key from basename '{basename}'. "
        "Expected one of '628c' or '628e'."
    )


def load_library_completeness(lcdir: str, basename: str, n_detected_filters: int) -> np.ndarray:
    """Load precomputed library completeness for one catalog filter subset."""
    galaxy_key = infer_galaxy_key_from_basename(basename)
    if n_detected_filters not in LEGUS_NGC628_COMPLETENESS_MAP[galaxy_key]:
        raise ValueError(
            f"Completeness mapping missing for galaxy={galaxy_key}, "
            f"n_detected_filters={n_detected_filters}"
        )
    fname = LEGUS_NGC628_COMPLETENESS_MAP[galaxy_key][n_detected_filters]
    path = os.path.join(lcdir, fname)
    return np.load(path)


def _resolve_nn_artifacts(nn_dir: str, galaxy_fullname: str) -> Tuple[str, str]:
    """Pair scaler pickle with Torch ``.pt`` model; galaxy name narrows globs, then directory fallbacks."""
    nn_dir = os.path.expanduser(nn_dir)
    gal = galaxy_fullname or ""
    search_dirs = [nn_dir]
    if gal:
        search_dirs.extend(
            [
                os.path.join(nn_dir, gal),
                os.path.join(nn_dir, gal, "checkpoints"),
            ]
        )
    search_dirs.extend(glob.glob(os.path.join(nn_dir, "*", "checkpoints")))
    search_dirs = [d for d in dict.fromkeys(search_dirs) if os.path.isdir(d)]

    def _uniq_sorted(paths: List[str]) -> List[str]:
        return sorted(set(paths))

    scaler_matches: List[str] = []
    for d in search_dirs:
        scaler_matches.extend(
            glob.glob(os.path.join(d, f"scaler_phot*{gal}*.pkl"))
            + glob.glob(os.path.join(d, f"*optuna*{gal}*scaler*.pkl"))
            + glob.glob(os.path.join(d, f"*{gal}*scaler*.pkl"))
            + glob.glob(os.path.join(d, f"*{gal}*scaler*.pkg"))
        )
    scaler_matches = _uniq_sorted(scaler_matches)
    if not scaler_matches:
        fallback_scalers: List[str] = []
        for d in search_dirs:
            fallback_scalers.extend(
                glob.glob(os.path.join(d, "scaler_phot*.pkl"))
                + glob.glob(os.path.join(d, "*scaler*.pkl"))
                + glob.glob(os.path.join(d, "*scaler*.pkg"))
                + glob.glob(os.path.join(d, "best_model_phot_model0.pkl"))
            )
        scaler_matches = _uniq_sorted(fallback_scalers)

    model_matches: List[str] = []
    for d in search_dirs:
        model_matches.extend(
            glob.glob(os.path.join(d, f"best_model_phot*{gal}*.pt"))
            + glob.glob(os.path.join(d, f"*optuna*{gal}*.pt"))
            + glob.glob(os.path.join(d, f"*{gal}*.pt"))
            + glob.glob(os.path.join(d, f"*{gal}*model*.pt"))
            + glob.glob(os.path.join(d, f"*{gal}*nn*.pt"))
            + glob.glob(os.path.join(d, f"*{gal}*model*.pkl"))
            + glob.glob(os.path.join(d, f"*{gal}*model*.pkg"))
            + glob.glob(os.path.join(d, f"*{gal}*nn*.pkl"))
            + glob.glob(os.path.join(d, f"*{gal}*nn*.pkg"))
        )
    model_matches = _uniq_sorted(model_matches)
    if not model_matches:
        fallback_models: List[str] = []
        for d in search_dirs:
            fallback_models.extend(
                glob.glob(os.path.join(d, "best_model_phot*.pt"))
                + glob.glob(os.path.join(d, "*.pt"))
            )
        model_matches = _uniq_sorted(fallback_models)

    if not scaler_matches or not model_matches:
        raise FileNotFoundError(
            f"Missing NN completeness artifacts for galaxy '{galaxy_fullname}' in '{nn_dir}'."
        )
    preferred_models = [m for m in model_matches if m.endswith(".pt")]
    if preferred_models:
        optuna_models = [m for m in preferred_models if "optuna" in os.path.basename(m).lower()]
        model_path = optuna_models[0] if optuna_models else preferred_models[0]
    else:
        model_path = model_matches[0]

    optuna_scalers = [s for s in scaler_matches if "optuna" in os.path.basename(s).lower()]
    scaler_path = optuna_scalers[0] if optuna_scalers else scaler_matches[0]

    return scaler_path, model_path


def _load_pickle(path: str):
    try:
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
    except (pickle.UnpicklingError, EOFError, UnicodeDecodeError, ValueError):
        import joblib

        obj = joblib.load(path)
    # Training scripts often pickle a dict, e.g. {"scaler_photo": StandardScaler(), ...}
    if isinstance(obj, dict):
        for key in ("scaler_photo", "photo_scaler", "scaler"):
            if key in obj and hasattr(obj[key], "transform"):
                return obj[key]
        if len(obj) == 1:
            sole = next(iter(obj.values()))
            if hasattr(sole, "transform"):
                return sole
    return obj


def _is_legus_uv_or_u_filter(band: str) -> bool:
    """True for LEGUS-style UV (F275W) or U (F336W) band names."""
    s = str(band).upper()
    return "F275W" in s or "F336W" in s


def legus_nn_missing_uv_u_fills(
    filters_cat: Sequence[str],
    phot: np.ndarray,
    detect: np.ndarray,
    dmod: float,
    full_filter_order: Sequence[str],
    subset_filters: Sequence[str],
    *,
    use_apparent_magnitude: bool = False,
) -> Dict[str, float]:
    """
    Magnitudes to use for **missing** UV / U columns when densifying NN input.

    For each band in ``full_filter_order`` that is absent from ``subset_filters`` and
    matches F275W or F336W, if that band exists as a column in ``filters_cat`` with at
    least one detected cluster, set fill to ``min(mag) + 0.5`` over detected rows, where
    ``mag`` is absolute catalog photometry or ``phot + dmod`` when
    ``use_apparent_magnitude`` is True (consistent with library NN paths that pass
    apparent magnitudes to the scaler).
    """
    subset_set = {str(f) for f in subset_filters}
    names = [str(f) for f in filters_cat]
    phot = np.asarray(phot, dtype=float)
    detect = np.asarray(detect, dtype=bool)
    dm = float(dmod)
    fills: Dict[str, float] = {}

    for f in full_filter_order:
        fs = str(f)
        if fs in subset_set:
            continue
        if not _is_legus_uv_or_u_filter(fs):
            continue
        if fs not in names:
            continue
        j = names.index(fs)
        m = detect[:, j] & np.isfinite(phot[:, j])
        if not np.any(m):
            continue
        mag = phot[m, j]
        if use_apparent_magnitude:
            mag = mag + dm
        fills[fs] = float(np.min(mag)) + 0.5
    return fills


def _apply_scaler(scaler, x: np.ndarray) -> np.ndarray:
    if hasattr(scaler, "transform"):
        return scaler.transform(x)
    # Allow dictionary-style scaler payloads as fallback.
    if isinstance(scaler, dict) and "mean_" in scaler and "scale_" in scaler:
        return (x - np.asarray(scaler["mean_"])) / np.asarray(scaler["scale_"])
    raise TypeError("Unsupported scaler object; expected sklearn-like scaler or dict with mean_/scale_.")


def _predict_prob(model, x_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        out = model.predict_proba(x_scaled)
        return out[:, -1] if out.ndim == 2 else np.ravel(out)
    if hasattr(model, "predict"):
        out = model.predict(x_scaled)
        return np.ravel(out)
    raise TypeError("Unsupported model object; expected predict_proba or predict method.")


def _build_dense_photo_features(
    phot_filterset: np.ndarray,
    subset_filters: List[str],
    full_filter_order: List[str],
    scaler=None,
    missing_band_fills: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Build fixed-width photo feature matrix in `full_filter_order`.

    Missing filters default to scaler.mean_ (if available) so their standardized value
    is close to 0; otherwise 0. Optional ``missing_band_fills`` overrides per-band
    constants (e.g. LEGUS UV/U fills from observed-cluster minima + 0.5 mag).
    """
    if phot_filterset.ndim != 2:
        raise ValueError("phot_filterset must be 2D")
    if len(subset_filters) != phot_filterset.shape[1]:
        raise ValueError("subset_filters length must match phot_filterset columns")
    if len(full_filter_order) == 0:
        raise ValueError("full_filter_order is empty")

    n = phot_filterset.shape[0]
    n_full = len(full_filter_order)
    if scaler is not None and hasattr(scaler, "mean_") and len(getattr(scaler, "mean_")) == n_full:
        dense = np.tile(np.asarray(scaler.mean_, dtype=float), (n, 1))
    else:
        dense = np.zeros((n, n_full), dtype=float)

    subset_map = {f: i for i, f in enumerate(subset_filters)}
    for j, f in enumerate(full_filter_order):
        if f in subset_map:
            dense[:, j] = phot_filterset[:, subset_map[f]]
        elif missing_band_fills and str(f) in missing_band_fills:
            dense[:, j] = float(missing_band_fills[str(f)])
    return dense


def _normalize_torch_state_dict(state: Dict[str, object]) -> Dict[str, object]:
    """Strip prefixes from common training wrappers (DataParallel, MLP.net, etc.)."""
    prefixes = ("module.", "net.")
    out: Dict[str, object] = {}
    for key, val in state.items():
        new_key = key
        while any(new_key.startswith(p) for p in prefixes):
            for p in prefixes:
                if new_key.startswith(p):
                    new_key = new_key[len(p) :]
                    break
        out[new_key] = val
    return out


_torch_nn_cache: Dict[str, Tuple[float, object, int]] = {}


def _load_torch_checkpoint_predictor(
    model_path: str,
    input_dim_hint: Optional[int] = None,
):
    """
    Build the same MLP skeleton as training: (Linear+GELU)*n_hidden -> Linear(1),
    BCEWithLogits at train time -> sigmoid(logits) at inference.

    Checkpoints from an ``MLP`` class that stores weights under ``net.*`` are
    supported after key normalization to match ``nn.Sequential`` keys.
    """
    import torch
    import torch.nn as nn

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError("Torch checkpoint should be a dict-like object")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))

    def _looks_like_flat_state_dict(obj: dict) -> bool:
        if not obj or "model_state_dict" in obj or "state_dict" in obj:
            return False
        for v in obj.values():
            if not isinstance(v, torch.Tensor):
                return False
        return any(isinstance(k, str) and k.endswith("weight") for k in obj)

    if state is None and _looks_like_flat_state_dict(ckpt):
        state = ckpt
        ckpt = {"model_state_dict": state, "model_config": {}}
    if state is None:
        raise KeyError("Checkpoint missing model_state_dict/state_dict")
    state = _normalize_torch_state_dict(state)

    cfg = ckpt.get("model_config", {})
    hidden_dim = int(cfg.get("hidden_dim", 128))
    n_hidden = int(cfg.get("n_hidden", 2))

    linear_w = [
        (int(k.split(".")[0]), k, state[k])
        for k in state
        if k.endswith("weight") and getattr(state[k], "ndim", 0) == 2
    ]
    linear_w.sort(key=lambda t: t[0])
    if linear_w:
        input_dim_sd = int(linear_w[0][2].shape[1])
    else:
        input_dim_sd = None
    input_dim = int(cfg.get("input_dim", input_dim_sd or input_dim_hint or 5))
    if input_dim_sd is not None and input_dim != input_dim_sd:
        input_dim = input_dim_sd
    if input_dim_hint is not None and input_dim != input_dim_hint:
        raise ValueError(
            f"First-layer in_features={input_dim} does not match scaled feature "
            f"width {input_dim_hint} (check scaler vs {model_path})."
        )

    layers = []
    in_dim = input_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, input_dim


def _predict_prob_torch(model_path: str, x_scaled: np.ndarray) -> np.ndarray:
    import torch

    mtime = os.path.getmtime(model_path)
    cache_key = model_path
    cached = _torch_nn_cache.get(cache_key)
    if cached is None or cached[0] != mtime:
        model, input_dim = _load_torch_checkpoint_predictor(
            model_path, input_dim_hint=int(x_scaled.shape[1])
        )
        _torch_nn_cache[cache_key] = (mtime, model, input_dim)
    else:
        _, model, input_dim = cached
        if int(x_scaled.shape[1]) != input_dim:
            raise ValueError(
                f"NN input feature mismatch: got {x_scaled.shape[1]}, expected {input_dim}"
            )

    with torch.no_grad():
        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        logits = model(x_t).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
    return np.ravel(prob)


def _nn_scaler_model_paths(
    nn_dir: Optional[str],
    galaxy_fullname: str,
    nn_scaler_path: Optional[str],
    nn_model_path: Optional[str],
) -> Tuple[str, str]:
    if nn_scaler_path is not None or nn_model_path is not None:
        if nn_scaler_path is None or nn_model_path is None:
            raise ValueError(
                "nn_scaler_path and nn_model_path must both be set, or both omitted "
                "(use nn_dir + galaxy_fullname for glob discovery)."
            )
        return os.path.expanduser(nn_scaler_path), os.path.expanduser(nn_model_path)
    if nn_dir is None:
        raise ValueError(
            "Provide nn_scaler_path + nn_model_path, or nn_dir for NN completeness."
        )
    return _resolve_nn_artifacts(nn_dir, galaxy_fullname)


def predict_catalog_completeness_with_nn(
    phot_filterset: np.ndarray,
    galaxy_fullname: str = "",
    nn_dir: Optional[str] = None,
    subset_filters: Optional[List[str]] = None,
    full_filter_order: Optional[List[str]] = None,
    nn_scaler_path: Optional[str] = None,
    nn_model_path: Optional[str] = None,
    missing_band_fills: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Predict per-cluster completeness using a phot-only MLP + StandardScaler,
    matching typical training: ``X_photo -> scaler_photo.transform``
    then ``sigmoid(MLP(x))`` with ``BCEWithLogitsLoss`` at train time.

    **Artifact resolution** (first match wins):

    1. If ``nn_scaler_path`` and ``nn_model_path`` are set, those files are used
       (user-supplied paths; ``nn_dir`` is ignored for discovery).
    2. Else ``nn_dir`` must be set; scaler and ``.pt`` model are found via
       ``_resolve_nn_artifacts(nn_dir, galaxy_fullname)``.

    When ``subset_filters`` and ``full_filter_order`` are set, photometry is
    expanded to a dense row in ``full_filter_order`` (missing bands filled with
    the scaler training means so standardized values are ~0, unless
    ``missing_band_fills`` supplies constants for those columns), then
    ``scaler.transform`` — same layout as ``nf = len(full_filter_order)`` in
    training.

    `phot_filterset` is shape ``(n_cluster, n_detected_filters)``.
    """
    scaler_path, model_path = _nn_scaler_model_paths(
        nn_dir, galaxy_fullname, nn_scaler_path, nn_model_path
    )
    scaler = _load_pickle(scaler_path)
    if subset_filters is not None and full_filter_order is not None:
        x_raw = _build_dense_photo_features(
            phot_filterset=phot_filterset,
            subset_filters=subset_filters,
            full_filter_order=full_filter_order,
            scaler=scaler,
            missing_band_fills=missing_band_fills,
        )
    else:
        x_raw = phot_filterset
    x_scaled = _apply_scaler(scaler, x_raw)

    # Support either sklearn-like pickled model or torch checkpoint (.pt).
    if model_path.endswith(".pt"):
        pred = _predict_prob_torch(model_path, x_scaled)
    else:
        model = _load_pickle(model_path)
        pred = _predict_prob(model, x_scaled)
    return np.clip(pred, 0.0, 1.0)
