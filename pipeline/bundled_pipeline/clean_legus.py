#!/usr/bin/env python3
"""
This script contains some code to do cleanup on the LEGUS
catalogs. The cleaning required is that (1) we need to remove clusters
that appear in more than one catalog; (2) remove clusters that
nominally have zero chance of being observed, which can get into the
sample anyway due to peculiarities of the way that the magnitude limit
for visual classification was combined with the aperture correction
"""

import glob
import os.path as osp

import numpy as np
from astropy.io import ascii as apyascii
from astropy.io import fits as apyfits

from completeness_calculator import *
from completeness_io import legus_nn_missing_uv_u_fills, predict_catalog_completeness_with_nn
def clean_legus(
    catalogs,
    verbose,
    nn_dir=None,
    nn_scaler_path=None,
    nn_model_path=None,
    comp_threshold=0.0,
    enforce_hybrid_criteria=False,
    lib_vmag_max=-6.0,
    min_bands_nonzero=4,
):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not
       nn_dir : str or None
          directory with NN completeness artifacts (glob by galaxy name), or
          None if ``nn_scaler_path`` and ``nn_model_path`` are set
       nn_scaler_path, nn_model_path : str or None
          explicit paths to scaler (pickle/joblib) and Torch ``.pt`` model;
          if both set, they override ``nn_dir`` for completeness

    Returns
       nothing
    """
    if (nn_scaler_path is None) ^ (nn_model_path is None):
        raise ValueError(
            "nn_scaler_path and nn_model_path must be given together or both omitted."
        )
    has_paths = nn_scaler_path is not None and nn_model_path is not None
    if not has_paths and nn_dir is None:
        raise ValueError(
            "clean_legus requires either nn_dir or both nn_scaler_path and "
            "nn_model_path for NN observational completeness."
        )

    # First identify clusters with extremely low expected completeness;
    # remove them from both the sorted and original photometry lists
    comp_min = comp_threshold
    per_filterset_fields = [
        'cid_filterset',
        'phot_filterset',
        'photerr_filterset',
        'mass_filterset',
        'age_filterset',
        'mass_min_filterset',
        'mass_max_filterset',
        'age_min_filterset',
        'age_max_filterset',
        'detect_filterset',
    ]
    per_catalog_fields = [
        'cid', 'phot', 'photerr', 'detect', 'filtersets_index',
        'ra', 'dec', 'mass', 'age', 'mass_min', 'mass_max', 'age_min', 'age_max'
    ]
    if enforce_hybrid_criteria:
        for c in catalogs:
            phot = np.asarray(c["phot"], dtype=float)
            detect = np.asarray(c["detect"], dtype=bool)
            filters = [str(f) for f in c["filters"]]

            v_idx = None
            for token in ("F555W", "F606W"):
                for ii, ff in enumerate(filters):
                    if token in ff:
                        v_idx = ii
                        break
                if v_idx is not None:
                    break
            if v_idx is None:
                raise ValueError(f"No V-like band (F555W/F606W) in filters for {c.get('basename', 'catalog')}")

            b_candidates = [ii for ii, ff in enumerate(filters) if ("F435W" in ff or "F438W" in ff)]
            i_candidates = [ii for ii, ff in enumerate(filters) if "F814W" in ff]
            b_detect = detect[:, b_candidates].any(axis=1) if b_candidates else np.zeros(len(phot), dtype=bool)
            i_detect = detect[:, i_candidates].any(axis=1) if i_candidates else np.zeros(len(phot), dtype=bool)
            n_detect = np.sum(detect, axis=1)

            keep_mask = (
                detect[:, v_idx]
                & np.isfinite(phot[:, v_idx])
                & (phot[:, v_idx] <= float(lib_vmag_max))
                & (n_detect >= int(min_bands_nonzero))
                & (b_detect | i_detect)
            )

            cids_to_delete = c["cid"][~keep_mask]
            if np.any(~keep_mask):
                for field in per_catalog_fields:
                    if field in c:
                        c[field] = c[field][keep_mask]
                for cid in cids_to_delete:
                    for i in range(len(c["filtersets"])):
                        idx_keep = c["cid_filterset"][i] != cid
                        for field in per_filterset_fields:
                            if field in c:
                                c[field][i] = c[field][i][idx_keep]
            if verbose and len(cids_to_delete) > 0:
                print(
                    ("   [clean_legus]: removed {:d} clusters from catalog {:s} "
                     "by hybrid-aligned criteria (V, B/I, Nband, Vmag cut)")
                    .format(len(cids_to_delete), c["basename"])
                )

    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            galaxy_fullname = c.get('galaxy_fullname', c['basename'])
            dmod = float(c["dmod"])
            phot_filterset_app = np.asarray(c["phot_filterset"][i], dtype=float) + dmod
            fills_obs = legus_nn_missing_uv_u_fills(
                c["filters"],
                np.asarray(c["phot"], dtype=float),
                np.asarray(c["detect"], dtype=bool),
                dmod,
                [str(f) for f in c["filters"]],
                [str(f) for f in c["filtersets"][i]],
                use_apparent_magnitude=True,
            )
            comp = predict_catalog_completeness_with_nn(
                phot_filterset_app,
                galaxy_fullname=galaxy_fullname,
                nn_dir=nn_dir,
                subset_filters=c['filtersets'][i],
                full_filter_order=c['filters'],
                nn_scaler_path=nn_scaler_path,
                nn_model_path=nn_model_path,
                missing_band_fills=fills_obs if fills_obs else None,
            )
            idx_del = comp < comp_min
            idx_keep = comp >= comp_min
            cids_to_delete = c['cid_filterset'][i][idx_del]
            for field in per_filterset_fields:
                if field in c:
                    c[field][i] = c[field][i][idx_keep]

            ndel += len(cids_to_delete)
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                for field in per_catalog_fields:
                    if field in c:
                        c[field] = c[field][idx_keep]

                
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to completeness < {:.3f} issue").
                  format(ndel, c['basename'], comp_min))

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        for field in per_catalog_fields:
            if field in c:
                c[field] = c[field][idx_keep]


        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                for field in per_filterset_fields:
                    if field in c:
                        c[field][i] = c[field][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

def clean_legus_cl12(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # Legacy allcomp/*.npy completeness removed; use clean_legus(..., nn_dir=...).

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['class'] = c['class'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]

        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['class_filterset'][i] = c['class_filterset'][i][idx_keep]
                c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]
                

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
            
    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            idx_del =  c['class_filterset'][i] >= 3.
            idx_keep = c['class_filterset'][i] < 3.
            cids_to_delete = c['cid_filterset'][i][idx_del]
            c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
            c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
            c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
            c['class_filterset'][i] = c['class_filterset'][i][idx_keep]
            c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]
            ndel += len(cids_to_delete)
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['class'] = c['class'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]



                
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} classified as class 3 clusters.").
                  format(ndel, c['basename']))
            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

def clean_legus_rOGC(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # Legacy allcomp/*.npy completeness removed; use clean_legus(..., nn_dir=...).

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['mass'] = c['mass'][idx_keep]
        c['age'] = c['age'][idx_keep]
        c['mass_min'] = c['mass_min'][idx_keep]
        c['mass_max'] = c['mass_max'][idx_keep]
        c['age_min'] = c['age_min'][idx_keep]
        c['age_max'] = c['age_max'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]

        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
                c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
                c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
                c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
                c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
                c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
                c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

# def clean_legus_rd5(catalogs, verbose):
#     """
#     This cleans up LEGUS catalogs, getting the data into a uniform
#     state suitable for Bayesian analysis

#     Parameters
#        catalogs : list
#           catalogs prepared by the analyze_catalog script
#        verbose : bool
#           print verbose output or not

#     Returns
#        nothing
#     """

#     # First identify clusters that nominally have zero completeness;
#     # remove them from both the sorted and original photometry lists
#     comp_dir = '/home/100/jt4478/slugfiles/allcomp/'
#     for c in catalogs:
#         ndel = 0
#         for i in range(len(c['phot_filterset'])):
#             # comp = c['comp'][i].comp(c['phot_filterset'][i])
#             if len(c['phot_filterset'][i][0,:]) == 5: 
#                 if '628c' in c['basename'] : 
#                     comp = np.load(comp_dir+'LEGUS628c_nUV_comp.npy')
#                     print('c loaded')
#                 elif '628e' in c['basename'] :
#                     comp = np.load(comp_dir+'LEGUS628e_nUV_comp.npy')
#                 else :
#                     print('data completeness is not valid')
#                     exit(0)
#             elif len(c['phot_filterset'][i][0,:]) == 4:
#                 if '628c' in c['basename'] : 
#                     comp = np.load(comp_dir+'LEGUS628c_nUV_noUV_comp.npy')
#                 elif '628e' in c['basename'] :
#                     comp = np.load(comp_dir+'LEGUS628e_nUV_noUV_comp.npy')
#                 else :
#                     print('data completeness is not valid')
#                     exit(0)
#             idx_del = comp == 0.0
#             idx_keep = comp > 0.
#             cids_to_delete = c['cid_filterset'][i][idx_del]
#             c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
#             c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
#             c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
#             # c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
#             # c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
#             # c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
#             # c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
#             # c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
#             # c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]

#             ndel += len(cids_to_delete)
#             for cid in cids_to_delete: 
#                 idx_keep = c['cid'] != cid
#                 c['cid'] = c['cid'][idx_keep]
#                 c['phot'] = c['phot'][idx_keep]
#                 c['photerr'] = c['photerr'][idx_keep]
#                 c['detect'] = c['detect'][idx_keep]
#                 c['filtersets_index'] = c['filtersets_index'][idx_keep]
#                 c['ra'] = c['ra'][idx_keep]
#                 c['dec'] = c['dec'][idx_keep]
#                 c['mass'] = c['mass'][idx_keep]
#                 c['age'] = c['age'][idx_keep]
#                 c['mass_min'] = c['mass_min'][idx_keep]
#                 c['mass_max'] = c['mass_max'][idx_keep]
#                 c['age_min'] = c['age_min'][idx_keep]
#                 c['age_max'] = c['age_max'][idx_keep]

                
#         if verbose and ndel > 0:
#             print(("   [clean_legus]: removed {:d} clusters from "
#                    "catalog {:s} due to completeness = 0 issue").
#                   format(ndel, c['basename']))

#     # Second remove clusters that are duplicated across catalogs
#     for i, c in enumerate(catalogs):
#         # Compare every RA and DEC in every other catalog to find
#         # duplicates
#         dup = np.zeros(len(c['ra']), dtype=bool)
#         for c2 in catalogs[i+1:]:
#             ra_match = np.equal.outer(c['ra'], c2['ra'])
#             dec_match = np.equal.outer(c['dec'], c2['dec'])
#             dup_cat = np.logical_or.reduce(
#                 np.logical_and(ra_match, dec_match), axis=1)
#             dup = np.logical_or(dup, dup_cat)

#         # Delete duplicates
#         idx_keep = np.logical_not(dup)
#         cids_to_delete = c['cid'][dup]
#         ndel = len(cids_to_delete)
#         c['cid'] = c['cid'][idx_keep]
#         c['phot'] = c['phot'][idx_keep]
#         c['photerr'] = c['photerr'][idx_keep]
#         c['detect'] = c['detect'][idx_keep]
#         c['filtersets_index'] = c['filtersets_index'][idx_keep]
#         c['mass'] = c['mass'][idx_keep]
#         c['age'] = c['age'][idx_keep]
#         c['mass_min'] = c['mass_min'][idx_keep]
#         c['mass_max'] = c['mass_max'][idx_keep]
#         c['age_min'] = c['age_min'][idx_keep]
#         c['age_max'] = c['age_max'][idx_keep]


#         for cid in cids_to_delete:
#             for i in range(len(c['filtersets'])):
#                 idx_keep = c['cid_filterset'][i] != cid
#                 c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
#                 c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
#                 c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
#                 # c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
#                 # c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
#                 # c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
#                 # c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
#                 # c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
#                 # c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]

#         # Print message
#         if verbose and ndel > 0:
#             print(("   [clean_legus]: removed {:d} clusters from "
#                    "catalog {:s} due to duplication").
#                   format(ndel, c['basename']))
            
#     # Recalculate total number of clusters and return
#     ncl = 0
#     for c in catalogs:
#         ncl += len(c['phot'])
#     return ncl

def clean_legus_comp(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """
    # First identify clusters that nominally have zero completeness;
    # remove them from both the sorted and original photometry lists
    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            comp = c['comp_filterset'][i]
            idx_del = comp == 0.0
            idx_keep = comp > 0
            cids_to_delete = c['cid_filterset'][i][idx_del]
            c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
            c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
            c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
            ndel += len(cids_to_delete)
            for cid in cids_to_delete:
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to completeness = 0 issue").
                  format(ndel, c['basename']))

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):

        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl


def clean_legus_rd5(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # Legacy allcomp/*.npy completeness removed; use clean_legus(..., nn_dir=...).
    libdir = '/g/data/jh2/jt4478/getpdfs_o/'

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['mass'] = c['mass'][idx_keep]
        c['age'] = c['age'][idx_keep]
        c['mass_min'] = c['mass_min'][idx_keep]
        c['mass_max'] = c['mass_max'][idx_keep]
        c['age_min'] = c['age_min'][idx_keep]
        c['age_max'] = c['age_max'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]

        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
                c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
                c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
                c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
                c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
                c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
                c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
    # remove the clusters with high D5 norm in both U and UV filters 
    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            if i == 0:
                if '628c' in c['basename']:
                    fname = osp.join(libdir, 
                                 'hlsp_628c5ft_phys_phot_pdf_1D.fits')
                elif '628e' in c['basename']:
                    fname = osp.join(libdir, 
                                 'hlsp_628e5ft_phys_phot_pdf_1D.fits')
            else:
                if '628c' in c['basename']:
                    fname = osp.join(libdir, 
                                 'hlsp_628c4ft_phys_phot_pdf_1D.fits')
                elif '628e' in c['basename']:
                    fname = osp.join(libdir, 
                                 'hlsp_628e4ft_phys_phot_pdf_1D.fits')
            csdist = []
            csdistnorm = []
            hdulist = apyfits.open(fname)
            csdist.append(hdulist[2].data['Phot_dist'])
            csdistnorm.append(hdulist[2].data['Phot_dist_norm'])
            ar1 = hdulist[2].data['Cluster_ID'][csdistnorm[0][:,1]>3]
            ar2 = hdulist[2].data['Cluster_ID'][csdistnorm[0][:,0]>3]
            common_cid = list(set(ar1) & set(ar2))
            hdulist.close()

            # Boolean array for indices to be deleted
            idx_del = np.in1d(c['cid_filterset'][i], common_cid)

            # Boolean array for indices to be kept
            idx_keep = ~idx_del

            # delete cids and other properties
            cids_to_delete = c['cid_filterset'][i][idx_del]
            c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
            c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
            c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
            c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
            c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
            c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
            c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
            c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
            c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
            c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]


            ndel += len(cids_to_delete)
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
                c['mass'] = c['mass'][idx_keep]
                c['age'] = c['age'][idx_keep]
                c['mass_min'] = c['mass_min'][idx_keep]
                c['mass_max'] = c['mass_max'][idx_keep]
                c['age_min'] = c['age_min'][idx_keep]
                c['age_max'] = c['age_max'][idx_keep]


        # else :
        #         print('[clean_legus]: Invalid filterset index. exiting...')
        #         exit(0)


        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} with high D5_norm values.)".
                  format(ndel, c['basename'])))

            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

'''
clean_legus_comp90(args,kwargs) is a function that filters out 
clusters with completeness < 90%, this test is to verify our 
results from analyze_midmdd pipeline is independent of the shape 
of the completeness function. 
'''

def clean_legus_comp90(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # Legacy allcomp/*.npy completeness (<90%) removed; use NN completeness upstream.

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['mass'] = c['mass'][idx_keep]
        c['age'] = c['age'][idx_keep]
        c['mass_min'] = c['mass_min'][idx_keep]
        c['mass_max'] = c['mass_max'][idx_keep]
        c['age_min'] = c['age_min'][idx_keep]
        c['age_max'] = c['age_max'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]

        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
                c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
                c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
                c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
                c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
                c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
            
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

# from LEGUS_comp_div import libcomp 
def clean_legus_cal_comp(catalogs, verbose, testm101, comptype):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not
       testm101 : bool
          print verbose output or not

    Returns
       nothing
    """

    # First identify clusters that nominally have zero completeness;
    # remove them from both the sorted and original photometry lists
    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            cg = c['galaxy'].split('_')[0]
            if testm101:
                comp_class = completeness_LEGUS_ngc5457(cg, comptype=comptype)
                comp = comp_class.LEGUS_cal(c['phot_filterset'][i],c['filtersets'][i], c['mass_filterset'][i])
            else:
                comp_class = completeness_LEGUS(cg)
                comp = comp_class.LEGUS_cal(c['phot_filterset'][i],c['filtersets'][i], c['mass_filterset'][i])
            idx_del = comp == 0.0
            idx_keep = comp > 0.
            cids_to_delete = c['cid_filterset'][i][idx_del]
            c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
            c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
            c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
            c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
            c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
            c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
            c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
            c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
            c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
            c['ra_filterset'][i] = c['ra_filterset'][i][idx_keep]
            c['dec_filterset'][i] = c['dec_filterset'][i][idx_keep]

            ndel += len(cids_to_delete)
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
                c['mass'] = c['mass'][idx_keep]
                c['age'] = c['age'][idx_keep]
                c['mass_min'] = c['mass_min'][idx_keep]
                c['mass_max'] = c['mass_max'][idx_keep]
                c['age_min'] = c['age_min'][idx_keep]
                c['age_max'] = c['age_max'][idx_keep]

                
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to completeness = 0 issue").
                  format(ndel, c['basename']))

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['mass'] = c['mass'][idx_keep]
        c['age'] = c['age'][idx_keep]
        c['mass_min'] = c['mass_min'][idx_keep]
        c['mass_max'] = c['mass_max'][idx_keep]
        c['age_min'] = c['age_min'][idx_keep]
        c['age_max'] = c['age_max'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]
        


        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
                c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
                c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
                c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
                c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
                c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
                c['ra_filterset'][i] = c['ra_filterset'][i][idx_keep]
                c['dec_filterset'][i] = c['dec_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
        for i, c in enumerate(catalogs):
            # Remove empty arrays from phot_filterset and related fields
                if c['cid'].size ==0:
                    index_to_remove = i
                    np.delete(catalogs, i)
                    continue
                else:
                    empty_idx = [i for i, arr in enumerate(c['phot_filterset']) if arr.size == 0]
                    if empty_idx:
                        for i in empty_idx:
                            try:
                                del c['cid_filterset'][i]
                                del c['phot_filterset'][i]
                                del c['ra_filterset'][i]
                                del c['dec_filterset'][i]
                                del c['photerr_filterset'][i]
                                del c['mass_filterset'][i]
                                del c['age_filterset'][i]
                                del c['mass_min_filterset'][i]
                                del c['mass_max_filterset'][i]
                                del c['age_min_filterset'][i]
                                del c['age_max_filterset'][i]
                                del c['filtersets'][i]
                                del c['filtersets_detect'][i]
                                del c['class_filterset'][i]
                                del c['detect_filterset'][i]
                                del c['libcomp'][i]
                            except:
                                print(f'emptyindex,{empty_idx}')
                                print(f'catalog {c}')
                                cp = c['phot_filterset'] 
                                print(f'filterset {cp}')
                
        for i, c in enumerate(catalogs):
            cids_empty = []
            # Remove empty arrays from phot_filterset and related fields
            empty_idx = [i for i, arr in enumerate(c['libcomp']) if arr.any() == 0]
            if empty_idx:
                for eid in empty_idx:
                    cids_empty.append(c['cid_filterset'][eid])
            try:
                cids_to_delete = np.concatenate(cids_empty)
            except:
                continue
            for i in empty_idx:
                try:
                    del c['cid_filterset'][i]
                    del c['phot_filterset'][i]
                    del c['dec_filterset'][i]
                    del c['ra_filterset'][i]
                    del c['photerr_filterset'][i]
                    del c['mass_filterset'][i]
                    del c['age_filterset'][i]
                    del c['mass_min_filterset'][i]
                    del c['mass_max_filterset'][i]
                    del c['age_min_filterset'][i]
                    del c['age_max_filterset'][i]
                    del c['filtersets'][i]
                    del c['filtersets_detect'][i]
                    del c['class_filterset'][i]
                    del c['detect_filterset'][i]
                    del c['libcomp'][i]
                except:
                    print(f'emptyindex,{empty_idx}')
                    print(f'catalog {c}')
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
                c['mass'] = c['mass'][idx_keep]
                c['age'] = c['age'][idx_keep]
                c['mass_min'] = c['mass_min'][idx_keep]
                c['mass_max'] = c['mass_max'][idx_keep]
                c['age_min'] = c['age_min'][idx_keep] 
                c['age_max'] = c['age_max'][idx_keep]

    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl
# from LEGUS_comp_div import libcomp 
def clean_legus_low_reff(catalogs, verbose):
    """
    This cleans up LEGUS catalogs, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # First identify clusters that nominally have zero completeness;
    # remove them from both the sorted and original photometry lists
    for c in catalogs:
        ndel = 0
        for i in range(len(c['phot_filterset'])):
            comp_class = completeness_LEGUS_low_reff(c['galaxy'])
            comp = comp_class.LEGUS_cal(c['phot_filterset'][i],c['filtersets'][i], c['mass_filterset'][i])
            idx_del = comp == 0.0
            idx_keep = comp > 0.
            cids_to_delete = c['cid_filterset'][i][idx_del]
            c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
            c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
            c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
            c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
            c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
            c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
            c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
            c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
            c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
            c['ra_filterset'][i] = c['ra_filterset'][i][idx_keep]
            c['dec_filterset'][i] = c['dec_filterset'][i][idx_keep]

            ndel += len(cids_to_delete)
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
                c['mass'] = c['mass'][idx_keep]
                c['age'] = c['age'][idx_keep]
                c['mass_min'] = c['mass_min'][idx_keep]
                c['mass_max'] = c['mass_max'][idx_keep]
                c['age_min'] = c['age_min'][idx_keep]
                c['age_max'] = c['age_max'][idx_keep]

                
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to completeness = 0 issue").
                  format(ndel, c['basename']))

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['mass'] = c['mass'][idx_keep]
        c['age'] = c['age'][idx_keep]
        c['mass_min'] = c['mass_min'][idx_keep]
        c['mass_max'] = c['mass_max'][idx_keep]
        c['age_min'] = c['age_min'][idx_keep]
        c['age_max'] = c['age_max'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]
        


        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['mass_filterset'][i] = c['mass_filterset'][i][idx_keep]
                c['age_filterset'][i] = c['age_filterset'][i][idx_keep]
                c['mass_min_filterset'][i] = c['mass_min_filterset'][i][idx_keep]
                c['mass_max_filterset'][i] = c['mass_max_filterset'][i][idx_keep]
                c['age_min_filterset'][i] = c['age_min_filterset'][i][idx_keep]
                c['age_max_filterset'][i] = c['age_max_filterset'][i][idx_keep]
                c['ra_filterset'][i] = c['ra_filterset'][i][idx_keep]
                c['dec_filterset'][i] = c['dec_filterset'][i][idx_keep]

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
        for i, c in enumerate(catalogs):
            # Remove empty arrays from phot_filterset and related fields
                if c['cid'].size ==0:
                    index_to_remove = i
                    np.delete(catalogs, i)
                    continue
                else:
                    empty_idx = [i for i, arr in enumerate(c['phot_filterset']) if arr.size == 0]
                    if empty_idx:
                        for i in empty_idx:
                            try:
                                del c['cid_filterset'][i]
                                del c['phot_filterset'][i]
                                del c['ra_filterset'][i]
                                del c['dec_filterset'][i]
                                del c['photerr_filterset'][i]
                                del c['mass_filterset'][i]
                                del c['age_filterset'][i]
                                del c['mass_min_filterset'][i]
                                del c['mass_max_filterset'][i]
                                del c['age_min_filterset'][i]
                                del c['age_max_filterset'][i]
                                del c['filtersets'][i]
                                del c['filtersets_detect'][i]
                                del c['class_filterset'][i]
                                del c['detect_filterset'][i]
                                del c['libcomp'][i]
                            except:
                                print(f'emptyindex,{empty_idx}')
                                print(f'catalog {c}')
                                cp = c['phot_filterset'] 
                                print(f'filterset {cp}')
                
        for i, c in enumerate(catalogs):
            cids_empty = []
            # Remove empty arrays from phot_filterset and related fields
            empty_idx = [i for i, arr in enumerate(c['libcomp']) if arr.any() == 0]
            if empty_idx:
                for eid in empty_idx:
                    cids_empty.append(c['cid_filterset'][eid])
            try:
                cids_to_delete = np.concatenate(cids_empty)
            except:
                continue
            for i in empty_idx:
                try:
                    del c['cid_filterset'][i]
                    del c['phot_filterset'][i]
                    del c['dec_filterset'][i]
                    del c['ra_filterset'][i]
                    del c['photerr_filterset'][i]
                    del c['mass_filterset'][i]
                    del c['age_filterset'][i]
                    del c['mass_min_filterset'][i]
                    del c['mass_max_filterset'][i]
                    del c['age_min_filterset'][i]
                    del c['age_max_filterset'][i]
                    del c['filtersets'][i]
                    del c['filtersets_detect'][i]
                    del c['class_filterset'][i]
                    del c['detect_filterset'][i]
                    del c['libcomp'][i]
                except:
                    print(f'emptyindex,{empty_idx}')
                    print(f'catalog {c}')
            for cid in cids_to_delete: 
                idx_keep = c['cid'] != cid
                c['cid'] = c['cid'][idx_keep]
                c['phot'] = c['phot'][idx_keep]
                c['photerr'] = c['photerr'][idx_keep]
                c['detect'] = c['detect'][idx_keep]
                c['filtersets_index'] = c['filtersets_index'][idx_keep]
                c['ra'] = c['ra'][idx_keep]
                c['dec'] = c['dec'][idx_keep]
                c['mass'] = c['mass'][idx_keep]
                c['age'] = c['age'][idx_keep]
                c['mass_min'] = c['mass_min'][idx_keep]
                c['mass_max'] = c['mass_max'][idx_keep]
                c['age_min'] = c['age_min'][idx_keep] 
                c['age_max'] = c['age_max'][idx_keep]

    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl

def clean_legus_simple(catalogs, verbose):
    """
    This cleans up LEGUS catalogs using the minimum requirements, getting the data into a uniform
    state suitable for Bayesian analysis

    Parameters
       catalogs : list
          catalogs prepared by the analyze_catalog script
       verbose : bool
          print verbose output or not

    Returns
       nothing
    """

    # Legacy allcomp/*.npy completeness removed; use clean_legus(..., nn_dir=...).

    # Second remove clusters that are duplicated across catalogs
    for i, c in enumerate(catalogs):
        # Compare every RA and DEC in every other catalog to find
        # duplicates
        dup = np.zeros(len(c['ra']), dtype=bool)
        for c2 in catalogs[i+1:]:
            ra_match = np.equal.outer(c['ra'], c2['ra'])
            dec_match = np.equal.outer(c['dec'], c2['dec'])
            dup_cat = np.logical_or.reduce(
                np.logical_and(ra_match, dec_match), axis=1)
            dup = np.logical_or(dup, dup_cat)

        # Delete duplicates
        idx_keep = np.logical_not(dup)
        cids_to_delete = c['cid'][dup]
        ndel = len(cids_to_delete)
        c['cid'] = c['cid'][idx_keep]
        c['class'] = c['class'][idx_keep]
        c['phot'] = c['phot'][idx_keep]
        c['photerr'] = c['photerr'][idx_keep]
        c['detect'] = c['detect'][idx_keep]
        c['filtersets_index'] = c['filtersets_index'][idx_keep]
        c['ra'] = c['ra'][idx_keep]
        c['dec'] = c['dec'][idx_keep]

        for cid in cids_to_delete:
            for i in range(len(c['filtersets'])):
                idx_keep = c['cid_filterset'][i] != cid
                c['cid_filterset'][i] = c['cid_filterset'][i][idx_keep]
                c['phot_filterset'][i] = c['phot_filterset'][i][idx_keep]
                c['photerr_filterset'][i] = c['photerr_filterset'][i][idx_keep]
                c['class_filterset'][i] = c['class_filterset'][i][idx_keep]
                c['detect_filterset'][i] = c['detect_filterset'][i][idx_keep]
                

        # Print message
        if verbose and ndel > 0:
            print(("   [clean_legus]: removed {:d} clusters from "
                   "catalog {:s} due to duplication").
                  format(ndel, c['basename']))
    # Recalculate total number of clusters and return
    ncl = 0
    for c in catalogs:
        ncl += len(c['phot'])
    return ncl