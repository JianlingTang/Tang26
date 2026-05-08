"""
This script finds the cluster population parameters using truncated model for catalog.
"""

import argparse
import copy
import glob
import os
import os.path as osp
import sys
import time
import re
from collections import namedtuple

import emcee
import numexpr as ne
import numpy as np
from numpy.random import rand
from slugpy import read_cluster, slug_pdf
from slugpy.cluster_slug import cluster_slug

from catalog_readers import reader_register
from clean_legus import clean_legus, clean_legus_comp
from completeness_calculator import comp_register
from completeness_io import (
    _nn_scaler_model_paths,
    legus_nn_missing_uv_u_fills,
    predict_catalog_completeness_with_nn,
)
from hybrid_libcomp import (
    HybridLegusLibCompletenessCalculator,
    format_hybrid_libcomp_summary,
    legus_catalog_abs_bounds,
)

# Parse the inputs
parser = argparse.ArgumentParser(
    description="Script to find best fit cluster population "
    "parameters for star cluster catalogs",
)
parser.add_argument(
    "libdir",
    help="cluster_slug library path passed to read_cluster() (directory or "
    "library prefix as required by slugpy for your build)",
)
parser.add_argument("massPDF",
                    help="name of the mass PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("agePDF",
                    help="name of the age PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("AVPDF",
                    help="names of the AV PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("catalogs", nargs="*", default=None,
                    help="catalog files to be processed (optional when --galaxy-names is used)")
parser.add_argument(
    "--galaxy-names",
    nargs="+",
    default=None,
    help="optional LEGUS galaxy names; auto-discovers catalogs under "
    "--legus-cct-root/<galaxy>/<catalog-glob>",
)
parser.add_argument(
    "--catalog-glob",
    default="hlsp_legus*{galaxy_name}*.tab",
    help="glob pattern used with --galaxy-names; supports {galaxy_name} placeholder",
)
parser.add_argument(
    "--nn-comp-dir",
    "--nn-dir",
    default=None,
    dest="nn_comp_dir",
    help="directory with NN completeness artifacts (glob by galaxy name); "
    "optional if --nn-scaler and --nn-model are set",
)
parser.add_argument(
    "--legus-cct-root",
    default="/g/data/jh2/jt4478/make_LEGUS_CCT",
    help="root for LEGUS CCT ancillary files (galaxy_names.npy, etc.); "
    "sets LEGUS_CCT_ROOT for catalog_readers",
)
parser.add_argument(
    "--legus-tab-dir",
    default="/home/100/jt4478/slugfiles/LEGUS_cat",
    help="directory for legacy LEGUS readme fallbacks; sets LEGUS_TAB_DIR",
)
parser.add_argument(
    "--cluster-slug-lib-dir",
    default="/g/data/jh2/jt4478/cluster_slug",
    help="cluster_slug parent directory for env CLUSTER_SLUG_LIB_DIR (informational; "
    "read_cluster uses positional libdir)",
)
parser.add_argument(
    "--output-mcmc-chains-dir",
    default="/g/data/jh2/jt4478/output_mcmc_chains",
    help="directory for output .h5 chain when -o is relative or unset basename only",
)
parser.add_argument(
    "--nn-scaler",
    default=None,
    help="explicit path to photometry scaler (pickle or joblib, e.g. StandardScaler)",
)
parser.add_argument(
    "--nn-model",
    default=None,
    help="explicit path to Torch completeness model (.pt checkpoint)",
)
parser.add_argument("--comp-threshold", type=float, default=0.01,
                    help="minimum expected completeness for keeping observed "
                    "clusters during clean_legus")
parser.add_argument(
    "--lib-vmag-max",
    type=float,
    default=-6.0,
    help="force library completeness to 0 for rows with absolute V-band "
    "magnitude > this cutoff (set to a very large value to effectively disable)",
)
parser.add_argument(
    "--hybrid-libcomp",
    default=True,
    action="store_true",
    help="use HybridLegusLibCompletenessCalculator (5D ABS box from catalog + per-band "
    "rules + NN) instead of raw predict_catalog_completeness_with_nn for library rows",
)
parser.add_argument(
    "--no-hybrid-libcomp",
    dest="hybrid_libcomp",
    action="store_false",
    help="disable hybrid libcomp path and use direct NN prediction for library rows",
)
parser.add_argument(
    "--hybrid-range-margin",
    type=float,
    default=0.0,
    help="expand catalog ABS [lo,hi] per band by this many mag for the 5D box (hybrid only)",
)
parser.add_argument(
    "--hybrid-min-bands-nonzero",
    type=int,
    default=4,
    help="hybrid rule: min bands with per-band flag 1 (capped to filterset size; hybrid only)",
)
parser.add_argument(
    "--hybrid-nn-batch-rows",
    type=int,
    default=65536,
    help="batch size for joint NN inside hybrid calculator (hybrid only)",
)
parser.add_argument("-ct", "--cattype", default="LEGUS",
                    help="type of input catalog; currently "
                    "known values are 'mock' and 'LEGUS' and 'LEGUS_rOGC'; this "
                    "parameter specifies both the reader used and "
                    "how the completeness is calculated")
parser.add_argument("-o", "--outname", default=None,
                    help="name of output file; default is first "
                    "catalog file with extension changed to .h5")
parser.add_argument("-p", "--photsystem", default="Vega",
                    help="photometric system used in the catalogs")
parser.add_argument("--tol", type=float, default=1.0e-2,
                    help="tolerance to use when evaluating the "
                    "likelihood function")
parser.add_argument("--bwphot", type=float, default=0.1,
                    help="photometric bandwidth")
parser.add_argument("--bwphys", type=float, default=0.1,
                    help="physical bandwidth")
parser.add_argument("-nw", "--nwalkers", type=int, default=100,
                    help="number of walkers to use in the MCMC")
parser.add_argument("-ni", "--niter", type=int, default=500,
                    help="number of MCMC iterations")
parser.add_argument("-mdd", "--mdd", default=False,
                    action="store_true",
                    help="fit to mass-dependent disruption model"
                    " instead of mass-independent disruption model"
                    " (default)")
parser.add_argument("--restart", default=False, action="store_true",
                    help="restart from existing output file")
parser.add_argument("-nav", "--nav", type=int, default=6,
                    help="number of intervals to use in approximating "
                    "the A_V distribution")
parser.add_argument("-nparam", "--nparam", type=int, default=4,
                    help="number of parameters needed in model "
                    "despite the A_V distribution")
parser.add_argument("-c", "--c", default=False,
                    action='store_true',
                    help="use 628c or 628e")
parser.add_argument("-pl", "--pl", default=False,
                    action='store_true',
                    help="use pure powerlaw model")
parser.add_argument("-trun", "--trun", default=False,
                    action='store_true',
                    help="use the truncated model")
parser.add_argument("-nl", "--nlib", type=int, default=-1,
                    help="if set to a positive value, this causes "
                    "the analysis to use only the first nlib clusters "
                    "in the library")
parser.add_argument("-v", "--verbose", default=False,
                    action='store_true',
                    help="produce verbose output")
args = parser.parse_args()

os.environ["LEGUS_CCT_ROOT"] = args.legus_cct_root
os.environ["LEGUS_TAB_DIR"] = args.legus_tab_dir
os.environ["CLUSTER_SLUG_LIB_DIR"] = args.cluster_slug_lib_dir
os.environ["CLUSTER_SLUG_LIB_NAME"] = args.libdir

if (args.nn_scaler is None) ^ (args.nn_model is None):
    parser.error("--nn-scaler and --nn-model must be given together.")
if args.nn_scaler is None and args.nn_comp_dir is None:
    parser.error("Provide --nn-comp-dir (--nn-dir), or both --nn-scaler and --nn-model.")

# Optional catalog discovery for multi-galaxy runs (similar to analyze_all.py)
discovered_catalogs = []
if args.galaxy_names:
    for gal in args.galaxy_names:
        gal_key = gal.split("_")[0]
        cat_dir = osp.join(args.legus_cct_root, gal_key)
        pattern = args.catalog_glob.replace("{galaxy_name}", gal_key)
        matches = sorted(glob.glob(osp.join(cat_dir, pattern)))
        if not matches:
            parser.error(
                f"No LEGUS catalog found for galaxy '{gal}' in {cat_dir}; "
                f"expected pattern {pattern}"
            )
        discovered_catalogs.append(matches[0])

args.catalogs = discovered_catalogs + (args.catalogs or [])
if len(args.catalogs) == 0:
    parser.error("Provide catalog paths, or use --galaxy-names for auto-discovery.")


################################
# Helper classes and functions #
################################

# Define a class that is built from the sampling PDFs, and has a
# function that takes as input an array of physical properties from
# cluster_slug and returns the sample density at that set of physical
# properties. We will use this to tell cluster_slug the sampling
# density of our library. Note that there is a minor subtlety in that
# cluster_slug works on log mass and log age, while the PDFs we have
# just read are PDFs on mass and age; we multiply by a factsor of m * t
# to correct for this.
class sample_den(object):
    def __init__(self, mpdf, tpdf, avpdf):
        self.mpdf = mpdf
        self.tpdf = tpdf
        self.avpdf = avpdf
    def sample_den(self, physprop):
        m = 10.**physprop[:,0]
        # Fix out of range errors due to mass-limited sampling
        m[m < self.mpdf.bkpts[0]] = self.mpdf.bkpts[0]
        m[m > self.mpdf.bkpts[-1]] = self.mpdf.bkpts[-1]
        t = 10.**physprop[:,1]
        av = physprop[:,2]
        sden = self.mpdf(m) * self.tpdf(t) * self.avpdf(av) * m * t
        return sden


# Class used to set the weights on the library; there are effectively
# two versions, one for mass-independent disruption and one for
# mass-dependent disruption, but we combine them into the same class
# for simplicity
class libwgts(object):
    def __init__(self, p, mid = not args.mdd): # for powerlaw 
        self.alphaM = p[0]
        self.mBreak = 10.**p[1]
        self.mid = mid
        if self.mid:
            self.alphaT = p[2]
            self.tMid   = 10.**p[3]
        else:
            self.gammaMdd = p[2]
            self.tMddMin  = 10.**p[3]
        self.nav       = len(p) - 4 
        self.delta_av = 3.0/self.nav
        self.av = np.arange(0, 3.0+self.delta_av/2.0, self.delta_av)
        self.pav = np.zeros(self.nav+1)
        self.pav[:-1] = 10.**p[4:]
        self.pav[-1] = 2.0/self.delta_av - self.pav[-2] \
                       - np.sum(self.pav[:-2]+self.pav[1:-1])
        # print('pav last is', self.pav[:-1])
    def wgts(self, physprop):
        # Note: need to make local, non-sliced references in order to
        # use numexpr

        # Inputs
        logm = physprop[:,0]
        logt = physprop[:,1]
        av = physprop[:,2]

        # Stored parameters:
        alphaM = self.alphaM
        mBreak = self.mBreak
        if self.mid:
            alphaT = self.alphaT
            tMid = self.tMid
            logtMid = np.log10(tMid)
        else:
            gammaMdd = self.gammaMdd
            tMddMin = self.tMddMin

        # Get weight for M, T distributions
        if self.mid:
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*"
                "exp(-10.**logm/mBreak) * "
                "where( logt <= logtMid, "
                "       10.**logt/tMid, "
                "       (10.**logt/tMid)**(alphaT+1) )")
        else:
            eta = ne.evaluate(
                "(1.0 + gammaMdd*(100.0/10.**logm)**gammaMdd"
                " * 10.**logt/tMddMin)**(1.0/gammaMdd)")
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*"
                "eta**(alphaM+1.0-gammaMdd)*"
                "exp(-10.**logm*eta/mBreak)*"
                "10.**logt")

        # Add A_V weights

        for i in range(self.nav):
            avlo = self.av[i]
            avhi = self.av[i+1]
            pavlo = self.pav[i]
            pavhi = self.pav[i+1]
            avslope = (pavhi-pavlo) / (avhi-avlo)
            wgt = ne.evaluate(
                "wgt * where( (av >= avlo) & (av < avhi),"
                "             pavlo + (av-avlo)*avslope,"
                "             1.0 )"
            )

        # Return final result
        return wgt

    
# Function to return log likelihood for mass-independent disruption
evalctr = 0
def lnprob(params):

    global evalctr

    # Parameters are, in order:
    # 0 = alphaM
    # 1 = log10(mBreak)
    # 2 = alphaT (if args.mdd is False) or gammaMdd (otherwise)
    # 3 = log10(tMid) (if args.mdd is True) or log10(tMddMin) (otherwise)
    # 4 ... 6+args.nav = log p(A_V) at points distributed at A_V = 0 - 3 mag

    # Start clock
    if args.verbose:
        tstart = time.time()
        twgt = tstart
   
    # Construct wgts object
    wgts = libwgts(params, mid = not args.mdd)

    # Enforce limits
    logL = 0.0
    if params[0] < -3.0 or \
    params[0] > 0.0 or \
    params[1] < 2.0 or \
    params[1] > 8.0 : 
        # alpha_M in [-4, 0], log m_Break in [2, 8]
        logL = -np.inf
    if not args.mdd:
        if params[2] < -3.0 or \
        params[2] > 0.0 or \
        params[3] < 5.0 or \
        params[3] > 10.0:
            # alpha_T in [-3, 0], log T_mid in [5, 10]
            logL = -np.inf
    else:
        if params[2] <= 0.0 or \
        params[2] > 1.0 or \
        params[3] < 4.0 or \
        params[3] > 10.0 :
            # gamma_mdd in (0, 1], log T_mdd,min in [4, 10]
            logL = -np.inf
    if wgts.pav[-1] < 0.0:
        # p(A_V) > 0 everywhere
        logL = -np.inf

    # Evaluate unless we're out of bounds
    if logL == 0.0:
        # Adjust catalog weights for this set of parameters
        for cat in catalogs:
            for cs in cat['cs']:
                cs.priors = wgts.wgts

        # Stop timer for application of weights
        if args.verbose:
            twgt = time.time()

        # Loop over catalogs and filter sets, adding contribution of
        # each to likelihood function
        for cat in catalogs:
            for cs, phot, photerr in zip(cat['cs'], 
                                        cat['phot_filterset'],
                                        cat['photerr_filterset']):
                print(len(phot),' is the # of cluster of this filterset.')
                if len(phot[:,0]) < 1:
                    print('skipping this loop with zero cluster.')
                    continue 
                else :
                    logL += np.sum(
                    cs.logL(None, phot,
                            photerr=photerr,
                            margindim=range(3)))
        # Stop clock
        if args.verbose:
            tend = time.time()
            evalctr += 1
            print(("lnprob evaluation {:d} "
                "completed in {:f} sec (reweighting = {:f} sec); "
                "input paramters are "
                "{:s}, logL = {:f}").format(
                    evalctr,
                    tend - tstart,
                    twgt - tstart,
                    repr(params),
                    logL))

    # Return log likelihood
    return logL

        



###############
# Main script #
###############


# Catalog list is validated above (explicit catalogs and/or --galaxy-names).
# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
ncl = 0
for cat in args.catalogs:

    # Read the data
    if args.cattype == "mock":
        data = reader_register['mock'].read(cat)
    elif args.cattype == "LEGUS":
        data = reader_register['LEGUS'].read(cat)
    elif args.cattype == "LEGUS_rOGC":
        data = reader_register['LEGUS_rOGC'].read(cat)
    else:
        raise ValueError("unknown catalog type {:s}".format(args.cattype))

    # Add filters used in this catalog to global list of filters
    for f in data["filters"]:
        if not f in allfilters:
            allfilters.append(f)

    # Construct the list of all combinations of filters found in this
    # catalog; for each cluster, assign it to one of the filter sets
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(data["phot"]))
    for i, d in enumerate(data["detect"]):
        f = list(np.array(data["filters"])[d])
        if f not in filtersets:
            filtersets.append(f)
            filtersets_detect.append(np.copy(d))
        fset[i] = filtersets.index(f)
    data['filtersets'] = filtersets
    data['filtersets_index'] = fset
    data['filtersets_detect'] = filtersets_detect

    
    cid_filterset = []
    phot_filterset = []
    photerr_filterset = []
    for i, d in enumerate(data['filtersets_detect']):
        idx = data['filtersets_index'] == i
        cid_filterset.append(data['cid'][idx])
        phot_filterset.append(data['phot'][idx][:,d])
        photerr_filterset.append(data['photerr'][idx][:,d])
    data['cid_filterset'] = cid_filterset
    data['phot_filterset'] = phot_filterset
    data['photerr_filterset'] = photerr_filterset
    
    if args.cattype == 'LEGUS_rOGC':
        data['comp_filterset']= comp_register['LEGUS'](data['path'],data['phot_filterset']).comp_LEGUS()

    # Increment total number of clusters
    ncl = ncl + len(data["phot"])
    catalogs.append(data)

# For LEGUS data we need to do some cleanup: (1) remove clusters that
# appear in more than one catalog; (2) remove clusters that nominally
# have zero chance of being observed, which can get into the sample
# anyway due to peculiarities of the way that the magnitude limit for
# visual classification was combined with the aperture correction
if args.cattype == "LEGUS":
    ncl_before_clean = sum(len(cat["phot"]) for cat in catalogs)
    if args.verbose:
        print(f"[clean_legus] comp_threshold={args.comp_threshold:.4f}")
        print(f"[clean_legus] total clusters before cleaning: {ncl_before_clean}")
        for cat in catalogs:
            print(f"[clean_legus] before {cat['basename']}: {len(cat['phot'])}")
    ncl = clean_legus(
        catalogs,
        args.verbose,
        nn_dir=args.nn_comp_dir,
        nn_scaler_path=args.nn_scaler,
        nn_model_path=args.nn_model,
        comp_threshold=args.comp_threshold,
        enforce_hybrid_criteria=bool(args.hybrid_libcomp),
        lib_vmag_max=float(args.lib_vmag_max),
        min_bands_nonzero=int(args.hybrid_min_bands_nonzero),
    )
    ncl_after_clean = sum(len(cat["phot"]) for cat in catalogs)
    print(f"[clean_legus] total clusters before cleaning: {ncl_before_clean}")
    print(f"[clean_legus] total clusters after cleaning : {ncl_after_clean}")
    for cat in catalogs:
        print(f"[clean_legus] after  {cat['basename']}: {len(cat['phot'])}")
elif args.cattype == "LEGUS_rOGC":
    ncl = clean_legus_comp(catalogs, args.verbose)

# We're now done ingesting the input catalogs; print status if verbose
if args.verbose:
    print("Completed reading the following input catalogs:")
    for cat in catalogs:
        print("   "+cat["basename"]+":")
        print("      {:d} clusters".format(len(cat["phot"])))
        print("      filters: {:s}".format(repr(cat["filters"])))
        print("      filter combinations:")
        for p, f in zip(cat['phot_filterset'], cat['filtersets']):
            print("         {:s} ({:d} clusters)".format(
                repr(f), len(p)))

if args.verbose:
    print("Completed reading the classifications of clusters of the following input catalogs:")
    for cat in catalogs:
        print("   "+cat["basename"]+":")
        print("      filter combinations:")
        for p, f in zip(cat['phot_filterset'], cat['filtersets']):
            print("         {:s} ({:d} clusters)".format(
                repr(f), len(p)))

# Read the PDF files that were used to generate cluster slug catalog,
# so that we can set the sample density
mass_pdf = slug_pdf(args.massPDF)
age_pdf = slug_pdf(args.agePDF)
av_pdf = slug_pdf(args.AVPDF)
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)

# Read the slug library
if args.verbose:
    print("Loading cluster_slug library data")
lib_all = read_cluster(
    args.libdir,
    photsystem=args.photsystem,
    read_filters=allfilters,
)  # all filters used in catalog data

# Save memory by extracting the fields we need and deleting the rest
cid = lib_all.id
actual_mass = lib_all.actual_mass
form_time = lib_all.form_time
eval_time = lib_all.time
A_V = lib_all.A_V
phot_neb_ex = lib_all.phot_neb_ex
filter_names = lib_all.filter_names
filter_units = lib_all.filter_units
del lib_all


# Compute observational completeness of library clusters for all
# catalogs and filter sets
ncl_init = len(actual_mass)
keep = np.zeros(ncl_init, dtype=np.bool)

v_filter_candidates = ("ACS_F555W", "WFC3_UVIS_F555W")
v_lib_idx = None
lib_filter_names = [str(f) for f in filter_names]
for fname in v_filter_candidates:
    if fname in lib_filter_names:
        v_lib_idx = lib_filter_names.index(fname)
        break
if v_lib_idx is None:
    for i, fname in enumerate(lib_filter_names):
        if str(fname).endswith("F555W"):
            v_lib_idx = i
            break
faint_v_mask = None
if v_lib_idx is not None:
    v_abs = phot_neb_ex[:, v_lib_idx]
    faint_v_mask = np.isfinite(v_abs) & (v_abs > args.lib_vmag_max)
    if args.verbose:
        print(
            f"[nn-libcomp] V-band cutoff active: filter={lib_filter_names[v_lib_idx]}, "
            f"cut={args.lib_vmag_max:.3f}, affected_rows={int(np.sum(faint_v_mask))}"
        )

for cat in catalogs:
    cat['libcomp'] = []
    for d in cat['filtersets_detect']:
        galaxy_fullname = cat.get("galaxy_fullname", cat["basename"])
        requested_filters = [str(f) for f in np.array(cat["filters"])[d]]
        # Derive NN input order from the catalog filterset by wavelength
        # (e.g., F275W < F336W < F435W < F555W < F814W), without hardcoding
        # specific filter names.
        def _filt_wave_key(filt):
            match = re.search(r"F(\d+)W", str(filt))
            if match is not None:
                return (0, int(match.group(1)), str(filt))
            return (1, lib_filter_names.index(str(filt)), str(filt))

        subset_filters = sorted(requested_filters, key=_filt_wave_key)

        def _filt_wave_key_lib(filt):
            match = re.search(r"F(\d+)W", str(filt))
            if match is not None:
                return (0, int(match.group(1)), str(filt))
            return (1, lib_filter_names.index(str(filt)), str(filt))

        all_cat_filters = [str(f) for f in cat["filters"]]
        if set(subset_filters) < set(all_cat_filters):
            nn_full_order = sorted(all_cat_filters, key=_filt_wave_key_lib)
        else:
            nn_full_order = subset_filters
        lib_indices = [lib_filter_names.index(f) for f in subset_filters]
        # phot_neb_ex is read as absolute magnitude; NN was trained on apparent
        # magnitude, so shift by this catalog's distance modulus first.
        dmod = float(cat.get("dmod")) #TODO: If dmod is not set, raise an error instead of silently using 0.0, which will lead to incorrect completeness calculations.
        if dmod is None:
            raise ValueError(f"Distance modulus not set for catalog {cat['basename']}")
        phot_cat_full = np.asarray(cat["phot"], dtype=float)
        detect_cat_full = np.asarray(cat["detect"], dtype=bool)
        nn_uv_u_fills = legus_nn_missing_uv_u_fills(
            cat["filters"],
            phot_cat_full,
            detect_cat_full,
            dmod,
            nn_full_order,
            subset_filters,
            use_apparent_magnitude=True,
        )
        nn_fill_kw = {"missing_band_fills": nn_uv_u_fills} if nn_uv_u_fills else {}
        if args.hybrid_libcomp:
            nn_scaler_path, nn_model_path = _nn_scaler_model_paths(
                args.nn_comp_dir,
                str(galaxy_fullname),
                args.nn_scaler,
                args.nn_model,
            )
            phot_sub = phot_cat_full[:, d]
            detect_sub = detect_cat_full[:, d]
            bounds_lo, bounds_hi = legus_catalog_abs_bounds(
                phot_sub,
                detect_sub,
                subset_filters,
                range_margin=float(args.hybrid_range_margin),
            )
            min_b = min(int(args.hybrid_min_bands_nonzero), len(subset_filters))
            full_nn_filters = [str(f) for f in cat["filters"]]
            calc = HybridLegusLibCompletenessCalculator(
                subset_filters,
                bounds_lo,
                bounds_hi,
                lib_filter_names,
                nn_scaler_path,
                nn_model_path,
                lib_vmag_max=float(args.lib_vmag_max),
                min_bands_nonzero=min_b,
                nn_full_filter_order=full_nn_filters,
                nn_batch_rows=int(args.hybrid_nn_batch_rows),
                nn_missing_band_fills=nn_uv_u_fills,
            )
            out = calc.compute(
                phot_neb_ex,
                dmod=dmod,
                galaxy_fullname=str(galaxy_fullname),
            )
            comp = out["comp_hybrid"]
            if args.verbose:
                summ = out["summary"]
                print(
                    f"[hybrid-libcomp] {cat['basename']} ({len(subset_filters)} bands) "
                    f"filterset={subset_filters!r}"
                )
                print(format_hybrid_libcomp_summary(summ))
        else:
            lib_phot_subset = phot_neb_ex[:, lib_indices] + dmod
            finite_rows = np.all(np.isfinite(lib_phot_subset), axis=1)
            if np.all(finite_rows):
                comp = predict_catalog_completeness_with_nn(
                    lib_phot_subset,
                    galaxy_fullname=galaxy_fullname,
                    nn_dir=args.nn_comp_dir,
                    subset_filters=subset_filters,
                    full_filter_order=nn_full_order,
                    nn_scaler_path=args.nn_scaler,
                    nn_model_path=args.nn_model,
                    **nn_fill_kw,
                )
            else:
                if args.verbose:
                    n_bad = int(np.sum(~finite_rows))
                    print(
                        f"[nn-libcomp] {cat['basename']} ({len(subset_filters)} bands): "
                        f"skipping {n_bad} / {len(finite_rows)} library rows with non-finite photometry"
                    )
                comp = np.zeros(len(finite_rows), dtype=float)
                if np.any(finite_rows):
                    comp[finite_rows] = predict_catalog_completeness_with_nn(
                        lib_phot_subset[finite_rows],
                        galaxy_fullname=galaxy_fullname,
                        nn_dir=args.nn_comp_dir,
                        subset_filters=subset_filters,
                        full_filter_order=nn_full_order,
                        nn_scaler_path=args.nn_scaler,
                        nn_model_path=args.nn_model,
                        **nn_fill_kw,
                    )
            if faint_v_mask is not None and np.any(faint_v_mask):
                comp[faint_v_mask] = 0.0
        cat['libcomp'].append(comp)
        # Keep only clusters that are at least mildly observable in any
        # catalog/filterset; near-zero probabilities are pruned.
        keep = np.logical_or(keep, cat['libcomp'][-1] >= args.comp_threshold)


# Prune the library of clusters for which there is 0 probability of
# the cluster being observed in any catalog and filter set
cid = cid[keep]
actual_mass = actual_mass[keep]
eval_time = eval_time[keep]
form_time = form_time[keep]
A_V = A_V[keep]
phot_neb_ex = phot_neb_ex[keep]
for cat in catalogs:
    for i in range(len(cat['libcomp'])):
        cat['libcomp'][i] = cat['libcomp'][i][keep]
if args.verbose:
    print("Pruned library from {:d} to {:d} clusters".
          format(ncl_init, len(actual_mass)))

# Initialize a cluster_slug library for every catalog and filter set
if args.verbose:
    print("Initializing cluster_slug objects for input catalogs:")
for cat in catalogs:
    if args.verbose:
        print("   {:s}:".format(cat["basename"]))
    cat['cs'] = []
    for i in range(len(cat['filtersets'])):
        if args.verbose:
            print("      filters {:s}...".
                  format(repr(cat["filtersets"][i])))
        idx_cat = []
        for j,idxx in enumerate(cat['filtersets'][i]):
            idx_cat.append(filter_names.index(idxx))
        field_list = ['id', 'actual_mass', 'time', 'form_time', 'A_V',
                      'phot_neb_ex', 'filter_names', 'filter_units']
        idx = cat['filtersets_detect'][i]
        keep = cat['libcomp'][i] >= args.comp_threshold            
        fields = [np.copy(cid[keep]), 
                  np.copy(actual_mass[keep]), 
                  np.copy(eval_time[keep]),
                  np.copy(form_time[keep]),
                  np.copy(A_V[keep]),
                  np.copy(phot_neb_ex[:,idx_cat][keep]),
                  list(np.array(cat['filters'])[idx]),
                  list(np.array(filter_units[:len(idx_cat)]))]
        print('filters are:,',list(np.array(cat['filters'])[idx]))
        print('filter units are:',list(np.array(filter_units[:len(idx_cat)])))
        lib_type = namedtuple('cluster_data', field_list)
        lib = lib_type(*fields)
        cat['cs'].append(
            cluster_slug(lib = lib,
                         sample_density = lib_den.sample_den,
                         reltol = args.tol,
                         bw_phot = args.bwphot,
                         bw_phys = args.bwphys))
        cat['cs'][-1].add_filters(lib.filter_names, 
                                  pobs=cat['libcomp'][i][keep])
        print(f'cs lib filternames are {lib.filter_names} \n')
        cat['cs'][-1].make_cache(range(3), filters=lib.filter_names)

# Construct name of output file if not specified (under --output-mcmc-chains-dir if relative)
os.makedirs(args.output_mcmc_chains_dir, exist_ok=True)
if args.outname is None:
    fname = osp.splitext(osp.basename(args.catalogs[0]))[0] + ".h5"
    outname = osp.join(args.output_mcmc_chains_dir, fname)
else:
    raw = args.outname if args.outname.endswith(".h5") else args.outname + ".h5"
    if osp.isabs(raw):
        outname = raw
    else:
        outname = osp.join(args.output_mcmc_chains_dir, osp.basename(raw))

# Set the initial walker positions; if this is a restart, read them
# from restart file, and if not start with an initial guess and
# disperse them around that
ndim = args.nparam+args.nav   # Number of free parameters in priors
p0 = np.zeros((args.nwalkers, ndim))
if args.restart:
    backend = emcee.backends.HDFBackend(outname)
    nread = backend.iteration
    if nread > 0:
        p0 = backend.get_last_sample().coords
    else:
        raise ValueError(f"Requested restart but backend has no samples: {outname}")
    if args.verbose:
        print("Loaded {:d} iterations of MCMC chain from {:s}".
              format(nread, outname))
else:
    backend = emcee.backends.HDFBackend(outname)
    backend.reset(args.nwalkers, ndim)
    nread = 0
    p0[:,0] = -2.0 + 2.0*(rand(args.nwalkers)-0.5) # alphaM = -3 to -1
    p0[:,1] = 6.0 + 2.0*(rand(args.nwalkers)-0.5) # mBreak = 5 to 7 
    if not args.mdd:
        p0[:,2] = -1.0 + 2.0*(rand(args.nwalkers)-0.5) # alphaT = -2 to 0
        p0[:,3] = 7.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmid = 6 - 8
    else:
        p0[:,2] = 0.5 + 0.5*(rand(args.nwalkers)-0.5)  # gammaMdd = 0.25 - 0.75
        p0[:,3] = 5.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmddmin = 4 - 6
    delta_av = 3.0/args.nav   # p_AV scattered uniformly in log around 1/3
    for i in range(args.nwalkers):
        pav = 1./3.*(1.0 + 0.25*(rand(args.nav)-0.5))
        integ = delta_av * (0.5*pav[0] + np.sum(pav[1:]))
        if integ >= 1.0:
            pav = 0.999/integ * pav
        p0[i,args.nparam:] = np.log10(pav)
    if args.verbose:
        print("Initialized walkers")

# Run the MCMC, saving periodically
if args.verbose:
    print("Starting MCMC")
sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob, backend=backend)
sampler.run_mcmc(p0, args.niter-nread)

