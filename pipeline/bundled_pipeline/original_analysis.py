"""
This script finds the cluster population parameters using truncated model for catalog.
"""

import argparse
import numpy as np
import numexpr as ne
import os.path as osp
import time
import copy
from collections import namedtuple
from numpy.random import seed, rand
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
import emcee
from catalog_readers import *
from completeness_calculator import *
from clean_legus import *
from datetime import date
import sys  
import os
from datetime import date

today = date.today()

# dd/mm
datetoday = today.strftime("%d_%m")


# Parse the inputs
parser = argparse.ArgumentParser(
    description="Script to find best fit cluster populuation "
    "parameters for star cluster catalogs")
parser.add_argument("galaxy_names", nargs="+", default=None,
                    help="names of galaxies to be processed")
parser.add_argument("-massPDF", "--massPDF", default="/g/data/jh2/jt4478/cluster_slug/lib_mass.pdf",
                    help="name of the mass PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("-agePDF", "--agePDF", default="/g/data/jh2/jt4478/cluster_slug/lib_time.pdf",
                    help="name of the age PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("-AVPDF", "--AVPDF", default="/g/data/jh2/jt4478/cluster_slug/lib_av.pdf",
                    help="names of the AV PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("-ct", "--cattype", default="LEGUS",
                    help="type of input catalog; currently "
                    "known values are 'mock' and 'LEGUS'; this "
                    "parameter specifies both the reader used and "
                    "how the completeness is calculated")
parser.add_argument("-outname", "--outname", default=None,
                    help="name of output file; default is first "
                    "catalog file with extension changed to .chain")
parser.add_argument("-compkeyword", "--compkeyword", default=None,
                    help="keyword to selct comp values.")
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
parser.add_argument("-subsolar", "--subsolar", default=False,
                    action="store_true",
                    help="use 0.2 solar metallcity library clusters")
parser.add_argument("--restart", default=False, action="store_true",
                    help="restart from existing output file")
parser.add_argument("-test101", "--test101", default=False,
                    action="store_true",
                    help="testm101")
parser.add_argument("-nav", "--nav", type=int, default=6,
                    help="number of intervals to use in approximating "
                    "the A_V distribution")
parser.add_argument("-nparam", "--nparam", type=int, default=4,
                    help="number of parameters needed in model "
                    "despite the A_V distribution")
parser.add_argument("-nl", "--nlib", type=int, default=-1,
                    help="if set to a positive value, this causes "
                    "the analysis to use only the first nlib clusters "
                    "in the library")
parser.add_argument("-v", "--verbose", default=False,
                    action='store_true',
                    help="produce verbose output")
args = parser.parse_args()
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
                if len(phot[:,0]) < 0.5:
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


# Exit immediately if given no input catalogs
if len(args.galaxy_names) == 0:
    exit(0)

# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
ncl = 0
for gal in args.galaxy_names:
    catpath = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT',gal)
    cat = glob.glob(os.path.join(catpath,f'hlsp_legus_hst*{gal}*avgapcor.tab'))[0]
    # Read the data
    data = reader_register['LEGUS_all'].read(cat, gal)

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


    # Organize the photometric data into groups by filter set, since
    # each will use a different cluster_slug tree to calculate its log
    # likelihood function; for each filter set, retain only the
    # photometry with detections
    cid_filterset = []
    ra_filterset = []
    dec_filterset = []
    phot_filterset = []
    photerr_filterset = []
    class_filterset = []
    detect_filterset = []
    mass_filterset = [] 
    age_filterset = []
    mass_max_filterset = []
    mass_min_filterset = []
    age_max_filterset = []
    age_min_filterset = []
    for i, d in enumerate(data['filtersets_detect']):
        idx = data['filtersets_index'] == i
        cid_filterset.append(data['cid'][idx])
        class_filterset.append(data['class'][idx])
        mass_filterset.append(data['mass'][idx])
        age_filterset.append(data['age'][idx])
        mass_max_filterset.append(data['mass_max'][idx])
        mass_min_filterset.append(data['mass_min'][idx])
        ra_filterset.append(data['ra'][idx])
        dec_filterset.append(data['dec'][idx])
        age_max_filterset.append(data['age_max'][idx])
        age_min_filterset.append(data['age_min'][idx])
        phot_filterset.append(data['phot'][idx][:,d])
        detect_filterset.append(data['detect'][idx][:,d])
        photerr_filterset.append(data['photerr'][idx][:,d])
    data['cid_filterset'] = cid_filterset
    data['phot_filterset'] = phot_filterset
    data['photerr_filterset'] = photerr_filterset
    data['class_filterset'] = class_filterset
    data['detect_filterset'] = detect_filterset
    data['ra_filterset'] = ra_filterset
    data['dec_filterset'] = dec_filterset
    data['mass_filterset'] = mass_filterset
    data['age_filterset'] = age_filterset
    data['mass_max_filterset'] = mass_max_filterset
    data['mass_min_filterset'] = mass_min_filterset
    data['age_max_filterset'] = age_max_filterset
    data['age_min_filterset'] = age_min_filterset
   
    
    # Increment total number of clusters
    ncl = ncl + len(data["phot"])
    catalogs.append(data)

# For LEGUS data we need to do some cleanup: (1) remove clusters that
# appear in more than one catalog; (2) remove clusters that nominally
# have zero change of being observed, which can get into the sample
# anyway due to peculiarities of the way that the magnitude limit for
# visual classification was combined with the aperture correction

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
            
galaxies = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy')
gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
allfilters_cam = []
for galaxy_name in args.galaxy_names:
    galaxy_name = galaxy_name.split('_')[0]
    for filt, cam in zip(gal_filters[galaxy_name][0], gal_filters[galaxy_name][1]):
        filt = filt.upper()
        cam = cam.upper()
        if cam == 'WFC3':
            cam = 'WFC3_UVIS'
        filt_string = f'{cam}_{filt}'  
        allfilters_cam.append(filt_string)   
allfilters_cam = list(set(allfilters_cam))
# Read the PDF files that were used to generate cluster slug catalog,
# so that we can set the sample density
mass_pdf = slug_pdf(args.massPDF)
age_pdf = slug_pdf(args.agePDF)
av_pdf = slug_pdf(args.AVPDF)
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)

# Read the slug library
if args.verbose:
    print("Loading cluster_slug library data")
    print('Avaliable filters are:', allfilters)
    
# lib_all = read_cluster(args.libname,
#                        photsystem=args.photsystem,
#                        read_filters=allfilters) # here all filters should have all filters used in catalog data 
libdir = '/scratch/mk27/jt4478/output_lib'
lib_all_list = []
lib_phot_files= glob.glob(os.path.join(libdir, 'tang_padova*_cluster_phot.fits'))
lib_phot_files = sorted(lib_phot_files)
allfilters_cam.sort(key=lambda x: x[-4:]) # Manually sort filter values to match the filter order of the LEGUS cluster catalogue (COL6-12)
for ilib, lib in enumerate(lib_phot_files[:2]):
    libname = lib.split('_cluster_phot.fits')[0]
    print(f'Reading library clusters from file {libname}...')
    lib_read = read_cluster(libname, read_filters=allfilters_cam)
    phot_length = np.shape(lib_read.phot_neb_ex)
    print(f'library phot shape is {phot_length}')
    lib_all_list.append(lib_read)
ncl_MIST = 9950000
cid = []
actual_mass = []
form_time = []
eval_time = []
A_V = []
phot_neb_ex = []
filter_names = lib_all_list[0].filter_names
filter_units = lib_all_list[0].filter_units

for lib_all in lib_all_list:
    cid.append(lib_all.id)
    actual_mass.append(lib_all.actual_mass)
    form_time.append(lib_all.form_time)
    eval_time.append(lib_all.time)
    A_V.append(lib_all.A_V)
    phot_neb_ex.append(lib_all.phot_neb_ex)
cid = np.concatenate(cid)[:ncl_MIST]
actual_mass = np.concatenate(actual_mass)[:ncl_MIST]
form_time = np.concatenate(form_time)[:ncl_MIST]
eval_time = np.concatenate(eval_time)[:ncl_MIST]
A_V = np.concatenate(A_V)[:ncl_MIST]
phot_neb_ex = np.concatenate(phot_neb_ex)[:ncl_MIST,:]
# Prune padova library length to match MIST library length of 9950000

print(f'The whole library length is {np.shape(phot_neb_ex)}... \n')
del lib_all_list
del lib_all

# Compute observational completeness of library clusters for all
# catalogs and filter sets
ncl_init = len(actual_mass)
test_old_comp = False

if test_old_comp:
    threscomp = 0.
    comp_dir = '/scratch/jh2/jt4478/tabulated_comp'
    keep = np.zeros(ncl_init, dtype=bool)
    for cat in catalogs:
        cat['libcomp'] = []
        for d in cat['filtersets_detect']:
            if '628-c' in cat['galaxy'] : 
                if np.sum(d) == 5:
                    # Collect all library completeness of sub library clusters 
                    comp_files = glob.glob(os.path.join(comp_dir, 'lib_padova*628c*fullcomp.npy'))
                    comp_files = sorted(comp_files)[:2]
                    comp_all = []
                    for comp_file in comp_files:
                        cf = np.load(comp_file)
                        comp_all.append(cf)
                    comp_all = np.concatenate(comp_all)[:ncl_MIST]
                    cat['libcomp'].append(comp_all)
                    keep = np.logical_or(keep,cat['libcomp'][-1] > threscomp)
                    print("Loaded library for ngc628c with all 5 bands.")
                elif np.sum(d) == 4:
                    comp_files = glob.glob(os.path.join(comp_dir, 'lib_padova*628c*noUVcomp.npy'))
                    comp_files = sorted(comp_files)[:2]
                    comp_all = []
                    for comp_file in comp_files:
                        cf = np.load(comp_file)
                        comp_all.append(cf)
                    
                    comp_all = np.concatenate(comp_all)[:ncl_MIST]
                    print(f'Shape of complenteess array is {np.shape(comp_all)}')
                    cat['libcomp'].append(comp_all)
                    keep = np.logical_or(keep, cat['libcomp'][-1] > threscomp)
                    print("Loaded library for ngc628c with no UV.")
                    
            elif '628-e' in cat['galaxy']:
                if np.sum(d) == 5:
                    # Collect all library completeness of sub library clusters 
                    comp_files = glob.glob(os.path.join(comp_dir, 'lib_padova*628e*fullcomp.npy'))
                    comp_files = sorted(comp_files)[:2]
                    comp_all = []
                    for comp_file in comp_files:
                        cf = np.load(comp_file)
                        comp_all.append(cf)
                    comp_all = np.concatenate(comp_all)[:ncl_MIST]
                    cat['libcomp'].append(comp_all)
                    keep = np.logical_or(keep,cat['libcomp'][-1] > threscomp)
                    print("Loaded library for ngc628e with all 5 bands.")
                elif np.sum(d) == 4:
                    comp_files = glob.glob(os.path.join(comp_dir, 'lib_padova*628e*noUVcomp.npy'))
                    comp_files = sorted(comp_files)[:2]
                    comp_all = []
                    for comp_file in comp_files:
                        cf = np.load(comp_file)
                        comp_all.append(cf)
                    comp_all = np.concatenate(comp_all)[:ncl_MIST]
                    cat['libcomp'].append(comp_all)
                    keep = np.logical_or(keep,cat['libcomp'][-1] > threscomp)
                    print("Loaded library for ngc628e with no UV.")
                else :
                    print('Completeness not valid')
                    raise NotImplementedError
            else:
                exit(0)
else:
    if not args.test101:
        keep_all = []
        if args.cattype == 'LEGUS':
            for cat in catalogs:
                cat['libcomp'] = []
                for filts in cat['filtersets']:
                    if 'wfc3_' in cat['basename']:
                        gal_str = cat['basename'].split('wfc3_')[1].split('_multiband')[0]
                    elif 'acs_' in cat['basename']:
                        gal_str = cat['basename'].split('acs_')[1].split('_multiband')[0]
                    else:
                        gal_str = None
                        exit('Library completeness file not found...')
                    directory = '/scratch/jh2/jt4478/tabulated_comp'
                    if not '336' in ''.join(filts).lower():
                        # glob all library completeness files 
                        pattern = f'lib{gal_str}_padova_onebatch_no_U_comp.npy'
                        lc_mean = np.load(os.path.join(directory, pattern))
                    if not '275' in ''.join(filts).lower():    
                        pattern = f'lib{gal_str}_padova_onebatch_no_UV_comp.npy'
                        lc_mean = np.load(os.path.join(directory, pattern))
                    elif '336' and '275' in ''.join(filts).lower():
                        pattern = f'lib{gal_str}_padova_onebatch_full_comp.npy'
                        lc_mean = np.load(os.path.join(directory, pattern))
                    cat['libcomp'].append(lc_mean)
                    keep_all.append(lc_mean)
                    # keep = np.logical_or(keep, cat['libcomp'][-1] > 0.0)time
    else:
        comp_keyword = {'literature': 'reff_Linden',
               'fixed': 'reff_L2pc',
               'mass_radius':'reff_mr'}
        keyword = comp_keyword[args.compkeyword]
        # example usage: libngc5457-c_padova_reff_mr_onebatch_full_comp.npy, libngc5457-se_padova_reff_L2pc_onebatch_full_comp.npy, libngc5457-se_padova_reff_reff_Linden_onebatch
        ncl_init = len(actual_mass)
        keep_all = []
        for cat in catalogs:
            cat['libcomp'] = []
            for filts in cat['filtersets']:
                if 'wfc3_' in cat['basename']:
                    gal_str = cat['basename'].split('wfc3_')[1].split('_multiband')[0]
                elif 'acs_' in cat['basename']:
                    gal_str = cat['basename'].split('acs_')[1].split('_multiband')[0]
                else:
                    gal_str = None
                    exit('Library completeness file not found...')
                directory = '/scratch/jh2/jt4478/tabulated_comp'
                if not '336' in ''.join(filts).lower():
                    # glob all library completeness files 
                    pattern = f'lib{gal_str}_padova_{keyword}_onebatch_no_U_comp.npy'
                    lc_mean = np.load(os.path.join(directory, pattern))
                if not '275' in ''.join(filts).lower():    
                    pattern = f'lib{gal_str}_padova_{keyword}_onebatch_no_UV_comp.npy'
                    lc_mean = np.load(os.path.join(directory, pattern))
                elif '336' and '275' in ''.join(filts).lower():
                    pattern = f'lib{gal_str}_padova_{keyword}_onebatch_full_comp.npy'
                    lc_mean = np.load(os.path.join(directory, pattern))
                cat['libcomp'].append(lc_mean)
                keep_all.append(lc_mean)
    keep = np.sum(keep_all, axis=0) > 0.
ncl = clean_legus_cal_comp(catalogs, args.verbose, args.test101, args.compkeyword)
print(f'Number of total clusters is {ncl}')

# sys.exit()
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

        if args.cattype == 'LEGUS_90':
            keep = cat['libcomp'][i] > 0.9
        else :
            keep = cat['libcomp'][i] > 0.
            
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

# Construct name of output file if not specified
chain_dir = '/g/data/jh2/jt4478/output_mcmc_chains'
if not args.mdd:
    outname = osp.join(chain_dir, "parametric_mid_" + args.compkeyword + '_' + args.outname)
else:
    outname = osp.join(chain_dir, "parametric_mdd_" + args.compkeyword + '_' + args.outname)



ndim = 4+args.nav   # Number of free parameters in priors
p0 = np.zeros((args.nwalkers, ndim))

if args.restart:
    nread = 0
    fp = open(outname, "r")
    chaindat = fp.read().split('\n')
    fp.close()
    while '' in chaindat:
        chaindat.remove('')
    for p, line in zip(p0, chaindat[-args.nwalkers-1:]):
        p[:] = [float(f) for f in line.split()[1:-1]]
    nread = len(chaindat) // args.nwalkers
    if args.verbose:
        print("Load iterations of MCMC chain from ".
              format(nread, outname))

else:
    print('Starting MCMC optimisation... \n')
    nread = 0
    p0[:,0] = -2.0 + 2.0*(rand(args.nwalkers)-0.5) # alphaM = -2 to -0
    p0[:,1] = 3 + 3.0*(rand(args.nwalkers)-0.5) # mBreak = 1.5 to 4.5 
    if not args.mdd:
        p0[:,2] = -0.5 + 2.0*(rand(args.nwalkers)-0.5) # alphaT = -1.5 to -0.5
        p0[:,3] = 5.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmid = 4.5 - 7.5
    else:
        p0[:,2] = 0.5 + 0.5*(rand(args.nwalkers)-0.5)  # gammaMdd = 0.25 - 0.75
        p0[:,3] = 5.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmddmin = 4 - 6
        
    delta_av = 3.0/args.nav   # p_AV scattered uniformly in log around 1/3
    for i in range(args.nwalkers):
        pav = 1./3.*(1.0 + 0.25*(rand(args.nav)-0.5))
        integ = delta_av * (0.5*pav[0] + np.sum(pav[1:]))
        if integ >= 1.0:
            pav = 0.999/integ * pav
        p0[i,4:] = np.log10(pav)
        print('p0 is',p0) 
    if args.verbose:
        print("Initialized walkers")
    # # Open empty file for output
    fp = open(outname, 'a')
    fp.close()

    
# Run the MCMC, saving periodically
if args.verbose:
    print("Starting MCMC")
sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob)
for i, result in enumerate(sampler.sample(p0, iterations=args.niter-nread)):
    position = result[0]
    lnp = result[1]
    if args.verbose:
        print("Completed {:d} / {:d} iterations, saving state"
              .format(i+nread+1, args.niter))
    fp = open(outname, "a")
    for k in range(position.shape[0]):
        fp.write("{0:4d}".format(k))
        for j in range(position.shape[1]):
            fp.write("  {:f}".format(position[k,j]))
        fp.write("   {:f}".format(lnp[k]))
        fp.write("\n")
    fp.close()

