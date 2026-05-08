#!/usr/bin/env python3
import numpy as np
import scipy as sc 
import numexpr as ne
from astropy.table import Table
import scipy
from astropy.io import fits
from slugpy.cluster_slug import cluster_slug
from catalog_readers import reader_register 
from slugpy import slug_pdf, read_cluster
import emcee
import argparse 
import os
import re 
import glob
import multiprocessing
import sys 
from multiprocessing import RawArray 

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         description="Function to compute the completenss") 
#     parser.add_argument("galaxies", default=None,
#                         help="names of galaxies to be processed.")
#     parser.add_argument("--libid", type=int, default=0,
#                         help="library id")
#     parser.add_argument("--allf", type=int, default=0,
#                         help="compute completeness for all avaliable filters (5 for LEGUS)")
#     args = parser.parse_args()
# parser.add_argument("--bid", type=int, default=0,
#                     help="batch id")
# parser.add_argument("--nt", type=int, default=100,
#                     help="number of MC trials")


###############################################################        
# Data for LEGUS fields where artificial star tests have been #
# performed; citations are given below                        #
###############################################################        
        
LEGUS_field_data = {
    
    # Data for NGC 628c from Adamo et al., 2017, ApJ, 841, 131
    'ngc628c' : {
        'filterset' : ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ],
        'phot_tab' : np.array([
            # F275W
            [20.2574257425742, 20.7590759075907, 21.2475247524752, 21.7491749174917,
            22.2508250825082, 22.7524752475247, 23.2541254125412, 23.7557755775577,
            24.2442244224422, 24.7458745874587, 25.2475247524752, 25.7491749174917],
            # F336W
            [20.25742574, 20.75907591, 21.24752475, 21.74917492, 22.25082508,
            22.75247525, 23.25412541, 23.75577558, 24.24422442, 24.74587459,
            25.24752475,25.74917492],
            # F435W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
            # F555W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
            # F814W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917]
        ]),

       
        'comp_tab' : np.array([
            # F275W
            [100,100,100,100,100,100,95.2727272727272,35.5454545454545,3.9090909090909,0,0,0],
            # F336W
           [100,100,100,100,100,100,100,97.18181818,74,17,0,0],
            # F435W
           [100,100,100,100,100,100,100,100,100,97.4545454545454,77.5454545454545,35.8181818181818],
            # F555W
           [100,100,100,100,100,100,100,100,100,100,82.7272727272727,32.8181818181818],
            # F814W
            [100,100,100,100,100,100,100,100,91.4545454545454,48.9090909090909,10.9999999999999,0.909090909090892]
         ] )
        },
    
    # Data for NGC 628e from Adamo et al., 2017, ApJ, 841, 131
    'ngc628e' : {
        'filterset' : ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ],
        'phot_tab' : np.array([
              # F275W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7557755775577,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F336W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F435W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F555W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F814W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917]

        ]),
        'comp_tab' : np.array([
            # F275W
            [100,100,100,100,100,100,100,32.8181818181818,2.54545454545451,0,0,0],
            # F336W
            [100,100,100,100,100,100,100,69.3636363636363,11.5454545454545,0,0,0],
            # F435W
            [100,100,100,100,100,100,100,100,100,95.2727272727272,91.1818181818181,30.090909090909],
            # F555W
            [100,100,100,100,100,100,100,100,100,100,82.7272727272727,32.8181818181818],     
            # F814W
            [100,100,100,100,100,100,100,100,100,59.2727272727272,15.3636363636363,0]
        ])            
    }
    }

class libcomp(object):
    def __init__(self, catname, Nt=100, allf=False, ind=None, noU=False, csdir=None, multi_lib_padova = False):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        self.allf = allf # if compute the completness contains only four filters without UV band, set False, else set True
        self.csdir = csdir # csdir indicate the directory of the cluster_slug library data, e.g. tang_phot.fits/tang_phys.fits 
        self.noU = noU
        self.ind = ind
        self.multi_lib_padova = multi_lib_padova
        self.Nt = Nt
        if self.allf :
            print('computing completenss in all filters')
        else:
            print('computing completenss in LEGUS filters without the UV band')
        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            if not self.multi_lib_padova:
                self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            else:
                None
                # libphot_list = []
                # lib_files = glob.glob('/scratch/mk27/jt4478/output_lib/tang_padova*cluster_phot.fits')
                # for lf in lib_files:
                #     libname = lf.split('_cluster_phot.fits')[0]
                #     print(f'library name is {libname}')
                #     lib = read_cluster(libname, read_filters = self.filterset).phot_neb_ex
                #     libphot_list.append(lib)
                # libphot = np.concatenate(libphot_list)
                # self.libphot = libphot

            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            if not self.multi_lib_padova:
                self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            else:
                None 
                # libphot_list = []
                # lib_files = glob.glob('/scratch/mk27/jt4478/output_lib/tang_padova_*cluster_phot.fits')
                # for lf in lib_files:
                #     libname = lf.split('_cluster_phot.fits')[0]
                #     print(f'library name is {libname}')
                #     libp = read_cluster(libname, read_filters = self.filterset).phot_neb_ex
                #     libphot_list.append(libp)
                # libphot = np.concatenate(libphot_list)
                # self.libphot = libphot
            self.dist = 9.9e6
        else:
            phot_val = np.zeros((5, 35))
            phot_comp = np.zeros((5, 35))
            if not self.multi_lib_padova:
                self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            else:
                None
                # libphot_list = []
                # lib_files = glob.glob('/scratch/mk27/jt4478/output_lib/tang_padova_*phot.fits')
                # sorted(lib_files)
                # for lf in lib_files:
                #     libphot_list.append(read_cluster(lf, read_filters = self.filterset).phot_neb_ex)
                # libphot = np.Concatenate(libphot_list)
                # self.libphot = libphot
            self.dist = 9.9e6
            gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
            gal_name =  os.path.basename(os.path.dirname(catname))
            galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
            filters = gal_filters.get(gal_name)[0]
            filters.sort()
            self.filterset = filters
            for i, filt in enumerate(filters):
                rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}/{filt}/recovery/rec*npy')
                rec = np.load(rec_files[0])
                phot_comp[i,:] = rec[1,:]
                phot_val[i,:] = np.arange(19, 26, 0.2)
            self.phot = phot_val
            self.comp = phot_comp
            # Assuming you have the README file name stored in readme_file
            readme_file = os.path.join(galdir, f"automatic_catalog_{gal_name}.readme")

            with open(readme_file, "r") as f:
                content = f.read()

            # Match aperture radius, distance modulus, and CI using regular expressions
            patterns = [
                (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
                (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
                (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
            ]

            for pattern, label in patterns:
                match = re.search(pattern, content)
                if match:
                    if "distance" in label:
                        galdist = float(match.group(2)) * 1e6
                    elif "CI" in label:
                        ci = float(match.group(1))
                    else:
                        useraperture = float(match.group(1))
                else:
                    raise FileNotFoundError(label + " not found in the readme.")
            self.dist = galdist 
                    

    def completeness(self):
        # first convert the data to absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        complist = np.zeros(self.libphot.shape)
        for i in range(len(self.filterset)):
            comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
            for j in range(len(self.libphot[:,0])):
                if min(abs_phot[i]) <= self.libphot[j,i] <= max(abs_phot[i]):
                    complist[j,i] = comp_func(self.libphot[j,i])
                elif self.libphot[j,i] < min(abs_phot[i]):
                    complist[j,i] = 1. # detection
                else:
                    complist[j,i] = 0. # non-detection 
        # Set up Markov Chain Monte Carlo experiment  
        prob_detected = np.zeros(complist.shape) # shape is the same (Nc,Nb)
        Nc = len(self.libphot[:,0]) # total number of library clusters 
        if self.allf : # If we are computing the completeness for filterset with all five bands 
            comp_l = np.zeros(Nc)
            for i in range(5):
                prob_detected[:,i] = complist[:,i]
            # start Monte Carlo Trials in series , 1000 as one batch 
            for i in range(self.Nt//100):
                d_etect = np.zeros((100, Nc, 5), dtype=bool) # set detect matrix 
                rand = np.random.rand(100,Nc) # set random matrix 
                for i in range(5):
                    d_etect[:,:,i] = rand<prob_detected[:,i]
                in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
                in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
                in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                in_all_bands = np.sum( d_etect, axis=2) >=4       # Sum over bands to get number of detections
                in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
                comp_l += np.mean(in_legus, axis=0) # This is the completeness values for cluster classes 1, 2 and 3
                del d_etect
                del rand
                del in_legus
                del in_all_bands
            comp_l = comp_l/(self.Nt//100)
        else : # If we are computing the completeness for filterset with only four filters: U (F336W), B (F435W), V (F555W), IR (F814W)
            if not self.noU:
                comp_l = np.zeros(Nc)
                prob_detected[:,0] = np.zeros(len(prob_detected[:,0]))
                for i in range(4): # Match the column number to the number of filters 
                    prob_detected[:,i+1] = complist[:,i+1]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//100):
                    d_etect = np.zeros((100, Nc, 5), dtype=bool) # set detect matrix 
                    rand = np.random.rand(100,Nc) # set random matrix 
                    for i in range(5):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    # Imposing the LEGUS criteria 
                    in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] ) 
                    in_v_and_b = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] ) 
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_v_and_b) # Need to be in V and I or V and B
                    in_four_bands = np.sum( d_etect, axis=2) >=4    # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_four_bands)    # Combine all LEGUS conditions
                    comp_l += np.mean(in_legus, axis=0)# This is the completeness values for cluster classes 1,2 and 3
                    del d_etect
                    del rand
                    del in_legus
                    del in_four_bands
            else:
                comp_l = np.zeros(Nc)
                prob_detected[:,0] = complist[:,0]
                prob_detected[:,1] = np.zeros(len(prob_detected[:,1]))
                for i in range(3): # Match the column number to the number of filters 
                    prob_detected[:,i+2] = complist[:,i+2]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//100):
                    d_etect = np.zeros((100, Nc, 5), dtype=bool) # set detect matrix 
                    rand = np.random.rand(100,Nc) # set random matrix 
                    for i in range(5):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    # Imposing the LEGUS criteria 
                    in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] ) 
                    in_v_and_b = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] ) 
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_v_and_b) # Need to be in V and I or V and B
                    in_four_bands = np.sum( d_etect, axis=2) >=4    # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_four_bands)    # Combine all LEGUS conditions
                    comp_l += np.mean(in_legus, axis=0)# This is the completeness values for cluster classes 1,2 and 3
                    del d_etect
                    del rand
                    del in_legus
                    del in_four_bands
            comp_l = comp_l/(self.Nt//100)
        # save completeness file in .npy format
        if self.allf:
            np.save('lib'+self.name+'_comp',comp_l) 
        else :
            if not self.noU:
                np.save('/g/data/jh2/jt4478/lib'+self.name+'noUV_comp',comp_l)
            else:
                np.save('/g/data/jh2/jt4478/lib'+self.name+f'noU_comp_{self.ind}',comp_l)
        return None 
    
class libcomp_LEGUS(object):
    def __init__(self, catname, Nt=10000, Nmr=20, Nstep = 100, csdir=None):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        self.csdir = csdir # csdir indicate the directory of the cluster_slug library data, e.g. tang_phot.fits/tang_phys.fits 
        self.Nt = Nt
        self.Nmr = Nmr
        self.Nstep = Nstep
        self.sigma_MR = 0.2937 # Derived from literatures mentioned in K2019 review. 
        
        # Effective radii used in completeness test. Default: 0.5pc-10pc. 
        rad_val = np.linspace(0.5 ,10., 10)
        self.rad_val = rad_val
        
        # Magnitude values used in artificial cluster test        
        phot_val = np.arange(19. ,26., 0.2)
        self.phot = phot_val
        
        # Create an empty matrix for final completeness with shape (Nfilter, Nmagnitude_bins, Neffective_radii)
        phot_comp_2d = np.zeros((5, 10, 35))
        
        # Load avaliable filters of this galaxy 
        gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
        # Get galaxy name
        gal_name =  os.path.basename(os.path.dirname(self.name))
        galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
        self.galname = gal_name
        # Iterate over camera names and filters
        filtersets = []
        filts, cameras = gal_filters.get(gal_name)
        for camera, filt in zip(cameras, filts) :
            # Capitalize the camera name
            camera = camera.upper()
            
            # If the camera name is 'WFC3', add 'UVIS' to the string
            if camera == 'WFC3':
                camera += '_UVIS'
            
            # Concatenate camera name and filter
            result_str = f'{camera}_{filt.upper()}'
            
            # Append the result to the list
            filtersets.append(result_str)
        filtersets.sort() # Manually sort filter values to match the filter order of the LEGUS cluster catalogue (COL6-12)
        filts.sort()
        print('filter set is {filtersets}')

        # Load library clusters' physical & photometric properties
        liball = read_cluster('/g/data/jh2/jt4478/cluster_slug/tang', read_filters = filtersets)
        libphot = liball.phot_neb_ex
        libmass = liball.actual_mass
        librad = 0.1415*np.log10(libmass)
        self.librad = librad
        del liball 
        del libmass 
        
        self.libphot = libphot
        self.filterset = filtersets
        # Get photometric values and completeness estimates from pre-computed artificial cluster test files 'xxxx.npy'
        for i, filt in enumerate(self.filterset):
            filtname = filts[i]
            rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}/{filt}/recovery/rec*npy')
            rec = np.load(rec_files[0])
            phot_comp_2d[i,:,:] = rec
            
        self.comp = phot_comp_2d
        # Assuming you have the README file name stored in readme_file
        readme_file = os.path.join(galdir, f"automatic_catalog_{gal_name}.readme")

        with open(readme_file, "r") as f:
            content = f.read()

        # Match aperture radius, distance modulus, and CI using regular expressions
        patterns = [
            (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
            (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
            (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
        ]

        for pattern, label in patterns:
            match = re.search(pattern, content)
            if match:
                if "distance" in label:
                    galdist = float(match.group(2)) * 1e6
                elif "CI" in label:
                    ci = float(match.group(1))
                else:
                    useraperture = float(match.group(1))
            else:
                raise FileNotFoundError(label + " not found in the readme.")
        self.dist = galdist 
                    

    def completeness(self, filts): # phot_arr - photometric values correpond to the input filters. filters - specific filterset to compute library completeness on 
        # Convert to absolute magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100. 
        phot_comp = np.zeros((self.libphot.shape[0], len(self.filterset)))

        prob_detected = np.zeros(phot_comp.shape) # shape = (N_cluster,N_filter)
        Nc = np.shape(phot_comp)[0] # Total # of library clusters 
        Nf = np.shape(phot_comp)[1] # Number of filters, note that library clusters always have 5 for various fitersets consistency 
        comp_ = np.zeros(Nc) # 1D array for final combined complentess values
        
        complist = np.zeros(self.libphot.shape)

        for i, filt in enumerate(filts):
            if (i == 0 and '275' not in filt) or (i == 1 and '336' not in filt):
                complist[:, i] = 0.  # non-detection
                continue
            points = (np.log10(self.rad_val), abs_phot)
            values =  self.comp[i, :, :] / 100.
            comp_func = scipy.interpolate.RegularGridInterpolator(points, values, method='linear', bounds_error=False, fill_value=0.0)
            
            for nt in range(self.Nmr):
                # Generating random numbers for each element
                rad_lib = self.librad
                # Adding the random component with sigma_MR for each element
                rad_with_random = rad_lib + np.random.randn(*rad_lib.shape) * self.sigma_MR
                mask = np.logical_and(min(abs_phot) <= self.libphot[:, i], self.libphot[:, i] <= max(abs_phot))
                mask_0 = np.where(self.libphot[:, i] < min(abs_phot))
                mask_1 = np.where(self.libphot[:, i] > max(abs_phot))
                comp_func_values = comp_func((np.array(rad_with_random[mask]), np.array(self.libphot[mask, i])))
                # Detect or non-detection
                complist[mask, i] += comp_func_values 
                complist[mask_0, i] = 0.
                complist[mask_1, i] = 1.

        # Normalize complist outside the loop
        complist /= self.Nmr
        
        # Set up Markov Chain Monte Carlo 
        for i in range(Nf):
            prob_detected[:,i] = complist[:,i]
        # Start Monte Carlo Trials in series , 1000 one batch 
        comp_ = 0 
        for j in range(self.Nt//self.Nstep):
            d_etect = np.zeros((self.Nstep, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(self.Nstep, Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            in_v_and_i = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4      # Sum over bands to get number of detections
            # Get correct V band index 
            try:
                v_index = np.where(np.logical_or(np.array(self.filterset) == 'ACS_F555W', np.array(self.filterset) == 'f555w'))[0][0]
                print('Visual filter index found.')
                in_legus = np.logical_and(np.logical_and(self.libphot[:,:, v_index] < -6, in_v_and_adjacent), in_all_bands)
            except:
                try:
                    v_index = np.where(np.logical_or(np.array(self.filterset) == 'ACS_F606W', np.array(self.filterset) == 'f606w'))[0][0]
                    in_legus = np.logical_and(np.logical_and(d_etect[:,:, v_index] < -6, in_v_and_adjacent), in_all_bands)
                except:
                    print('Visual filter index NOT found.')
                    exit('exiting due to missing V band value.')
                    # in_legus = np.logical_and(np.logical_and(d_etect[:, :, -2] < -6, in_v_and_adjacent), in_all_bands)
                # Combine LEGUS conditions
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands
        comp_ = comp_/(self.Nt//self.Nstep)
        # save completeness file in .npy format
        print('filts are: ', filts)
        if not '336' in ''.join(filts).lower():
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib'+self.galname+'_no_U_comp.npy', comp_)
        if not '275' in ''.join(filts).lower():    
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib'+self.galname+f'_no_UV_comp.npy', comp_)
        elif '336' and '275' in ''.join(filts).lower():
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib'+self.galname+f'_full_comp.npy', comp_)

        return None 
    
class libcomp_LEGUS_padova(object):
    def __init__(self, catname, Nt=10000, Nmr=20, Nstep = 100, csdir=None):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        self.csdir = csdir # csdir indicate the directory of the cluster_slug library data, e.g. tang_phot.fits/tang_phys.fits 
        self.Nt = Nt
        self.Nmr = Nmr
        self.Nstep = Nstep
        self.sigma_MR = 0.2937 # Derived from literatures mentioned in K2019 review. 
        
        # Effective radii used in completeness test. Default: 0.5pc-10pc. 
        rad_val = np.linspace(0.5 ,10., 10)
        self.rad_val = rad_val
        
        # Magnitude values used in artificial cluster test        
        phot_val = np.arange(19. ,26., 0.2)
        self.phot = phot_val
        
        # Create an empty matrix for final completeness with shape (Nfilter, Nmagnitude_bins, Neffective_radii)
        phot_comp_2d = np.zeros((5, 10, 35))
        
        # Load avaliable filters of this galaxy 
        gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
        # Get galaxy name
        gal_name =  os.path.basename(os.path.dirname(self.name))
        galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
        self.galname = gal_name
        # Iterate over camera names and filters
        filtersets = []
        filtersets_ = []
        filts, cameras = gal_filters.get(gal_name)
        for camera, filt in zip(cameras, filts) :
#             Capitalize the camera name
            camera = camera.upper()
            
            # If the camera name is 'WFC3', add 'UVIS' to the string
            if camera == 'WFC3':
                camera += '_UVIS'
            
            # Concatenate camera name and filter
            result_str = f'{camera}_{filt.upper()}'
            
#             Append the result to the list
            filtersets.append(result_str)
            filtersets_.append(filt)
#             filtersets.append(filt)
        filtersets.sort() # Manually sort filter values to match the filter order of the LEGUS cluster catalogue (COL6-12)
        filts.sort()
        filtersets_.sort() 
        print(filtersets)
        print(f'filter set is {filtersets}')
        self.filterset = filtersets
        self.filterset_ = filtersets_
        libdir = '/scratch/mk27/jt4478/output_lib'
        lib_phot_files= glob.glob(os.path.join(libdir, 'tang_padova*_cluster_phot.fits'))
        lib_phot_files = sorted(lib_phot_files)[:2]

        libphot_list = []
        libprop_list = []
        for i, lf in enumerate(lib_phot_files):
            libname = lf.split('_cluster_phot.fits')[0]
            print(f'Reading library clusters from {libname}. \n')
            liball = read_cluster(libname, read_filters = self.filterset)
            libphot_list.append(liball.phot_neb_ex)
            libprop_list.append(liball.actual_mass)
        self.libphot = np.concatenate(libphot_list)
        libmass = np.concatenate(libprop_list)
        librad = 0.1415*np.log10(libmass)
        self.librad = librad
        self.filterset = filtersets
        
        del liball 
        del libmass 

        # Get photometric values and completeness estimates from pre-computed artificial cluster test files 'xxxx.npy'
        for i, filt in enumerate(self.filterset):
            filtname = filts[i]
            rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}/{filt}/recovery/rec*npy')
            rec = np.load(rec_files[0])
            phot_comp_2d[i,:,:] = rec
            
        self.comp = phot_comp_2d
        # Assuming you have the README file name stored in readme_file
        readme_file = os.path.join(galdir, f"automatic_catalog_{gal_name}.readme")

        with open(readme_file, "r") as f:
            content = f.read()

        # Match aperture radius, distance modulus, and CI using regular expressions
        patterns = [
            (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
            (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
            (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
        ]

        for pattern, label in patterns:
            match = re.search(pattern, content)
            if match:
                if "distance" in label:
                    galdist = float(match.group(2)) * 1e6
                elif "CI" in label:
                    ci = float(match.group(1))
                else:
                    useraperture = float(match.group(1))
            else:
                raise FileNotFoundError(label + " not found in the readme.")
        self.dist = galdist 
                    

    def completeness(self, filts): # phot_arr - photometric values correpond to the input filters. filters - specific filterset to compute library completeness on 
        # Convert to absolute magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100. 
        phot_comp = np.zeros((self.libphot.shape[0], len(self.filterset)))

        prob_detected = np.zeros(phot_comp.shape) # shape = (N_cluster,N_filter)
        Nc = np.shape(phot_comp)[0] # Total # of library clusters 
        Nf = np.shape(phot_comp)[1] # Number of filters, note that library clusters always have 5 for various fitersets consistency 
        comp_ = np.zeros(Nc) # 1D array for final combined complentess values
        
        complist = np.zeros(self.libphot.shape)

        for i, filt in enumerate(filts):
            if (i == 0 and '275' not in filt) or (i == 1 and '336' not in filt):
                complist[:, i] = 0.  # non-detection
                continue
            points = (np.log10(self.rad_val), abs_phot)
            values =  self.comp[i, :, :] / 100.
            comp_func = scipy.interpolate.RegularGridInterpolator(points, values, method='linear', bounds_error=False, fill_value=0.0)
            
            for nt in range(self.Nmr):
                # Generating random numbers for each element
                # Adding the random component with sigma_MR for each element
                rad_with_random = rad_lib + np.random.randn(*rad_lib.shape) * self.sigma_MR
                mask = np.logical_and(min(abs_phot) <= self.libphot[:, i], self.libphot[:, i] <= max(abs_phot))
                mask_0 = np.where(self.libphot[:, i] < min(abs_phot))
                mask_1 = np.where(self.libphot[:, i] > max(abs_phot))
                comp_func_values = comp_func((np.array(rad_with_random[mask]), np.array(self.libphot[mask, i])))
                # Detect or non-detection
                complist[mask, i] += comp_func_values 
                complist[mask_0, i] = 0.
                complist[mask_1, i] = 1.

        # Normalize complist outside the loop
        complist /= self.Nmr
        
        # Set up Markov Chain Monte Carlo 
        for i in range(Nf):
            prob_detected[:,i] = complist[:,i]
        # Start Monte Carlo Trials in series , 1000 one batch 
        comp_ = 0 
        for j in range(self.Nt//self.Nstep):
            d_etect = np.zeros((self.Nstep, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(self.Nstep, Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            in_v_and_i = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4      # Sum over bands to get number of detections
            # Get correct V band index 
            try:
                v_index = np.where(np.logical_or(np.array(self.filterset) == 'ACS_F555W', np.array(self.filterset) == 'f555w'))[0][0]
                print('Visual filter index found.')
                in_legus = np.logical_and(np.logical_and(self.libphot[:,:, v_index] < -6, in_v_and_adjacent), in_all_bands)
            except:
                try:
                    v_index = np.where(np.logical_or(np.array(self.filterset) == 'ACS_F606W', np.array(self.filterset) == 'f606w'))[0][0]
                    in_legus = np.logical_and(np.logical_and(self.libphot[:,:, v_index] < -6, in_v_and_adjacent), in_all_bands)
                except:
                    print('Visual filter index NOT found.')
                    exit('exiting due to missing V band value.')
                    # in_legus = np.logical_and(np.logical_and(d_etect[:, :, -2] < -6, in_v_and_adjacent), in_all_bands)
                # Combine LEGUS conditions
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands
        comp_ = comp_/(self.Nt//self.Nstep)
        # save completeness file in .npy format
        print('filts are: ', filts)
        if not '336' in ''.join(filts).lower():
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib_padova_'+self.galname+'_no_U_comp.npy', comp_)
        if not '275' in ''.join(filts).lower():    
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib_padova_'+self.galname+f'_no_UV_comp.npy', comp_)
        elif '336' and '275' in ''.join(filts).lower():
            np.save('/g/data/jh2/jt4478/tabulated_comp/lib_padova_'+self.galname+f'_full_comp.npy', comp_)

        return None 
    
def monte_carlo_comp(args): # phot_arr - photometric values correpond to the input filters. filters - specific filterset to compute library completeness on 
    # Convert to absolute magnitude 
    # Create lis containting batch number:
    batchnum, comp_class = args
    libphot = np.frombuffer(libphot_base, dtype=np.float64).reshape(num_cluster, num_filter)
    # first convert the data to absolute Magnitude 
    abs_phot = comp_class.phot-5*(np.log10(comp_class.dist)-1) # Formula to calculate the absolute magnitude
    comp_class.comp[comp_class.comp < 0.] = 0.
    comp_class.comp[comp_class.comp > 100.] = 100.
    complist = np.zeros(libphot.shape)
    for i in range(len(comp_class.filterset)):
        comp_func = scipy.interpolate.interp1d(abs_phot[i],comp_class.comp[i]/100)
        for j in range(len(libphot[:,0])):
            if min(abs_phot[i]) <= libphot[j,i] <= max(abs_phot[i]):
                complist[j,i] = comp_func(libphot[j,i])
            elif libphot[j,i] < min(abs_phot[i]):
                complist[j,i] = 1. # detection
            else:
                complist[j,i] = 0. # non-detection 
    # Set up Markov Chain Monte Carlo experiment  
    prob_detected = np.zeros(complist.shape) # shape is the same (Nc,Nb)
    Nc = len(libphot[:,0]) # total number of library clusters 
    if comp_class.allf : # If we are computing the completeness for filterset with all five bands 
        comp_l = np.zeros(Nc)
        for i in range(5):
            prob_detected[:,i] = complist[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        d_etect = np.zeros((batchnum, Nc, 5), dtype=bool) # set detect matrix 
        rand = np.random.rand(batchnum,Nc) # set random matrix 
        for i in range(5):
            d_etect[:,:,i] = rand<prob_detected[:,i]
        in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
        in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
        in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
        in_all_bands = np.sum( d_etect, axis=2) >=4       # Sum over bands to get number of detections
        in_legus = np.logical_and(np.logical_and(libphot[:,3]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
        comp_l += np.mean(in_legus, axis=0) # This is the completeness values for cluster classes 1, 2 and 3
        del d_etect
        del rand
        del in_legus
        del in_all_bands
    else : # If we are computing the completeness for filterset with only four filters: U (F336W), B (F435W), V (F555W), IR (F814W)
        comp_l = np.zeros(Nc)
        prob_detected[:,0] = np.zeros(len(prob_detected[:,0]))
        for i in range(4): # Match the column number to the number of filters 
            prob_detected[:,i+1] = complist[:,i+1]
        # start Monte Carlo Trials in series , 1000 as one batch 
        d_etect = np.zeros((batchnum, Nc, 5), dtype=bool) # set detect matrix 
        rand = np.random.rand(batchnum,Nc) # set random matrix 
        for i in range(5):
            d_etect[:,:,i] = rand<prob_detected[:,i]
        # Imposing the LEGUS criteria 
        in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] ) 
        in_v_and_b = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] ) 
        in_v_and_adjacent = np.logical_or( in_v_and_i, in_v_and_b) # Need to be in V and I or V and B
        in_four_bands = np.sum( d_etect, axis=2) >=4    # Sum over bands to get number of detections
        in_legus = np.logical_and(np.logical_and(libphot[:,3]<-6, in_v_and_adjacent),in_four_bands)    # Combine all LEGUS conditions
        comp_l += np.mean(in_legus, axis=0)# This is the completeness values for cluster classes 1,2 and 3
        del d_etect
        del rand
        del in_legus
        del in_four_bands
    # save completeness file in .npy format
    # if comp_class.allf:
    #     np.save(os.path.join('/scratch/mk27/jt4478/output_lib', comp_class.name+'_comp'),comp_l) 
    # else :
    #     if not comp_class.noU:
    #         np.save(os.path.join('/scratch/mk27/jt4478/output_lib', comp_class.name+'noUV_comp'),comp_l) 
        # else:
        #     np.save('/g/data/jh2/jt4478/lib'+comp_class.catname+f'noU_comp_{self.ind}',comp_l)
    return comp_l
    

class libcomp_good_phot(object):
    def __init__(self, catname, csdir='/g/data/jh2/jt4478/cluster_slug/tang', Nt=1000):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        self.csdir = csdir # csdir indicate the directory of the cluster_slug library data, e.g. tang_phot.fits/tang_phys.fits 
        self.Nt = Nt
        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            self.dist = 9.9e6
        else:
            exit('completenss cannot be computed without tabulated data')

    def completeness(self, filters):
        Nt = self.Nt
        Nf = len(self.filterset)
        indices = np.where(np.in1d(self.filterset, filters))[0]
        # first convert the data to absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        # indices of filters in filterset
        # Create lib_phot array with zeros
        Nlib = np.shape(self.libphot)[0]
        lib_comp = np.zeros((Nlib, len(self.filterset)))

        for f_idx, filt in enumerate(filters):
            comp_func = scipy.interpolate.interp1d(abs_phot[indices][f_idx], self.comp[indices][f_idx]/100.)
            for ncl in range(Nlib):
                if min(abs_phot[indices][f_idx]) <= self.libphot[ncl, f_idx] <= max(abs_phot[indices][f_idx]):
                    lib_comp[ncl, indices[f_idx]] = comp_func(self.libphot[ncl, f_idx])
                elif self.libphot[ncl, f_idx] < min(abs_phot[indices][f_idx]):
                    lib_comp[ncl, indices[f_idx]]= 1.
    
        prob_detected = np.zeros(lib_comp.shape) # shape is the same (Nc,Nb)
        Nc = np.shape(lib_comp)[0] # total number of observed clusters 
        Nf = np.shape(lib_comp)[1] # number of filters 
        comp_ = np.zeros(Nc) # set up an 1D array for final combined complentess values
        
        for i in range(Nf):
            prob_detected[:,i] = lib_comp[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        for i in range(self.Nt//1000):
            d_etect = np.zeros((1000, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(1000, Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            in_v_and_i = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4      # Sum over bands to get number of detections
            in_legus = np.logical_and(np.logical_and(lib_comp[:, -2] <-6, in_v_and_adjacent),in_all_bands) # Combine LEGUS conditions
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands

        comp_ = comp_/(self.Nt//1000)
        
        return comp_
    
class libcomp_serial(object):
    def __init__(self, catname, libindex=0,  Nt=30000, padova=True):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        print(f'galaxy name is {catname}')
        self.Nt = Nt
        self.libindex = libindex
        self.padova = padova
        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6
        if self.padova:
            if self.libindex is None:
                lib_files = glob.glob(f'/scratch/mk27/jt4478/output_lib/tang_padova*_cluster_phot.fits')
            else:
                lib_files = glob.glob(f'/scratch/mk27/jt4478/output_lib/tang_padova*_{libindex}_cluster_phot.fits')
            libphot_list = []
            for lf in lib_files:
                libname = lf.split('_cluster_phot.fits')[0]
                print(f'Reading library clusters from {libname}. \n')
                lib = read_cluster(libname, read_filters = self.filterset).phot_neb_ex
                libphot_list.append(lib)
            self.libphot = np.concatenate(libphot_list)
        else:
            lib = read_cluster('/g/data/jh2/jt4478/cluster_slug/tang', read_filters = self.filterset).phot_neb_ex
            self.libphot = lib

    def completeness(self, allf=0):
        Nt = self.Nt
        Nf = len(self.filterset)
        if allf ==0 : 
            filters = self.filterset
        else:
            filters = self.filterset[1:]
            
        indices = np.where(np.in1d(self.filterset, filters))[0]
        # first convert the data to absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        # indices of filters in filterset
        # Create lib_phot array with zeros
        Nlib = np.shape(self.libphot)[0]
        lib_comp = np.zeros((Nlib, len(self.filterset)))

        for f_idx, filt in enumerate(filters):
            comp_func = scipy.interpolate.interp1d(abs_phot[indices][f_idx], self.comp[indices][f_idx]/100.)
            for ncl in range(Nlib):
                if min(abs_phot[indices][f_idx]) <= self.libphot[ncl, f_idx] <= max(abs_phot[indices][f_idx]):
                    lib_comp[ncl, indices[f_idx]] = comp_func(self.libphot[ncl, f_idx])
                elif self.libphot[ncl, f_idx] < min(abs_phot[indices][f_idx]):
                    lib_comp[ncl, indices[f_idx]]= 1.
    
        prob_detected = np.zeros(lib_comp.shape) # shape is the same (Nc,Nb)
        Nc = np.shape(lib_comp)[0] # total number of observed clusters 
        Nf = np.shape(lib_comp)[1] # number of filters 
        comp_ = np.zeros(Nc) # set up an 1D array for final combined complentess values
        
        for i in range(Nf):
            prob_detected[:,i] = lib_comp[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        for i in range(self.Nt//1000):
            d_etect = np.zeros((1000, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(1000, Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            in_v_and_i = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4      # Sum over bands to get number of detections
            in_legus = np.logical_and(np.logical_and(self.libphot[:, -2] <-6, in_v_and_adjacent),in_all_bands) # Combine LEGUS conditions
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands

        comp_ = comp_/(self.Nt//1000)

        if self.padova:
            if allf == 0:
                if self.libindex is None:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_padova_{self.name}_all_fullcomp',comp_) 
                else:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_padova_{self.name}_libid{self.libindex}_fullcomp',comp_) 
            else :
                if self.libindex is None:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_padova_{self.name}_all_noUVcomp',comp_)
                else:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_padova_{self.name}_libid{self.libindex}_noUVcomp',comp_)
        else :
            if allf == 0:
                np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_MIST_{self.name}_fullcomp',comp_) 
            else :
                np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_MIST_{self.name}_noUVcomp',comp_)
        return None
    
class libcomp_nOGC(object):
    def __init__(self, catname, libindex=0,  Nt=30000, padova=True):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        print(f'galaxy name is {catname}')
        self.Nt = Nt
        self.libindex = libindex
        self.padova = padova
        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6
        if self.padova:
            if self.libindex is None:
                lib_files = glob.glob(f'/scratch/mk27/jt4478/output_lib/tang_padova*_cluster_phot.fits')
            else:
                lib_files = glob.glob(f'/scratch/mk27/jt4478/output_lib/tang_padova*_{libindex}_cluster_phot.fits')
            libphot_list = []
            for lf in lib_files:
                libname = lf.split('_cluster_phot.fits')[0]
                print(f'Reading library clusters from {libname}. \n')
                lib = read_cluster(libname, read_filters = self.filterset).phot_neb_ex
                libphot_list.append(lib)
            self.libphot = np.concatenate(libphot_list)
        else:
            lib = read_cluster('/g/data/jh2/jt4478/cluster_slug/tang', read_filters = self.filterset).phot_neb_ex
            self.libphot = lib

    def completeness(self,allf=0):
        # first convert the data to absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        complist = np.zeros(self.libphot.shape)
        for i in range(len(self.filterset)):
            comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
            for j in range(len(self.libphot[:,0])):
                if min(abs_phot[i]) <= self.libphot[j,i] <= max(abs_phot[i]):
                    complist[j,i] = comp_func(self.libphot[j,i])
                elif self.libphot[j,i] < min(abs_phot[i]):
                    complist[j,i] = 1. # detection
                else:
                    complist[j,i] = 0. # non-detection 
        # Set up Markov Chain Monte Carlo experiment  
        prob_detected = np.zeros(complist.shape) # shape is the same (Nc,Nb)
        Nc = len(self.libphot[:,0]) # total number of library clusters 
        if allf == 0 : # If we are computing the completeness for filterset with all five bands 
            comp_l = np.zeros(Nc)
            for i in range(5):
                prob_detected[:,i] = complist[:,i]
            # start Monte Carlo Trials in series , 1000 as one batch 
            for i in range(self.Nt//1000):
                d_etect = np.zeros((1000, Nc, 5), dtype=bool) # set detect matrix 
                rand = np.random.rand(1000,Nc) # set random matrix 
                for i in range(5):
                    d_etect[:,:,i] = rand<prob_detected[:,i]
                in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
                in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
                in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                in_all_bands = np.sum( d_etect, axis=2) >=4       # Sum over bands to get number of detections
                in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
                # set the VI limit phot[:,-2]-phot[:,-1] <= 0.95: V-I <=0.95 FORM WHITEMORE 2023 : VI LIMIT SOLUTION FOR OGC BOX
                in_legus_nOGC = np.logical_and(in_legus, self.libphot[:,3]-self.libphot[:,4] <= 0.95)

                comp_l += np.mean(in_legus_nOGC, axis=0) # This is the completeness values for cluster classes 1, 2 and 3
                del d_etect
                del rand
                del in_legus
                del in_legus_nOGC
                del in_all_bands
            comp_l = comp_l/(self.Nt//1000)
        else : # If we are computing the completeness for filterset with only four filters: U (F336W), B (F435W), V (F555W), IR (F814W)
            comp_l = np.zeros(Nc)
            prob_detected[:,0] = np.zeros(len(prob_detected[:,0]))
            for i in range(4): # Match the column number to the number of filters 
                prob_detected[:,i+1] = complist[:,i+1]
            # start Monte Carlo Trials in series , 1000 as one batch 
            for i in range(self.Nt//100):
                d_etect = np.zeros((100, Nc, 5), dtype=bool) # set detect matrix 
                rand = np.random.rand(100,Nc) # set random matrix 
                for i in range(5):
                    d_etect[:,:,i] = rand<prob_detected[:,i]
                # Imposing the LEGUS criteria 
                in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] ) 
                in_v_and_b = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] ) 
                in_v_and_adjacent = np.logical_or( in_v_and_i, in_v_and_b) # Need to be in V and I or V and B
                in_four_bands = np.sum( d_etect, axis=2) >=4    # Sum over bands to get number of detections
                in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_four_bands)    # Combine all LEGUS conditions
                in_legus_nOGC = np.logical_and(in_legus, self.libphot[:,3]-self.libphot[:,4] <= 0.95)
                comp_l += np.mean(in_legus_nOGC, axis=0)# This is the completeness values for cluster classes 1,2 and 3
                del d_etect
                del rand
                del in_legus
                del in_legus_nOGC
                del in_four_bands
            comp_l = comp_l/(self.Nt//1000)
        # save completeness file in .npy format
        if self.padova:
            if allf == 0:
                if self.libindex is None:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_nOGC_padova_{self.name}_all_fullcomp',comp_l) 
                else:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_nOGC_padova_{self.name}_libid{self.libindex}_fullcomp',comp_l) 
            else :
                if self.libindex is None:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_nOGC_padova_{self.name}_all_noUVcomp',comp_l)
                else:
                    np.save(f'/scratch/jh2/jt4478/tabulated_comp/lib_nOGC_padova_{self.name}_libid{self.libindex}_noUVcomp',comp_l)
        return None
    


class completeness_LEGUS(object):
    def __init__(self, galname, Nt=30000):
        """
        This instantiates a completeness object

        Parameters
           name : string
              the names of the observational catalogs 
          phot_filterset : array
              an array that contains boolean arrays of filterset detected from reading the input catalogs 
           Nf : int 
              number of filters of this filterset 
           Nt     : int 
              number of trials for artificial completeness test 
        """
        self.name = galname
        self.Nt = Nt 
        self.sigma_MR = -0.2103855
        rad_val = np.linspace(0.5 ,10., 10)[1:]
        # read deterministic masses of clusters 
        self.rad_val = rad_val
        # self.phot_filterset = phot_filterset

        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        else:
            phot_val = np.zeros((5, 35))
            phot_comp_2d = np.zeros((5, 9, 35))

            # self.dist = 9.9e6
            
            galaxies = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy')
            gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
            allfilters_cam = []
            filters = []
            gal_name = galname
            for filt, cam in zip(gal_filters[gal_name][0], gal_filters[gal_name][1]):
                filt = filt.upper()
                cam = cam.upper()
                if cam == 'WFC3':
                    cam = 'WFC3_UVIS'
                filt_string = f'{cam}_{filt}'  
                allfilters_cam.append(filt_string)     
                filters.append(filt.lower())
            allfilters_cam.sort(key=lambda x: x[-4:]) 
            self.filterset = allfilters_cam
            filters.sort(key=lambda x: x[-4:]) 

            for i, filt in enumerate(filters):
                rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}/{filt}/recovery/rec*npy')
                rec = np.load(rec_files[0])
                phot_comp_2d[i,:, :] = rec[1:,:]
            phot_val = np.arange(19, 26, 0.2)
            self.phot = phot_val 
            self.comp = phot_comp_2d
            # Assuming you have the README file name stored in readme_file
            galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
            readme_files = glob.glob(f'{galdir}/automatic_catalog*{gal_name}.readme')
            readme_file = readme_files[0]

            with open(readme_file, "r") as f:
                content = f.read()

            # Match aperture radius, distance modulus, and CI using regular expressions
            patterns = [
                (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
                (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
                (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
            ]

            for pattern, label in patterns:
                match = re.search(pattern, content)
                if match:
                    if "distance" in label:
                        galdist = float(match.group(2)) * 1e6
                    elif "CI" in label:
                        ci = float(match.group(1))
                    else:
                        useraperture = float(match.group(1))
                else:
                    raise FileNotFoundError(label + " not found in the readme.")
            self.dist = galdist
    
    def LEGUS_cal(self, phot_arr, filters, mass_filterset):
        min_value = np.min(mass_filterset[mass_filterset > 0.0])
        idx_0 = np.where(mass_filterset == 0.)
        # Replace all elements in mass_filterset that are equal to 0.0 with min_value
        mass_filterset[idx_0] = min_value
        rad_cluster = 10**(0.1415*np.log10(mass_filterset))
        rad_cluster[rad_cluster<0.] = 1.56
        indices = np.where(np.in1d(self.filterset, filters))[0]
        print(indices)
        phot_arr = phot_arr # DO NOT DOUBLE SUBSTRACT THE VALUES 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        phot_comp = np.zeros((phot_arr.shape[0], len(self.filterset)))
        values =  self.comp/ 100.
        complist = np.zeros(phot_arr.shape)
        
        for f_idx, filt in enumerate(filters):
            # comp_func = scipy.interpolate.interp1d(abs_phot[indices][f_idx], self.comp[indices][f_idx]/100.)
            comp_func = scipy.interpolate.RegularGridInterpolator((np.log10(self.rad_val), abs_phot), values[indices[f_idx],:,:], method='slinear', bounds_error=False, fill_value=0.0)
            Nmr = 100
            for j in range(Nmr):
                rad_obs = np.log10(rad_cluster)
                rad_with_random = rad_obs + np.random.randn() * self.sigma_MR
                # Ensure rad_with_random is greater than 1.5
                while np.any(rad_with_random <= np.log10(1.55)):
                    rad_with_random = rad_obs + 0.5*np.random.randn() * self.sigma_MR
                mask = np.logical_and(min(abs_phot) <= phot_arr[:, f_idx], phot_arr[:, f_idx] <= max(abs_phot))
                comp_func_values = comp_func((np.array(rad_with_random[mask]), np.array(phot_arr[mask, f_idx])))
                phot_comp[mask, indices[f_idx]] += comp_func_values 
            phot_comp[mask, indices[f_idx]] /= Nmr
            # Adjust completeness values outside the photometric range
            mask_0 = np.where(phot_arr[:, f_idx] < min(abs_phot))
            mask_1 = np.where(phot_arr[:, f_idx] > max(abs_phot))
            phot_comp[mask_0, indices[f_idx]] = 1.
            phot_comp[mask_1, indices[f_idx]] = 0.

            # Ensure completeness values are clipped within the range [0, 1]
            # phot_comp[:, indices[f_idx]] = np.clip(phot_comp[:, indices[f_idx]], 0.0, 1.0)

            # for ncl in range(np.shape(phot_arr)[0]):
            #     if min(abs_phot[indices][f_idx]) <= phot_arr[ncl, f_idx] <= max(abs_phot[indices][f_idx]):
            #         phot_comp[ncl, indices[f_idx]] = comp_func(phot_arr[ncl, f_idx])
            #     elif phot_arr[ncl, f_idx] < min(abs_phot[indices][f_idx]):
            #         phot_comp[ncl, indices[f_idx]]= 1.
            #     elif phot_arr[ncl, f_idx] > max(abs_phot[indices][f_idx]):
            #         phot_comp[ncl, indices[f_idx]]= 0.

        prob_detected = np.zeros(phot_comp.shape) # shape is the same (Nc,Nb)
        Nc = np.shape(phot_comp)[0] # total number of observed clusters 
        Nf = np.shape(phot_comp)[1] # number of filters 
        comp_ = np.zeros(Nc) # set up an 1D array for final combined complentess values
        
        for i in range(Nf):
            prob_detected[:,i] = phot_comp[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        for i in range(self.Nt//5000):
            d_etect = np.zeros((5000, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(5000,Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            v_band_index = np.where([('555' in band or '606' in band) for band in self.filterset])[0]
            i_band_index = np.where(['814' in band for band in self.filterset])[0]
            b_band_index = np.where([('435' in band or '438' in band) for band in self.filterset])[0]

            # Assuming that only one match per band is expected, select the first match
            v_band_index = v_band_index[0] if len(v_band_index) > 0 else None
            i_band_index = i_band_index[0] if len(i_band_index) > 0 else None
            b_band_index = b_band_index[0] if len(b_band_index) > 0 else None
            in_v_and_i = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, i_band_index] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, b_band_index] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4   # Sum over bands to get number of detections
            # Get correct V band index 
            try:
                v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                print('Visual filter index found.')
            except:
                try:
                    v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F606W', np.array(self.filterset)[indices] == 'f606w'))[0][0]
                except:
                    try:
                        v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'WFC3_UVIS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                    except:
                        print('Visual filter index NOT found.')
                        exit('exiting due to missing V band value.')
                    
            in_legus = np.logical_and(np.logical_and(in_v_and_adjacent, in_all_bands), phot_arr[:,v_index] < -6)
            # Combine LEGUS conditions
            print(in_legus)
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands
        comp_ = comp_/(self.Nt//5000)
        return comp_
    
class completeness_LEGUS_low_reff(object):
    def __init__(self, galname, Nt=30000):
        """
        This instantiates a completeness object

        Parameters
           name : string
              the names of the observational catalogs 
          phot_filterset : array
              an array that contains boolean arrays of filterset detected from reading the input catalogs 
           Nf : int 
              number of filters of this filterset 
           Nt     : int 
              number of trials for artificial completeness test 
        """
        self.name = galname
        self.Nt = Nt 
        self.sigma_MR = -0.2103855
        rad_val = np.linspace(0.5 ,10., 10)[1:2]
        # read deterministic masses of clusters 
        self.rad_val = rad_val
        # self.phot_filterset = phot_filterset

        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        else:
            phot_val = np.zeros((5, 35))
            phot_comp_2d = np.zeros((5, 35))

            # self.dist = 9.9e6
            
            galaxies = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy')
            gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
            allfilters_cam = []
            filters = []
            gal_name = galname
            for filt, cam in zip(gal_filters[gal_name][0], gal_filters[gal_name][1]):
                filt = filt.upper()
                cam = cam.upper()
                if cam == 'WFC3':
                    cam = 'WFC3_UVIS'
                filt_string = f'{cam}_{filt}'  
                allfilters_cam.append(filt_string)     
                filters.append(filt.lower())
            allfilters_cam.sort(key=lambda x: x[-4:]) 
            self.filterset = allfilters_cam
            filters.sort(key=lambda x: x[-4:]) 

            for i, filt in enumerate(filters):
                rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}/{filt}/recovery/rec*npy')
                rec = np.load(rec_files[0])
                phot_comp_2d[i,:] = rec[1:2,:]
                
            phot_val = np.arange(19, 26, 0.2)
            self.phot = phot_val 
            self.comp = phot_comp_2d
            # Assuming you have the README file name stored in readme_file
            galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
            readme_files = glob.glob(f'{galdir}/automatic_catalog*{gal_name}.readme')
            readme_file = readme_files[0]

            with open(readme_file, "r") as f:
                content = f.read()

            # Match aperture radius, distance modulus, and CI using regular expressions
            patterns = [
                (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
                (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
                (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
            ]

            for pattern, label in patterns:
                match = re.search(pattern, content)
                if match:
                    if "distance" in label:
                        galdist = float(match.group(2)) * 1e6
                    elif "CI" in label:
                        ci = float(match.group(1))
                    else:
                        useraperture = float(match.group(1))
                else:
                    raise FileNotFoundError(label + " not found in the readme.")
            self.dist = galdist
    
    def LEGUS_cal(self, phot_arr, filters, mass_filterset):
        min_value = np.min(mass_filterset[mass_filterset > 0.0])
        idx_0 = np.where(mass_filterset == 0.)
        # Replace all elements in mass_filterset that are equal to 0.0 with min_value
        mass_filterset[idx_0] = min_value
        indices = np.where(np.in1d(self.filterset, filters))[0]
        print(indices)
        phot_arr = phot_arr # DO NOT DOUBLE SUBSTRACT THE VALUES 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        phot_comp = np.zeros((phot_arr.shape[0], len(self.filterset)))
        values =  self.comp/ 100.
        complist = np.zeros(phot_arr.shape)
        
        for f_idx, filt in enumerate(filters):
            comp_func = scipy.interpolate.interp1d(abs_phot, values[indices[f_idx],:])
            mask = np.logical_and(min(abs_phot) <= phot_arr[:, f_idx], phot_arr[:, f_idx] <= max(abs_phot))
            comp_func_values = comp_func(np.array(phot_arr[mask, f_idx])) 
            phot_comp[mask, indices[f_idx]] += comp_func_values 
            # Adjust completeness values outside the photometric range
            mask_0 = np.where(phot_arr[:, f_idx] < min(abs_phot))
            mask_1 = np.where(phot_arr[:, f_idx] > max(abs_phot))
            phot_comp[mask_0, indices[f_idx]] = 1.
            phot_comp[mask_1, indices[f_idx]] = 0.

        prob_detected = np.zeros(phot_comp.shape) # shape is the same (Nc,Nb)
        Nc = np.shape(phot_comp)[0] # total number of observed clusters 
        Nf = np.shape(phot_comp)[1] # number of filters 
        comp_ = np.zeros(Nc) # set up an 1D array for final combined complentess values
        
        for i in range(Nf):
            prob_detected[:,i] = phot_comp[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        for i in range(self.Nt//5000):
            d_etect = np.zeros((5000, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(5000,Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            v_band_index = np.where([('555' in band or '606' in band) for band in self.filterset])[0]
            i_band_index = np.where(['814' in band for band in self.filterset])[0]
            b_band_index = np.where([('435' in band or '438' in band) for band in self.filterset])[0]

            # Assuming that only one match per band is expected, select the first match
            v_band_index = v_band_index[0] if len(v_band_index) > 0 else None
            i_band_index = i_band_index[0] if len(i_band_index) > 0 else None
            b_band_index = b_band_index[0] if len(b_band_index) > 0 else None
            in_v_and_i = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, i_band_index] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, b_band_index] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4   # Sum over bands to get number of detections
            # Get correct V band index 
            try:
                v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                print('Visual filter index found.')
            except:
                try:
                    v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F606W', np.array(self.filterset)[indices] == 'f606w'))[0][0]
                except:
                    try:
                        v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'WFC3_UVIS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                    except:
                        print('Visual filter index NOT found.')
                        exit('exiting due to missing V band value.')
                    
            in_legus = np.logical_and(np.logical_and(in_v_and_adjacent, in_all_bands), phot_arr[:,v_index] < -6)
            # Combine LEGUS conditions
            print(in_legus)
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands
        comp_ = comp_/(self.Nt//5000)
        return comp_
    
class completeness_LEGUS_ngc5457(object):
    def __init__(self, galname, comptype=None, Nt=30000):
        """
        This instantiates a completeness object

        Parameters
           name : string
              the names of the observational catalogs 
          phot_filterset : array
              an array that contains boolean arrays of filterset detected from reading the input catalogs 
           Nf : int 
              number of filters of this filterset 
           Nt     : int 
              number of trials for artificial completeness test 
           comptype : type of the completness estimation used, options are 'literature', 'fixed', 'mass_radius' 
        """
        self.name = galname
        self.comptype = comptype 
        self.Nt = Nt 
        self.sigma_MR = -0.2103855

        # set the values 
        phot_val = np.arange(18. ,26., 0.3)
        if self.comptype == 'literature':
            rad_val = np.array([1., 2., 3., 4., 5., 10.]) # Because of CI cut, we need the cluster catalogue to contain only clusters  > 1pc. 
            # Magnitude values used in artificial cluster test        
            phot_comp_2d = np.zeros((5, np.shape(phot_val)[0]))
        elif self.comptype == 'fixed':
            rad_val = 2.
            phot_comp_2d = np.zeros((5, np.shape(phot_val)[0])) 
        elif self.comptype == 'mass_radius':
            rad_val = np.array([1., 2., 3., 4., 5., 10.])
            phot_comp_2d = np.zeros((5, 6, np.shape(phot_val)[0])) 
        
        # read deterministic masses of clusters 
        self.rad_val = rad_val
        self.phot = phot_val
        # self.phot_filterset = phot_filterset


        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        else:
            # self.dist = 9.9e6
            galaxies = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy')
            gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
            allfilters_cam = []
            filters = []
            gal_name = galname
            for filt, cam in zip(gal_filters[gal_name][0], gal_filters[gal_name][1]):
                filt = filt.upper()
                cam = cam.upper()
                if cam == 'WFC3':
                    cam = 'WFC3_UVIS'
                filt_string = f'{cam}_{filt}'  
                allfilters_cam.append(filt_string)     
                filters.append(filt.lower())
            allfilters_cam.sort(key=lambda x: x[-4:]) 
            self.filterset = allfilters_cam
            filters.sort(key=lambda x: x[-4:]) 
            if self.comptype == 'literature':
                for i, filt in enumerate(filters):
                    rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}_Linden/{filt}/recovery/recov_mag_CI_Linden.npy')
                    rec = np.load(rec_files[0])
                    phot_comp_2d[i,:] = rec
            elif self.comptype == 'fixed':
                for i, filt in enumerate(filters):
                    rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}_Linden/{filt}/recovery/recov_mag_CI_test_sim.npy')
                    rec = np.load(rec_files[0])
                    phot_comp_2d[i,:] = rec[1:2,:]
            elif self.comptype == 'mass_radius':
                for i, filt in enumerate(filters):
                    rec_files = glob.glob(f'/g/data/jh2/jt4478/make_LEGUS_CCT/{gal_name}_Linden/{filt}/recovery/recov_mag_CI_test_sim.npy')
                    rec = np.load(rec_files[0])
                    phot_comp_2d[i,:,:] = rec
            else:
                raise NotImplementedError('Invalid completeness estimation type, exiting...')
            
            
            self.comp = phot_comp_2d
            # Assuming you have the README file name stored in readme_file
            galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
            readme_files = glob.glob(f'{galdir}/automatic_catalog*{gal_name}.readme')
            readme_file = readme_files[0]

            with open(readme_file, "r") as f:
                content = f.read()

            # Match aperture radius, distance modulus, and CI using regular expressions
            patterns = [
                (r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", "User-aperture radius"),
                (r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", "Galactic distance"),
                (r'This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.', "CI value")
            ]

            for pattern, label in patterns:
                match = re.search(pattern, content)
                if match:
                    if "distance" in label:
                        galdist = float(match.group(2)) * 1e6
                    elif "CI" in label:
                        ci = float(match.group(1))
                    else:
                        useraperture = float(match.group(1))
                else:
                    raise FileNotFoundError(label + " not found in the readme.")
            self.dist = galdist
    
    def LEGUS_cal(self, phot_arr, filters, mass_filterset):
        min_value = np.min(mass_filterset[mass_filterset > 0.0])
        idx_0 = np.where(mass_filterset == 0.)
        # Replace all elements in mass_filterset that are equal to 0.0 with min_value
        mass_filterset[idx_0] = min_value
        indices = np.where(np.in1d(self.filterset, filters))[0]
        print(indices)
        phot_arr = phot_arr # DO NOT DOUBLE SUBSTRACT THE VALUES 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        phot_comp = np.zeros((phot_arr.shape[0], len(self.filterset)))
        values =  self.comp/ 100.
        complist = np.zeros(phot_arr.shape)
        
        if self.comptype == 'literature' or self.comptype == 'fixed':
            for f_idx, filt in enumerate(filters):
                comp_func = scipy.interpolate.interp1d(abs_phot, values[indices[f_idx],:])
                mask = np.logical_and(min(abs_phot) <= phot_arr[:, f_idx], phot_arr[:, f_idx] <= max(abs_phot))
                comp_func_values = comp_func(np.array(phot_arr[mask, f_idx])) 
                phot_comp[mask, indices[f_idx]] += comp_func_values 
                # Adjust completeness values outside the photometric range
                mask_0 = np.where(phot_arr[:, f_idx] < min(abs_phot))
                mask_1 = np.where(phot_arr[:, f_idx] > max(abs_phot))
                phot_comp[mask_0, indices[f_idx]] = 1.
                phot_comp[mask_1, indices[f_idx]] = 0.
        elif self.comptype == 'mass_radius':
            min_value = np.min(mass_filterset[mass_filterset > 0.0])
            idx_0 = np.where(mass_filterset == 0.)
            # Replace all elements in mass_filterset that are equal to 0.0 with min_value
            mass_filterset[idx_0] = min_value
            rad_cluster = 10**(0.1415*np.log10(mass_filterset))
            rad_cluster[rad_cluster<0.] = 1.56
            indices = np.where(np.in1d(self.filterset, filters))[0]
            print(indices)
            phot_arr = phot_arr # DO NOT DOUBLE SUBSTRACT THE VALUES 
            abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
            phot_comp = np.zeros((phot_arr.shape[0], len(self.filterset)))
            values =  self.comp/ 100.
            complist = np.zeros(phot_arr.shape)
            
            for f_idx, filt in enumerate(filters):
                # comp_func = scipy.interpolate.interp1d(abs_phot[indices][f_idx], self.comp[indices][f_idx]/100.)
                comp_func = scipy.interpolate.RegularGridInterpolator((np.log10(self.rad_val), abs_phot), values[indices[f_idx],:,:], method='slinear', bounds_error=False, fill_value=0.0)
                Nmr = 100
                for j in range(Nmr):
                    rad_obs = np.log10(rad_cluster)
                    rad_with_random = rad_obs + np.random.randn() * self.sigma_MR
                    # Ensure rad_with_random is greater than 1.5
                    while np.any(rad_with_random <= np.log10(1.55)):
                        rad_with_random = rad_obs + 0.5*np.random.randn() * self.sigma_MR
                    mask = np.logical_and(min(abs_phot) <= phot_arr[:, f_idx], phot_arr[:, f_idx] <= max(abs_phot))
                    comp_func_values = comp_func((np.array(rad_with_random[mask]), np.array(phot_arr[mask, f_idx])))
                    phot_comp[mask, indices[f_idx]] += comp_func_values 
                phot_comp[mask, indices[f_idx]] /= Nmr
                # Adjust completeness values outside the photometric range
                mask_0 = np.where(phot_arr[:, f_idx] < min(abs_phot))
                mask_1 = np.where(phot_arr[:, f_idx] > max(abs_phot))
                phot_comp[mask_0, indices[f_idx]] = 1.
                phot_comp[mask_1, indices[f_idx]] = 0.

        else:
            raise NotImplementedError


        prob_detected = np.zeros(phot_comp.shape) # shape is the same (Nc,Nb)
        Nc = np.shape(phot_comp)[0] # total number of observed clusters 
        Nf = np.shape(phot_comp)[1] # number of filters 
        comp_ = np.zeros(Nc) # set up an 1D array for final combined complentess values
        
        for i in range(Nf):
            prob_detected[:,i] = phot_comp[:,i]
        # start Monte Carlo Trials in series , 1000 as one batch 
        for i in range(self.Nt//5000):
            d_etect = np.zeros((5000, Nc, Nf), dtype=np.bool_) # set detect matrix 
            rand = np.random.rand(5000,Nc) # set random matrix 
            for f in range(Nf):
                d_etect[:,:,f] = rand<prob_detected[:,f]
            v_band_index = np.where([('555' in band or '606' in band) for band in self.filterset])[0]
            i_band_index = np.where(['814' in band for band in self.filterset])[0]
            b_band_index = np.where([('435' in band or '438' in band) for band in self.filterset])[0]

            # Assuming that only one match per band is expected, select the first match
            v_band_index = v_band_index[0] if len(v_band_index) > 0 else None
            i_band_index = i_band_index[0] if len(i_band_index) > 0 else None
            b_band_index = b_band_index[0] if len(b_band_index) > 0 else None
            in_v_and_i = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, i_band_index] )     # Check V and I detection
            in_b_and_v = np.logical_and(d_etect[:,:, v_band_index], d_etect[:,:, b_band_index] )# Check V and B detection
            in_v_and_adjacent = np.logical_or(in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
            in_all_bands = np.sum(d_etect, axis=2)>=4   # Sum over bands to get number of detections
            # Get correct V band index 
            try:
                v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                print('Visual filter index found.')
            except:
                try:
                    v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'ACS_F606W', np.array(self.filterset)[indices] == 'f606w'))[0][0]
                except:
                    try:
                        v_index = np.where(np.logical_or(np.array(self.filterset)[indices] == 'WFC3_UVIS_F555W', np.array(self.filterset)[indices] == 'f555w'))[0][0]
                    except:
                        print('Visual filter index NOT found.')
                        exit('exiting due to missing V band value.')
                    
            in_legus = np.logical_and(np.logical_and(in_v_and_adjacent, in_all_bands), phot_arr[:,v_index] < -6)
            # Combine LEGUS conditions
            print(in_legus)
            comp_ += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
            del d_etect
            del rand
            del in_legus
            del in_all_bands
        comp_ = comp_/(self.Nt//5000)
        return comp_
    
class completeness_LEGUS_nOGC(object):
    def __init__(self, catname, phot_filterset,Nt = 50000):
        """
        This instantiates a completeness object

        Parameters
           name : string
              the names of the observational catalogs 
          phot_filterset : array
              an array that contains boolean arrays of filterset detected from reading the input catalogs 
           Nf : int 
              number of filters of this filterset 
           Nt     : int 
              number of trials for artificial completeness test 
        """
        self.name = catname 
        self.Nt = Nt 
        self.phot_filterset = phot_filterset

        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        else:
            print('unknown observational completenss!!')
            exit(0)

    def comp_cal(self):
        # first convert the data to Absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        comp_filterset = []
        for pf in range(len(self.phot_filterset)):
            # loop through all filtersets 
            allf = len(self.phot_filterset[pf][0]) == 5 # check if the filterset is 5 filters
            complistLEGUS= np.zeros(np.shape(self.phot_filterset[pf])) # 2D complist
            if allf :  
                for i in range(len(self.filterset)):
                    comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
                    for j in range(len(self.phot_filterset[pf])):
                        if min(abs_phot[i]) <= self.phot_filterset[pf][j,i] <= max(abs_phot[i]):
                            complistLEGUS[j,i] = comp_func(self.phot_filterset[pf][j,i])
                        elif self.phot_filterset[pf][j,i] < min(abs_phot[i]):
                            complistLEGUS[j,i] = 1.
                        else:
                            complistLEGUS[j,i] = 0.
            else : 
                for i in range(len(self.filterset)-1):
                    comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
                    for j in range(len(self.phot_filterset[pf])):
                        if min(abs_phot[i]) <= self.phot_filterset[pf][j,i] <= max(abs_phot[i]):
                            complistLEGUS[j,i] = comp_func(self.phot_filterset[pf][j,i])
                        elif self.phot_filterset[pf][j,i] < min(abs_phot[i]):
                            complistLEGUS[j,i] = 1.
                        else:
                            complistLEGUS[j,i] = 0.
            # next set up a Monte Carlo Simulation base case 
            prob_detected = np.zeros(complistLEGUS.shape) # shape is the same (Nc,Nb)
            Nc = len(self.phot_filterset[pf]) # total number of library clusters 
            if allf:
                comp_l = np.zeros(Nc)
                for i in range(5):
                    prob_detected[:,i] = complistLEGUS[:,i]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//1000):
                    d_etect = np.zeros((1000, Nc, 5), dtype=bool) # set detect matrix 
                    rand = np.random.rand(1000,Nc) # set random matrix 
                    for i in range(5):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
                    in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                    in_all_bands = np.sum( d_etect, axis=2) >=4      # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.phot_filterset[pf][:,3]<-6, in_v_and_adjacent),in_all_bands) # Combine LEGUS conditions
                    in_legus_nOGC = np.logical_and(in_legus, self.phot_filterset[pf][:,3]-self.phot_filterset[pf][:,4] <= 0.95)
                    comp_l += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
                    del d_etect
                    del rand
                    del in_legus
                    del in_all_bands
                comp_l = comp_l/(self.Nt//1000)
                comp_filterset.append(comp_l)
                np.save('LEGUS'+self.name+'_nOGC_comp',comp_l) 
            elif not allf:
                comp_l = np.zeros(Nc)
                for i in range(4):
                    prob_detected[:,i] = complistLEGUS[:,i]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//1000):
                    d_etect = np.zeros((1000, Nc, 4), dtype=bool) # set detect matrix 
                    rand = np.random.rand(1000,Nc) # set random matrix 
                    for i in range(4):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    in_v_and_i = np.logical_and( d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
                    in_b_and_v = np.logical_and( d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                    in_all_bands = np.sum( d_etect, axis=2) >=4        # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.phot_filterset[pf][:,-2]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
                    in_legus_nOGC = np.logical_and(in_legus, self.phot_filterset[pf][:,-2]-self.phot_filterset[pf][:,-1] <= 0.95)
                    comp_l += np.mean(in_legus_nOGC, axis=0)# This is the comp for classes 1,2,3
                    del d_etect
                    del rand
                    del in_legus_nOGC
                    del in_all_bands
                comp_l = comp_l/(self.Nt//1000)
                np.save('LEGUS'+self.name+'_nOGC_noUV_comp',comp_l)
                comp_filterset.append(comp_l)
            else:
                exit(0) 
        return None

##############################################################
# Here we register the list of known completeness functions; #
# note that the register entry "LEGUS" is an abstract class  #
# that must be completed by instantiating it with a filter   #
# set and completeness data                                  #
##############################################################

comp_register = {
    'LEGUS' : completeness_LEGUS,
    'library' : libcomp,
    'LEGUS_rOGC': completeness_LEGUS_nOGC,
    'library_rOGC' : libcomp_nOGC
}



if __name__ == "__main__":
    # Create comp class 
    cl = libcomp_nOGC(catname=args.galaxies, libindex=None, Nt=10000)
    cl.completeness(allf=args.allf)
    # if args.fid == 0:
    #     allf_= True
    # else:
    #     allf_=False

    # # Load library cluster photometric properties using RawArray to read from buffer for memory saving 
    # libphot_list = []
    # lib_files = glob.glob('/scratch/mk27/jt4478/output_lib/tang_padova*cluster_phot.fits')
    # if '628c' in args.galaxies:
    #     LEGUS_filters = LEGUS_field_data['ngc628c']['filterset']
    # elif '628e' in args.galaxies:
    #     LEGUS_filters = LEGUS_field_data['ngc628e']['filterset']
    # else:
    #     raise NotImplementedError('Unknown galaxy filters, exiting.')

    # for lf in lib_files:
    #     libname = lf.split('_cluster_phot.fits')[0]
    #     print(f'Reading library clusters from {libname}. \n')
    #     lib = read_cluster(libname, read_filters = LEGUS_filters).phot_neb_ex
    #     libphot_list.append(lib)

    # # Concatenate all sub-libraries 
    # libphot = np.concatenate(libphot_list)
    # num_cluster, num_filter = np.shape(libphot)
    # # Convert libphot to RawArray
    # libphot_base = RawArray('d', num_cluster*num_filter)
    # libphot_base[:] = libphot.flatten()

    # # Define completness object 
    # instance  = libcomp(catname=args.galaxies, Nt=args.nt, allf=allf_, multi_lib_padova=True)
    # # Perform multiprocessing with Pool
    # num_processes = 1
    # comp_batch_num = [(instance.Nt // num_processes, instance) for i in range(num_processes)]
    
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(monte_carlo_comp, comp_batch_num)

    # comp_all = sum(results) / len(comp_batch_num)

    # # save completeness file in .npy format
    # if instance.multi_lib_padova:
    #     if args.allf:
    #         np.save('/scratch/jh2/jt4478/tabulated_comp/lib'+instance.catname+f'_full_comp_padova_{args.bid}', comp_all)
    #     else:
    #         np.save('/scratch/jh2/jt4478/tabulated_comp/lib'+instance.catname+f'_no_UV_comp_padova_{args.bid}', comp_all)
    

    # cl.completeness(Nt=300000)
