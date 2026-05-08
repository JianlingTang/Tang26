"""Observational catalog readers (HLSP, LEGUS, mocks).

Ancillary layout files use ``LEGUS_CCT_ROOT`` and legacy readme fallbacks
``LEGUS_TAB_DIR`` (set by the analyze script or the environment).
"""

import glob
import os
import os.path as osp
import re

import numpy as np
from astropy.io import ascii as asc


def _legus_cct_root() -> str:
    return os.environ.get("LEGUS_CCT_ROOT", "/g/data/jh2/jt4478/make_LEGUS_CCT")


def _legus_tab_dir() -> str:
    return os.environ.get("LEGUS_TAB_DIR", "/home/100/jt4478/slugfiles/LEGUS_cat")


class catalog_reader(object):
    """
    This is a class whose job is to read observational catalogs. It is
    a purely abstract class that defines the minimum set entries that
    the read function must return
    """
    def __init__(self):
        """
        The initializer does nothing
        """
        pass

    def read(self, fname):
        """
        Function to read a catalog

        Parameters
           fname : string
              name of the catalog file to read

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
        """
        raise NotImplementedError(
            "completeness is an abstract class; "
            "implement a derived class")


class catalog_reader_rOGC(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__

    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        age = np.array(data['col17'], dtype='float')
        age_max = np.array(data['col18'], dtype='float')
        age_min = np.array(data['col19'], dtype='float')

        mass = np.array(data['col20'], dtype='float')
        mass_max = np.array(data['col21'], dtype='float')
        mass_min = np.array(data['col22'], dtype='float')

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')
        
        #excluding class zero clusters from legus and potential OGCs using V-I and U-B colors
        # OGC_ = np.logical_and(np.logical_and(phot[:,-4]-phot[:,-3] > -0.7,\
        #                                                  phot[:,-4]-phot[:,-3] < 1.2), \
        #                                   np.logical_and(phot[:,-2]-phot[:,-1] > 0.95, \
        #                                                  phot[:,-2]-phot[:,-1] < 1.7))
        # OGC_b = [not bool_val for bool_val in OGC_]
        
        # print('Checking the boolean lists have equal shapes:',np.shape(OGC_b) == np.shape(OGC_))
        i_nocut = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        i = np.logical_and(np.logical_and(classification > classcut[0],
                             classification < classcut[1]), \
                             phot[:,-2]-phot[:,-1] <= 0.95)
        # cid_nocut = cid[i_nocut]
        cid = cid[i]
        # save a index list of non-OGC box for later use. 
        # ind_OGC = []
        # subarr = list(cid)
        # arr = list(cid_nocut)

        # for i in range(len(subarr)):
        #     ind_OGC.append(arr.index(subarr[i]))
        # ind_OGC = np.array(ind_OGC)
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        age = age[i]
        age_min = age_min[i]
        age_max = age_max[i]

        mass = mass[i]
        mass_min = mass_min[i]
        mass_max = mass_max[i]



        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max
            # "index_OGC" : ind_OGC 
        }

        # Return
        return out

class catalog_reader_LEGUS_div(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
        
    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        fmeta = osp.splitext(fname)[0]+'.tab'
        fp = open(fmeta, 'r')
        metadata = fp.read().splitlines()
        fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = data['col34']


        
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]

        # Package the output
        out = { "path"       : fname,
            "basename"   : osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
             "viscat"     : viscat
        }

        # Return
        return out

class catalog_reader_LEGUS(object):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    # def __init__(self):
    #     __doc__ == catalog_reader.__init__.__doc__

    def read(self, fname, galname):

        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
            dmod = 29.98 # Calzetti et al 2015
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
            dmod = 29.98 # Calzetti et al 2015
        else :
            gallname =   galname.split('_')[0]
            galaxies = np.load(osp.join(_legus_cct_root(), "galaxy_names.npy"))
            gal_filters = np.load(
                osp.join(_legus_cct_root(), "galaxy_filter_dict.npy"),
                allow_pickle=True,
            ).item()
            filters = []
            for filt, cam in zip(gal_filters[gallname][0], gal_filters[gallname][1]):
                filt = filt.upper()
                cam = cam.upper()
                if cam == 'WFC3':
                    cam = 'WFC3_UVIS'
                filt_string = f'{cam}_{filt}'  
                filters.append(filt_string)  
            filters.sort(key=lambda x: x[-4:])
            print(f" filters are {filters}")
            gal_name = galname
            galdir = os.path.join(_legus_cct_root(), gal_name)
            # try:
            #     readme_file = os.path.join(galdir,'automatic_catalog_ngc5194-ngc5195-mosaic.readme')
            # except:
            readmes = glob.glob(f'{galdir}/automatic_catalog*{gallname}.readme')
            readme_file = readmes[0]
            with open(readme_file, "r") as f:
                content = f.read()
            # Use regular expression to extract the magnitude value
            magnitude_match = re.search(r'(\d+\.\d+) mag', content)
            if magnitude_match:
                dmod = float(magnitude_match.group(1))
                print("Magnitude Value:", dmod)
            else:
                print("Magnitude not found in the text.")


        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        try:
            age = np.array(data['col17'], dtype='float')
            age_max = np.array(data['col18'], dtype='float')
            age_min = np.array(data['col19'], dtype='float')

            mass = np.array(data['col20'], dtype='float')
            mass_max = np.array(data['col21'], dtype='float')
            mass_min = np.array(data['col22'], dtype='float')
        except:
            None

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)

        
        first_mag_index = None
        class_id = None
        if not '628' in fname:
            readme_cluster_file = fname.replace(".tab", ".readme")
            with open(readme_cluster_file, 'r') as f:
                lines = f.readlines()
        else: 
            readme_cluster_file = osp.join(_legus_tab_dir(), "ngc628.readme")
            with open(readme_cluster_file, 'r') as f:
                lines = f.readlines()


        # Iterate through the lines to find the line containing "final total mag in"
        for line in lines:
            if "final total mag in" in line:
                # Extract the number before "final total mag in" using split and strip
                first_mag_index = line.split("final total mag in")[0].strip()
                first_mag_index = int(first_mag_index.rstrip('.'))
                break
        readme_cluster_file = fname.replace(".tab", ".readme")


        for line in lines:
            if "Final assigned class of the source after visual inspection" in line:
                # Extract the number before "final total mag in" using split and strip
                class_id = line.split("Final assigned class of the source after visual inspection")[0].strip()
                class_id = int(class_id.rstrip('.'))
                print(class_id)
                break
            elif "Visual classification done by LEGUS team-member Sean Linden and Brad Whitmore." in line:
                class_id = line.split("Visual classification done by LEGUS team-member Sean Linden and Brad Whitmore.")[0].strip()
                class_id = int(class_id.split('.')[0])
                print(class_id)
            elif "Final morphological classification assigned by 1 human who has verified the ML classification" in line:
                class_id = line.split("Final morphological classification assigned by 1 human who has verified the ML classification")[0].strip()
                class_id = int(class_id.split('.')[0])
                print(class_id)
            elif "Visual classification done by " in line:
                class_id = line.split("Visual classification done by ")[0].strip()
                class_id = int(class_id.split('.')[0])
                print(class_id)
            else:
                continue

        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+first_mag_index)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+first_mag_index+1)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+first_mag_index)]< 99.999, data['col{:d}'.format(2*i+first_mag_index)] != 66.666) # Flag value
        for line in lines:
            if "RA coordinates in" in line:
                # Extract the number before "final total mag in" using split and strip
                ra_d = line.split("RA coordinates in")[0].strip()
                ra_d = int(ra_d.rstrip('.'))
                break

        ra = np.array(data['col{:d}'.format(ra_d)], dtype= 'float')
        dec = np.array(data['col{:d}'.format(ra_d+1)], dtype= 'float')
        classification = np.array(data['col{:d}'.format(class_id)], dtype='int')
        classcut= [0, 3.5]
        try:
            #excluding class zero clusters from legus
            i = np.logical_and(classification > classcut[0],  classification < classcut[1])
            cid = cid[i]
            phot = phot[i]
            photerr = photerr[i]
            detect = detect[i]
            ra = ra[i]
            dec = dec[i]
            cla = classification[i]
            age = age[i]
            age_min = age_min[i]
            age_max = age_max[i]

            mass = mass[i]
            mass_min = mass_min[i]
            mass_max = mass_max[i]

             # Package the output
            output = { "path"       : fname,
            "galaxy"     :gal_name,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max}
        except:
            #excluding class zero clusters from legus
            i = np.logical_and(classification > classcut[0],  classification < classcut[1])
            cid = cid[i]
            phot = phot[i]
            photerr = photerr[i]
            detect = detect[i]
            ra = ra[i]
            dec = dec[i]
            cla = classification[i]

             # Package the output
            output = { "path"       : fname,
            "galaxy"     :gal_name,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat}
            
        
        # Return
        return output

class catalog_reader_LEGUS_with_EBV(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        age = np.array(data['col17'], dtype='float')
        age_max = np.array(data['col18'], dtype='float')
        age_min = np.array(data['col19'], dtype='float')

        mass = np.array(data['col20'], dtype='float')
        mass_max = np.array(data['col21'], dtype='float')
        mass_min = np.array(data['col22'], dtype='float')

        EBV = np.array(data['col23'], dtype='float')
        EBV_max = np.array(data['col24'], dtype='float')
        EBV_min = np.array(data['col25'], dtype='float')

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')

        
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        age = age[i]
        age_min = age_min[i]
        age_max = age_max[i]

        mass = mass[i]
        mass_min = mass_min[i]
        mass_max = mass_max[i]

        EBV = EBV[i]
        EBV_min = EBV_min[i]
        EBV_max = EBV_max[i]

        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max,
            "EBV"        : EBV,
            "EBV_min"     : EBV_min,
            "EBV_max"     : EBV_max
        }

        # Return
        return out

class catalog_reader_LEGUS_good_phot(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        age = np.array(data['col17'], dtype='float')
        age_max = np.array(data['col18'], dtype='float')
        age_min = np.array(data['col19'], dtype='float')

        mass = np.array(data['col20'], dtype='float')
        mass_max = np.array(data['col21'], dtype='float')
        mass_min = np.array(data['col22'], dtype='float')

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+7)] <=0.3, np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666)) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')

        classcut=[0, 3.5]
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        age = age[i]
        age_min = age_min[i]
        age_max = age_max[i]

        mass = mass[i]
        mass_min = mass_min[i]
        mass_max = mass_max[i]



        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max
        }

        # Return
        return out

class catalog_reader_simple(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        age = np.array(data['col17'], dtype='float')
        age_max = np.array(data['col18'], dtype='float')
        age_min = np.array(data['col19'], dtype='float')

        mass = np.array(data['col20'], dtype='float')
        mass_max = np.array(data['col21'], dtype='float')
        mass_min = np.array(data['col22'], dtype='float')

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')

        
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        age = age[i]
        age_min = age_min[i]
        age_max = age_max[i]

        mass = mass[i]
        mass_min = mass_min[i]
        mass_max = mass_max[i]



        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max
        }

        # Return
        return out

class catalog_reader_cl12(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname, classcut=[0, 2.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
        data = asc.read(fname)
        cid = np.array(data['col1'], dtype='int')
        age = np.array(data['col17'], dtype='float')
        age_max = np.array(data['col18'], dtype='float')
        age_min = np.array(data['col19'], dtype='float')

        mass = np.array(data['col20'], dtype='float')
        mass_max = np.array(data['col21'], dtype='float')
        mass_min = np.array(data['col22'], dtype='float')

        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        for i in range(nf):
            phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
            photerr[:,i] = data['col{:d}'.format(2*i+7)] 
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')

        
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        age = age[i]
        age_min = age_min[i]
        age_max = age_max[i]

        mass = mass[i]
        mass_min = mass_min[i]
        mass_max = mass_max[i]



        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
            "viscat"     : viscat,
            "age"         : age,
            "age_min"     : age_min,
            "age_max"     : age_max,
            "mass"        : mass,
            "mass_min"     : mass_min,
            "mass_max"     : mass_max
        }

        # Return
        return out


def _infer_galaxy_fullname_from_path(fname):
    base = osp.basename(fname).lower()
    m = re.search(r"(ngc[0-9]+[a-z0-9\-]*)", base)
    if m:
        return m.group(1)
    return osp.splitext(base)[0]


def _find_hlsp_readme_for_catalog(fname):
    direct = osp.splitext(fname)[0] + ".readme"
    if osp.exists(direct):
        return direct
    dirname = osp.dirname(fname)
    gal = _infer_galaxy_fullname_from_path(fname)
    candidates = sorted(glob.glob(osp.join(dirname, f"hlsp*{gal}*.readme")))
    if not candidates:
        raise FileNotFoundError(f"No HLSP readme matched pattern hlsp*{gal}*.readme in {dirname}")
    return candidates[0]


def _normalize_filter_token(filter_token):
    token = filter_token.strip().upper().replace("-", "_")
    token = token.replace("/", "_")
    if "_" in token and token.startswith(("WFC3", "ACS")):
        if token.startswith("WFC3_") and not token.startswith("WFC3_UVIS_"):
            # LEGUS UVIS naming in slug libs uses WFC3_UVIS_*
            parts = token.split("_", 1)
            if len(parts) == 2 and parts[1].startswith("F"):
                token = f"WFC3_UVIS_{parts[1]}"
        return token
    # Fallback mapping for common LEGUS bands.
    if token in {"F275W", "F336W", "F555W"}:
        return f"WFC3_UVIS_{token}"
    if token in {"F435W", "F814W"}:
        return f"ACS_{token}"
    return token


def _parse_hlsp_readme_metadata(readme_path):
    with open(readme_path, "r") as fp:
        lines = fp.readlines()
    dmod = None
    dmod_line = None
    for line in lines:
        m = re.search(
            r"Distance modulus used\s+([0-9]+(?:\.[0-9]+)?)\s+mag",
            line,
            re.IGNORECASE,
        )
        if m:
            dmod = float(m.group(1))
            dmod_line = line.strip()
            break
    if dmod is None:
        raise ValueError(f"Distance modulus not found in readme: {readme_path}")

    first_mag_index = None
    class_index = None
    ra_index = None
    filter_tokens = []
    first_mag_line = None

    for line in lines:
        line_clean = line.strip()
        if "final total mag in" in line_clean.lower():
            # e.g. "6. final total mag in F275W"
            idx_match = re.match(r"(\d+)\.", line_clean)
            filt_match = re.search(
                r"final total mag in\s+([A-Za-z0-9_\-/]+)",
                line_clean,
                re.IGNORECASE,
            )
            if idx_match and first_mag_index is None:
                first_mag_index = int(idx_match.group(1))
                first_mag_line = line_clean
            if filt_match:
                filter_tokens.append(_normalize_filter_token(filt_match.group(1)))
        if "RA coordinates in" in line_clean:
            idx_match = re.match(r"(\d+)\.", line_clean)
            if idx_match:
                ra_index = int(idx_match.group(1))
        if "Final assigned class of the source after visual inspection" in line_clean:
            idx_match = re.match(r"(\d+)\.", line_clean)
            if idx_match:
                idx_val = int(idx_match.group(1))
                lower = line_clean.lower()
                # Prefer "applying the mode" (class_mode) when available.
                if "applying the mode" in lower:
                    class_index = idx_val
                elif class_index is None and "applying the mean" not in lower:
                    # Backward-compatible fallback for older readmes with a single class line.
                    class_index = idx_val

    if class_index is None:
        # Conservative fallback for HLSP LEGUS catalogs where class_mode is column 34.
        class_index = 34

    if first_mag_index is None or ra_index is None:
        raise ValueError(f"Required HLSP column metadata missing in readme: {readme_path}")

    # Preserve order while removing duplicates.
    filters = []
    for f in filter_tokens:
        if f not in filters:
            filters.append(f)
    if not filters:
        raise ValueError(f"No filters parsed from readme: {readme_path}")

    debug_meta = {
        "dmod_line": dmod_line,
        "dmod_value": dmod,
        "first_mag_line": first_mag_line,
        "first_mag_index": first_mag_index,
    }
    return dmod, filters, first_mag_index, class_index, ra_index, debug_meta


class catalog_reader_legus_hlsp(catalog_reader):
    """
    LEGUS reader that requires HLSP tab/readme pairing and reads
    distance modulus + filter metadata from readme.
    """

    def read(self, fname, classcut=[0, 3.5]):
        readme_path = _find_hlsp_readme_for_catalog(fname)
        dmod, filters, first_mag_index, class_index, ra_index, debug_meta = _parse_hlsp_readme_metadata(readme_path)
        data = asc.read(fname)

        cid = np.array(data["col1"], dtype=int)
        nc = len(cid)
        nf = len(filters)
        phot = np.zeros((nc, nf))
        photerr = np.zeros((nc, nf))
        detect = np.ones((nc, nf), dtype=bool)
        nondetect_flags = np.array([44.444, 66.666, 99.999])
        for i in range(nf):
            mag_col = f"col{2 * i + first_mag_index}"
            err_col = f"col{2 * i + first_mag_index + 1}"
            mag_obs = np.array(data[mag_col], dtype=float)
            err_obs = np.array(data[err_col], dtype=float)
            is_flagged = np.isclose(mag_obs[:, None], nondetect_flags[None, :], atol=1.0e-6).any(axis=1)
            high_err = np.logical_not(np.isfinite(err_obs)) | (err_obs > 0.3)
            bad_mag = np.logical_not(np.isfinite(mag_obs))
            detect[:, i] = np.logical_not(is_flagged | high_err | bad_mag)
            phot[:, i] = np.where(detect[:, i], mag_obs - dmod, np.nan)
            photerr[:, i] = np.where(detect[:, i], err_obs, np.nan)

        if os.environ.get("LEGUS_DEBUG_HLSP_READ", "0") == "1":
            first_col = f"col{first_mag_index}"
            first_vals = np.array(data[first_col], dtype=float)
            print(f"[hlsp-debug] catalog={fname}")
            print(f"[hlsp-debug] readme={readme_path}")
            print(
                f"[hlsp-debug] readme dmod line: {debug_meta['dmod_line']} ; "
                f"parsed dmod={debug_meta['dmod_value']}"
            )
            print(
                f"[hlsp-debug] readme first-mag line: {debug_meta['first_mag_line']} ; "
                f"parsed first_mag_index={debug_meta['first_mag_index']}"
            )
            print(
                f"[hlsp-debug] tab first magnitude column={first_col} "
                "(apparent mag, one line per cluster as cid,value)"
            )
            for cid_i, mag_i in zip(cid, first_vals):
                print(f"[hlsp-debug] cid={int(cid_i)}, {first_col}={float(mag_i)}")

        ra = np.array(data[f"col{ra_index}"], dtype=float)
        dec = np.array(data[f"col{ra_index + 1}"], dtype=float)
        classification = np.array(data[f"col{class_index}"], dtype=int)

        keep_class = np.logical_and(classification > classcut[0], classification < classcut[1])

        # LEGUS quality selection:
        # - detected in V band (V = F555W when present, else F606W, else F547M)
        # - V-band absolute magnitude < -6
        # - detected in at least 4 filters
        # - detected in either B or I
        v_order = ["F555W", "F606W"]
        v_idx = None
        for token in v_order:
            for i, f in enumerate(filters):
                if token in f:
                    v_idx = i
                    break
            if v_idx is not None:
                break
        if v_idx is None:
            raise ValueError(f"No V-like filter (F555W/F606W) in parsed filters: {filters}")

        b_candidates = [i for i, f in enumerate(filters) if any(x in f for x in ["F435W", "F438W"])]
        i_candidates = [i for i, f in enumerate(filters) if "F814W" in f]
        b_detect = detect[:, b_candidates].any(axis=1) if len(b_candidates) > 0 else np.zeros(nc, dtype=bool)
        i_detect = detect[:, i_candidates].any(axis=1) if len(i_candidates) > 0 else np.zeros(nc, dtype=bool)

        keep_v_det = detect[:, v_idx]
        keep_v_mag = np.where(keep_v_det, phot[:, v_idx] < -6.0, False)
        keep_nflt = np.sum(detect, axis=1) >= 4
        keep_b_or_i = np.logical_or(b_detect, i_detect)
        # match with visual catalog classes 1, 2, and 3 (exclude class 0 and 4)
        keep_only_class_1_to_3 = np.logical_and(classification > 0, classification < 3.5)
        keep = keep_class & keep_v_det & keep_v_mag & keep_nflt & keep_b_or_i 
        galaxy_fullname = _infer_galaxy_fullname_from_path(fname)

        out = {
            "path": fname,
            "basename": osp.splitext(osp.basename(fname))[0],
            "galaxy_fullname": galaxy_fullname,
            "cid": cid[keep],
            "phot": phot[keep],
            "photerr": photerr[keep],
            "detect": detect[keep],
            "filters": filters,
            "dmod": dmod,
            "ra": ra[keep],
            "dec": dec[keep],
            "class": classification[keep],
            "viscat": True,
        }
        return out

##############################################
# Here we register the list of known readers #
##############################################

reader_register = {
    'LEGUS' : catalog_reader_legus_hlsp(),
    'LEGUS_all' : catalog_reader_LEGUS(),
    'LEGUS_div': catalog_reader_LEGUS_div(),
    'LEGUS_rOGC': catalog_reader_rOGC(),
    'LEGUS_cl12':catalog_reader_cl12(),
    'LEGUS_good_phot': catalog_reader_LEGUS_good_phot(),
    'LEGUS_with_EBV': catalog_reader_LEGUS_with_EBV()
    }
