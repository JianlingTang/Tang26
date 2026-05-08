import numpy as np
import os.path as osp
from astropy.io import ascii as asc
import os 
import re 
import glob
class catalog_reader(object):
    """
    This is a class whose job is to read observational catalogs. It is
    a purely abstract class that defines the minimum set entires that
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
            galaxies = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy')
            gal_filters = np.load('/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy', allow_pickle=True).item()
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
            galdir = os.path.join('/g/data/jh2/jt4478/make_LEGUS_CCT', gal_name)
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
            readme_cluster_file = '/home/100/jt4478/slugfiles/LEGUS_cat/ngc628.readme'
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
        
##############################################
# Here we register the list of known readers #
##############################################

reader_register = {
    'LEGUS' : catalog_reader_simple(),
    'LEGUS_all' : catalog_reader_LEGUS(),
    'LEGUS_div': catalog_reader_LEGUS_div(),
    'LEGUS_rOGC': catalog_reader_rOGC(),
    'LEGUS_cl12':catalog_reader_cl12(),
    'LEGUS_good_phot': catalog_reader_LEGUS_good_phot(),
    'LEGUS_with_EBV': catalog_reader_LEGUS_with_EBV()
    }