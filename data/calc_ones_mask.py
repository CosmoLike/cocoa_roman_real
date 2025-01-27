import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import math as mt

ggl_efficiency_cut = [0.05]

#all ones cut
ξp_CUTOFF = 44.096 # cutoff scale in arcminutes
ξm_CUTOFF = 139.128 # cutoff scale in arcminutes
gc_CUTOFF = 21 # Galaxy clustering cutoff in Mpc/h (should be 21)

#comsic sheer - lmin =30, lmax =3000
#galaxy clustering - exclude high-l bins (Rmin = 21 Mpc/h)
#ggl

#VM GLOBAL VARIABLES -------------------------------------------------------
THETA_MIN  = 2.5    # Minimum angular scale (in arcminutes)
THETA_MAX  = 250.  # Maximum angular scale (in arcminutes)
N_ANG_BINS = 15    # Number of angular bins
N_LENS = 10  # Number of lens tomographic bins
N_SRC  = 10  # Number of source tomographic bins
N_XI_PS = int(N_SRC * (N_SRC + 1) / 2) 
N_XI    = int(N_XI_PS * N_ANG_BINS)

# COMPUTE SHEAR SCALE CUTS
vtmin = THETA_MIN * 2.90888208665721580e-4;
vtmax = THETA_MAX * 2.90888208665721580e-4;
logdt = (mt.log(vtmax) - mt.log(vtmin))/N_ANG_BINS;
theta = np.zeros(N_ANG_BINS+1)

for i in range(N_ANG_BINS):
    tmin = mt.exp(mt.log(vtmin) + (i + 0.0) * logdt);
    tmax = mt.exp(mt.log(vtmin) + (i + 1.0) * logdt);
    x = 2./ 3.
    theta[i] = x * (tmax**3 - tmin**3) / (tmax**2- tmin**2)
    theta[i] = theta[i]/2.90888208665721580e-4

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
def ang_cut(z):
    "Get Angular Cutoff from redshit z"
    theta_rad = gc_CUTOFF / cosmo.angular_diameter_distance(z).value
    return theta_rad * 180. / np.pi * 60

zavg = [0.195,
 0.470,
 0.625,
 0.775,
 0.930,
 1.1,
 1.30,
 1.555,
 1.93,
 3.08]

#zavg = [0.25,0.5,0.7,0.9,1.2,1.45,1.8,2.6]

#VM COSMIC SHEAR SCALE CUT -------------------------------------------------
ξp_mask = np.hstack([(theta[:-1] > ξp_CUTOFF) for i in range(N_XI_PS)])
ξm_mask = np.hstack([(theta[:-1] > ξm_CUTOFF) for i in range(N_XI_PS)])
#ξp_mask = np.hstack([1.0 for i in range(N_XI_PS)])
#ξm_mask = np.hstack([1.0 for i in range(N_XI_PS)])


γt_mask = []
for i in range(N_LENS):
    for j in range(N_SRC):
        if i>j:
            γt_mask.append(np.zeros(N_ANG_BINS))
        #γt_mask.append(np.ones(N_ANG_BINS))
        else:
            γt_mask.append((theta[:-1] > ang_cut(zavg[i])))
γt_mask = np.hstack(γt_mask)

#VM w_theta mask -----------------------------------------------------------
w_mask = np.hstack([(theta[:-1] > ang_cut(zavg[i])) for i in range(N_LENS)])
#w_mask = np.hstack([1.0 for i in range(N_LENS)])

#VM output -----------------------------------------------------------------
mask = np.hstack([ξp_mask, ξm_mask, γt_mask, w_mask])
np.savetxt("Roman_ones.mask", np.column_stack((np.arange(0,len(mask)), np.ones(len(mask)))),fmt='%d %1.1f')