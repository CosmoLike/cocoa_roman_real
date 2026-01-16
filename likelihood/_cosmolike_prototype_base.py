# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time

#Adding baryon package imports
import pyspk as spk
import BCemu
from astropy.cosmology import FlatLambdaCDM

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

import euclidemu2 as ee2
import math

import cosmolike_roman_real_interface as ci

survey = "roman"

class _cosmolike_prototype_base(DataSetLikelihood):

  def initialize(self, probe):
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo") #5
    self.source_ntomo = ini.int("source_ntomo") #4
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")
    
    # ------------------------------------------------------------------------   
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp))),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp))),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    tmp=int(min(120 + 20*self.accuracyboost,250))
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
                                       np.linspace(3.01,50.1,max(30,int(0.25*tmp)))),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------
    
    ci.initial_setup()
    
    ci.init_probes(possible_probes=self.probe)

    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    ci.init_ggl_exclude(np.array(self.ggl_exclude).flatten())

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    if self.use_emulator:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_ntable_lmax(lmax=int(self.lmax))

      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))

      ci.init_cosmo_runmode(is_linear=False)

      if self.external_nz_modeling: 
        (  self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 
      
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      ci.init_IA(ia_model = int(self.IA_model), 
                 ia_redshift_evolution = int(self.IA_redshift_evolution))
     
      if self.probe != "xi":
        # (b1, b2, bs2, b3, bmag). 0 = one amplitude per bin
        ci.init_bias(bias_model=self.bias_model)
      
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()
      
      if self.baryon_suppression != 0:
        self.use_baryon_pca = False

      if self.create_baryon_pca:
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=allsims)

    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')
  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    if self.use_emulator:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.baryon_suppression == 1: #SP(k)
      return {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "alpha_spk": None, #starting baryon code
        "beta_spk": None,
        "gamma_spk": None,
         #ending baryon code 
        "Pk_interpolator": {
          "z": self.z_interp_2D,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
    elif self.baryon_suppression == 2: #BCemu
      return {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "log10Mc_bcemu": None, #starting baryon code
        "mu_bcemu": None,
        "thej_bcemu": None,
        "gamma_bcemu": None,
        "delta_bcemu": None,
        "eta_bcemu": None,
        "deta_bcemu": None,
         #ending baryon code 
        "Pk_interpolator": {
          "z": self.z_interp_2D,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
    else:
      return {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
      }
      }

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0
    if not self.use_emulator:
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : 0.0
        }
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D, 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array([tmp_bt[i] for i in range(self.len_z_interp_2D)],dtype='float64')  
        lnbt = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        lnbt[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnPNL=(lnPL.reshape(self.len_z_interp_2D, 
                            self.len_log10k_interp_2D, 
                            order='F') + lnbt).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", self.non_linear_emul)

      G_growth = np.sqrt(PKL.P(self.z_interp_2D,0.0005)/PKL.P(0,0.0005))*(1+self.z_interp_2D)
      G_growth /= G_growth[-1]

      #Adding baryons here. Following Kunhao, iterating every z-specified. Calling suppression before nonlinear
      #pk calculation

      if self.baryon_suppression == 1: #SP(k) version
        #if self.non_linear_emu == 2:
          #raise NotImplementedError("Baryon suppression with non_linear_emu = 2 not implemented yet")
        
        cosmo = FlatLambdaCDM(H0=self.provider.get_param("H0"), Om0 = self.provider.get_param("omegam"),)

        alpha = self.provider.get_param("alpha_spk")
        beta = self.provider.get_param("beta_spk")
        gamma = self.provider.get_param("gamma_spk")
        kmin_spk = 8.73e-3 #in h/Mpc (copying from Euclidemu2 min k)
        kmax_spk = 8 # in h/Mpc (Spk says might be inaccurate above k = 8)
        for i, this_z in enumerate(self.z_interp_2D):
          #Note that SPk only works for z<3, going to assume high z doesn't matter
          #need to test calibration at high z.
          if this_z < 3. and this_z>0.125: #Limits set by Spk improve chi2
            #using method 2 for now, will add more methods later
            k_spk, sup = spk.sup_model(SO=500, z = this_z, alpha = alpha, beta = beta, gamma = gamma, cosmo = cosmo, 
                                       k_min = kmin_spk, k_max = kmax_spk, verbose = False)
            
            
            interp_spk = interp1d(np.log10(k_spk), 
                                  np.log10(sup), kind = 'linear', fill_value = 'extrapolate', assume_sorted = True)
            
            lnbt_spk = interp_spk(self.log10k_interp_2D)
            lnbt_spk[np.power(10,self.log10k_interp_2D)<kmin_spk] = 0.0 #Kunhao masked this

            lnPNL[i::self.len_z_interp_2D] += lnbt_spk
      elif self.baryon_suppression == 2: #BCemu version
        bfcemu = BCemu.BCM_7param(Ob= self.provider.get_param("omegab"), Om = self.provider.get_param("omegam"))
        bcmdict = {
          'log10Mc': self.provider.get_param("log10Mc_bcemu"),
          'mu': self.provider.get_param("mu_bcemu"),
          'thej': self.provider.get_param("thej_bcemu"),
          'gamma': self.provider.get_param("gamma_bcemu"),
          'delta': self.provider.get_param("delta_bcemu"),
          'eta': self.provider.get_param("eta_bcemu"),
          'deta': self.provider.get_param("deta_bcemu")
        }
        kmin_bcemu = 0.034 #from BCemu
        kmax_bcemu = 12.517 #from BCemu
        logkmin_bcemu = np.log10(kmin_bcemu)
        logkmax_bcemu = np.log10(kmax_bcemu)
        for i, this_z in enumerate(self.z_interp_2D):
          #Note that SPk only works for z<3, going to assume high z doesn't matter
          #need to test calibration at high z.
          if this_z < 2. and this_z>0: #Limits set by BCemu training
            #using method 2 for now, will add more methods later
            k_eval = 10**np.linspace(logkmin_bcemu, logkmax_bcemu, 100) #h/Mpc
            sup_k = bfcemu.get_boost(this_z, bcmdict, k_eval)

            interp_bcemu = interp1d(np.log10(k_eval), 
                                  np.log10(sup_k), kind = 'linear', fill_value = 'extrapolate', assume_sorted = True)
            
            lnbt_bcemu = interp_bcemu(self.log10k_interp_2D)
            lnbt_bcemu[np.power(10,self.log10k_interp_2D)<kmin_bcemu] = 0.0 #mask at small k to not suppress

            lnPNL[i::self.len_z_interp_2D] += lnbt_bcemu

      #End baryons here

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
    else:
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_source_related(self, **params):
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not self.use_emulator:
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
       
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        source_nz_local = self.source_nz.copy()

        # insert mod function here <-
        #source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params):
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not self.use_emulator:
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
       
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        lens_nz_local = self.lens_nz.copy()

        # insert mod function here <-
        #lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  
  def logp(self, **params):
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_datavector(self, **params):        
    if self.use_emulator:
      dv = self.internal_get_datavector_emulator(**params)
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    # ---------------------------------------------------------------
    # fast parameters: m's and pm's are never emulated
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    if not self.use_baryon_pca: 
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
