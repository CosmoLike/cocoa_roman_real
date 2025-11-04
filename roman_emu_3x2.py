#Call emu as cobaya likelihood
from cobaya.likelihood import Likelihood
import numpy as np
import os
import sys
import torch
### Replaced with load/predict function below; be careful with normalization choicies
#from projects.lsst_y1 import cocoa_emu
sys.path.append('./')
#from projects.roman_real_emu.cocoa_emu.nn_emulator import nn_pca_emulator 
from projects.roman_real_emu.cocoa_emu.config import cocoa_config
#from projects.roman.cocoa_emu.nn_emulator import Affine,ResBlock,Better_Transformer,Better_Attention
from projects.roman_real_emu.cocoa_emu.nn_emulator import nn_emulator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5

import sys
#sys.path.insert(0, './projects/roman_real_emu_emu/emulator_output/models/')

class roman_emu_3x2(Likelihood):
    def initialize(self):
        super(roman_emu_3x2,self)
        torch.set_num_threads(1)
        self.n_pcas_baryon     = 0
        self.output_dims       = 1950 #Need a way to set this automatically

        sample_temp=512 #was 512
        #### pick the correct configuration -- TO DO: NEED TO GET FROM COBAYA AND NOT HERE
        self.full_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real_emu'
        self.model_path = os.path.join(self.full_dir, 'chains/emulator')
        self.config_file = os.path.join(self.full_dir, 'DC1_emu_EVAL.yaml')
        #self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        #model1_path = os.path.join(self.model_path, 'dc1_xi_emulator_w0wa_'+str(sample_temp)+'_v2.pt')
        model1_path = os.path.join(self.model_path, 'dc1_xi_'+str(sample_temp)+'_emulator_lcdm_testing.pt')
        self.emu1 = nn_emulator(preset='xi_restrf', output_dim=1080, input_dim=15)
        self.emu1.load(model1_path, self.config_file, self.device)

        #model2_path = os.path.join(self.model_path, 'dc1_GGL_emulator_w0wa_'+str(sample_temp)+'_v2.pt')
        model2_path = os.path.join(self.model_path, 'dc1_GGL_'+str(sample_temp)+'_emulator_lcdm_testing.pt')
        self.emu2 = nn_emulator(preset='xi_restrf', output_dim=769, input_dim=15)
        self.emu2.load(model2_path, self.config_file, self.device)

        #model3_path = os.path.join(self.model_path, 'dc1_W_emulator_w0wa_'+str(sample_temp)+'_v2.pt')
        model3_path = os.path.join(self.model_path, 'dc1_W_'+str(sample_temp)+'_emulator_lcdm_testing.pt')
        self.emu3 = nn_emulator(preset='xi_restrf', output_dim=101, input_dim=15)
        self.emu3.load(model3_path, self.config_file, self.device)

        #if hasattr(torch, 'compile'):
        #    print("Applying torch.compile optimization")
        #    self.emu1.model = torch.compile(self.emu1.model, mode="reduce-overhead")
        #    self.emu2.model = torch.compile(self.emu2.model, mode="reduce-overhead")
        #    self.emu3.model = torch.compile(self.emu3.model, mode="reduce-overhead")
        #    print("Torch compilation complete")
        #else:
        #    print("PyTorch version < 2.0")



        self.config = cocoa_config(self.config_file)
        cov = self.config.cov
        self.dv = self.config.dv_fid
        self.mask = self.config.mask
        self.cov_inv_masked = np.linalg.inv(cov[self.mask][:,self.mask])

        self.shear_calib_mask = np.load('./projects/roman_real_emu/shear_calib_mask.npy')[:,self.mask]
        self.galaxy_bias_mask = np.load('./projects/roman_real_emu/cocoa_emu/linear_galbias_mask.npy')[:,self.mask]
        self.gal_bias_fid= np.array([1.18,1.40,1.55,1.71,1.90,2.15,2.52,3.44])
        self.n_fast_pars = 16
        self.source_ntomo = 8
        self.dv_len = 1950

    def get_requirements(self):
        return {
          "As_1e9": None,
          "H0": None,
          "ns": None,
          "omegab": None,
          "omegam": None,
          "w0pwa": None,
          "w": None,
          "roman_DZ_S1": None,
          "roman_DZ_S2": None,
          "roman_DZ_S3": None,
          "roman_DZ_S4": None,
          "roman_DZ_S5": None,
          "roman_DZ_S6": None,
          "roman_DZ_S7": None,
          "roman_DZ_S8": None,
          "roman_A1_1": None,
          "roman_A1_2": None,
          "roman_M1": None,
          "roman_M2": None,
          "roman_M3": None,
          "roman_M4": None,
          "roman_M5": None,
          "roman_M6": None,
          "roman_M7": None,
          "roman_M8": None,
          "roman_B1_1": None,
          "roman_B1_2": None,
          "roman_B1_3": None,
          "roman_B1_4": None,
          "roman_B1_5": None,
          "roman_B1_6": None,
          "roman_B1_7": None,
          "roman_B1_8": None,
        }

    def get_theta(self, **params_values):

      theta = np.array([])

      # 7 cosmological parameter for w0waCDM
      #logAs = self.provider.get_param("logA")
      As_1e9 = self.provider.get_param("As_1e9")
      ns = self.provider.get_param("ns")
      H0 = self.provider.get_param("H0")
      omegab = self.provider.get_param("omegab")
      omegam = self.provider.get_param("omegam")
      #w0pwa = self.provider.get_param("w0pwa")
      #w = self.provider.get_param("w")

      # 10 nuissance parameter emulated
      roman_DZ_S1 = self.provider.get_param('roman_DZ_S1')#params_values['LSST_DZ_S1']
      roman_DZ_S2 = self.provider.get_param('roman_DZ_S2')#params_values['LSST_DZ_S2']
      roman_DZ_S3 = self.provider.get_param('roman_DZ_S3')#params_values['LSST_DZ_S3']
      roman_DZ_S4 = self.provider.get_param('roman_DZ_S4')#params_values['LSST_DZ_S4']
      roman_DZ_S5 = self.provider.get_param('roman_DZ_S5')#params_values['LSST_DZ_S5']
      roman_DZ_S6 = self.provider.get_param('roman_DZ_S6')#params_values['LSST_DZ_S3']
      roman_DZ_S7 = self.provider.get_param('roman_DZ_S7')#params_values['LSST_DZ_S4']
      roman_DZ_S8 = self.provider.get_param('roman_DZ_S8')#params_values['LSST_DZ_S5']

      roman_A1_1 = self.provider.get_param('roman_A1_1')#params_values['LSST_A1_1']
      roman_A1_2 = self.provider.get_param('roman_A1_2')#params_values['LSST_A1_2']

      # 8 fast parameters don't emulate, no baryons for now
      roman_M1 = self.provider.get_param('roman_M1')#params_values['LSST_M1']
      roman_M2 = self.provider.get_param('roman_M2')#params_values['LSST_M2']
      roman_M3 = self.provider.get_param('roman_M3')#params_values['LSST_M3']
      roman_M4 = self.provider.get_param('roman_M4')#params_values['LSST_M4']
      roman_M5 = self.provider.get_param('roman_M5')#params_values['LSST_M5']
      roman_M6 = self.provider.get_param('roman_M6')#params_values['LSST_M3']
      roman_M7 = self.provider.get_param('roman_M7')#params_values['LSST_M4']
      roman_M8 = self.provider.get_param('roman_M8')#params_values['LSST_M5']

      # 8 more parameters that I don't emulate
      roman_B1 = self.provider.get_param('roman_B1_1')#params_values['LSST_M1']
      roman_B2 = self.provider.get_param('roman_B1_2')#params_values['LSST_M2']
      roman_B3 = self.provider.get_param('roman_B1_3')#params_values['LSST_M3']
      roman_B4 = self.provider.get_param('roman_B1_4')#params_values['LSST_M4']
      roman_B5 = self.provider.get_param('roman_B1_5')#params_values['LSST_M5']
      roman_B6 = self.provider.get_param('roman_B1_6')#params_values['LSST_M3']
      roman_B7 = self.provider.get_param('roman_B1_7')#params_values['LSST_M4']
      roman_B8 = self.provider.get_param('roman_B1_8')#params_values['LSST_M5']
      
      return np.array([As_1e9,ns,H0,omegab,omegam,
                    roman_DZ_S1,roman_DZ_S2,roman_DZ_S3,roman_DZ_S4,roman_DZ_S5,
                    roman_DZ_S6,roman_DZ_S7,roman_DZ_S8,
                    roman_A1_1,roman_A1_2,
                    roman_B1,roman_B2,roman_B3,roman_B4,roman_B5,
                    roman_B6,roman_B7,roman_B8,
                    roman_M1,roman_M2,roman_M3,roman_M4,roman_M5,
                    roman_M6,roman_M7,roman_M8])

    # Get the dv from emulator
    #def compute_datavector(self, theta):        
    #    param = np.copy(theta)

    #    predicted_xi = self.emu1.predict(torch.Tensor(param))[0]
    #    predicted_gamma = self.emu2.predict(torch.Tensor(param))[0]
    #    predicted_w = self.emu3.predict(torch.Tensor(param))[0]
    #    predicted_datavector = np.concatenate((predicted_xi,predicted_gamma,predicted_w),axis=0)
        
    #    print(np.shape(predicted_datavector))
        #print(datavector)

    #    return predicted_datavector

    # add the fast parameter part into the dv
    def get_data_vector_emu(self, theta):
        theta_emu     = theta[:-self.n_fast_pars]
        bias_theta    = theta[(len(theta)-self.n_fast_pars):(len(theta)-8)]
        m_shear_theta = theta[(len(theta)-8):]

        #print("TESTING theta_emu=", theta_emu)
        
        datavector_xi = self.emu1.predict(theta_emu)[0]
        #datavector_xi = self.add_shear_calib(m_shear_theta, datavector_xi)

        datavector_gamma = self.emu2.predict(theta_emu)[0]
        datavector_w = self.emu3.predict(theta_emu)[0]

        #bias_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo):
        #                           self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo)]
        
        #datavector_gamma = self.add_bias(bias_theta, datavector_gamma)

        datavector = np.concatenate((datavector_xi,datavector_gamma,datavector_w),axis=0)
        datavector = self.add_shear_calib(m_shear_theta, datavector)
        datavector = self.add_bias(bias_theta, datavector)
        #datavector = self.compute_datavector(theta_emu)
        #np.savetxt("test_emu.datavector", datavector)
        

        if np.isnan(datavector).any():
            print('nan encountered with params: ',theta_emu)
        #print('dv after m_shear',datavector[0:20])
        # if(self.n_pcas_baryon > 0):
        #     baryon_q   = theta[-self.n_pcas_baryon:]
        #     #print("TESTING baryon_q=", baryon_q)
        #     datavector = self.add_baryon_q(baryon_q, datavector)
        return datavector

    def add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i][0:self.dv_len]
        return datavector

    def add_shear_calib(self, m, datavector):
        #shear_calib_mask = np.load('shear_calib_mask.npy')
        #print('TESTING shear_calib_mask=', shear_calib_mask)
        factors = ((1 + m[:, None]) ** self.shear_calib_mask[:, :self.dv_len])
        factor = factors.prod(axis=0)
        #for i in range(self.source_ntomo):
        #    factor = (1 + m[i])**self.shear_calib_mask[i]
        #    factor = factor[0:self.dv_len] # for cosmic shear
        #    datavector = factor * datavector
        return factor * datavector

    def add_bias(self, b1, datavector):
        #if self.fast_linear_gal_bias:
        factors = ((b1/self.gal_bias_fid)[:,None] ** self.galaxy_bias_mask[:, :self.dv_len])
        factor = factors.prod(axis=0)
        #for i in range(self.source_ntomo):
        #    factor = (b1[i]/self.gal_bias_fid[i])**self.galaxy_bias_mask[i]
        #    datavector = factor * datavector
        #return datavector
        return factor * datavector


    def logp(self, **params_values):
        theta = self.get_theta(**params_values)
        model_datavector = self.get_data_vector_emu(theta)
        #print("TESTING model_datavector=", model_datavector)
        #print(len(self.dv))
        delta_dv = (model_datavector - self.dv[self.mask])
        #print("TESTING delta_dv=", delta_dv)
        log_p = -0.5 * delta_dv @ self.cov_inv_masked @ delta_dv 
        return log_p