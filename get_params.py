import sys, platform, os
sys.path.insert(0, os.environ['ROOTDIR'] + 
                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
                   +os.environ['PYTHON_VERSION'])
from os.path import join as pjoin
import numpy as np
from cocoa_emu import Config, get_params_list, get_lhs_params_list
from scipy.stats import qmc, norm

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info

configfile = sys.argv[1]
config = Config(configfile)

label_train = f'{config.init_sample_type}_{config.n_lhs}'
label_valid = label_train

print("Initializing configuration space data vector dimension!")
print("N_xi_pm: %d"%(config.probe_size[0]))
print("N_ggl: %d"%(config.probe_size[1]))
print("N_w: %d"%(config.probe_size[2]))
print("N_gk: %d"%(config.probe_size[3]))
print("N_sk: %d"%(config.probe_size[4]))
print("N_kk: %d"%(config.probe_size[5]))

def get_params_from_lhs_sample(unit_sample, priors):
    assert len(unit_sample)==len(priors), "Length of the labels not equal to the length of samples"
    
    params = {}
    for i, label in enumerate(priors):
        if('dist' in priors[label]):
            lhs_loc = priors[label]['loc']
            lhs_scale = priors[label]['scale']
            param_i = norm(loc=lhs_loc,scale=lhs_scale).ppf(unit_sample[i])
        else:
            lhs_min = priors[label]['min']
            lhs_max = priors[label]['max']
            #param_i = qmc.scale(unit_sample[:,i], lhs_min, lhs_max)
            param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[i]
        params[label] = param_i
    return params

def get_lhs_params_list(samples, priors):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_lhs_sample(samples[i], priors)
        params_list.append(params)
    return params_list

def get_lhs_samples(N_dim, N_lhs):
    ''' Generate Latin Hypercube sample at parameter space
    Input:
    ======
        - N_dim: 
            Dimension of parameter space
        - N_lhs:
            Number of LH grid per dimension in the parameter space
        - lhs_minmax:
            The boundary of parameter space along each dimension
    Output:
    =======
        - lhs_params:
            LHS of parameter space
    '''
    sample = qmc.LatinHypercube(d=N_dim).random(N_lhs)
    priors = {}
    for i in config.params:
        if('prior' in config.params[i]):
            priors[i] = config.params[i]['prior']
    lhs_params = get_lhs_params_list(sample, priors)
    return lhs_params

params_train = get_lhs_samples(config.n_dim, config.n_lhs)
np.save(pjoin(config.traindir,f'total_samples_{label_train}.npy'),params_train)