import sys, platform, os
sys.path.insert(0, os.environ['ROOTDIR'] + 
                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
                   +os.environ['PYTHON_VERSION'])
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
#import torch
from cocoa_emu import Config, get_params_list, CocoaModel, get_lhs_params_list
import emcee
from scipy.stats import qmc, norm

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model

from cobaya.conventions import packages_path_input
from cobaya.run import run

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)

#n = int(sys.argv[2])  
n=10

if config.init_sample_type == "lhs":
    label_train = f'{config.init_sample_type}_{config.n_lhs}'
    label_valid = label_train

def get_params_from_sample(sample, labels):
    """
    Format arrays into cocoa params
    """
    assert len(sample)==len(labels), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(labels):
        param_i = sample[i]
        params[label] = param_i
    return params

def get_params_list(samples, labels):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_sample(samples[i], labels)
        params_list.append(params)
    return params_list


def get_params_from_lhs_sample(unit_sample, priors, fid):
    assert len(unit_sample)==len(priors), "Length of the labels not equal to the length of samples"
    params = {}
    #for i in range(len(priors)):
    for i, label in enumerate(priors):
        if(label=='omegam'):
            print(label)
            lhs_min = priors[label]['min']
            lhs_max = priors[label]['max']
            param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[i]
        else:
            #param_i = qmc.scale(unit_sample[:,i], lhs_min, lhs_max)
            param_i = 0 * unit_sample[i] + fid[label]['loc']
        params[label] = param_i
    return params

def get_lhs_params_list(samples, priors, fid):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_lhs_sample(samples[i], priors, fid)
        params_list.append(params)
    return params_list

def get_lhs_samples(N_dim, N_lhs):
    sample = qmc.LatinHypercube(d=N_dim).random(N_lhs)
    priors = {}
    fid = {}
    for i in config.params:
        if('prior' in config.params[i]):
            priors[i] = config.params[i]['prior']
            fid[i] = config.params[i]['ref']
    lhs_params = get_lhs_params_list(sample, priors, fid)
    return lhs_params

#print(config.n_dim)
params_train = get_lhs_samples(config.n_dim, config.n_lhs)
#print(params_train)
#print(len(params_train))
np.save(pjoin(config.traindir,f'test_OM_params.npy'),params_train)

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank, label):
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}/{N_local}...')
        if type(params_list[i]) != dict:
            _p = {k:v for k,v in zip(config.running_params, params_list[i])}
        else:
            _p = params_list[i]
        params_arr  = np.array([_p[k] for k in config.running_params])
        # Here it calls cocoa to calculate data vectors at requested parameters
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
    return train_params_list, train_data_vector_list

def get_data_vectors(params_list, comm, rank, label):
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list,rank,label)
    comm.Barrier() # Synchronize before collecting results
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list], dest=0)
        train_params       = None
        train_data_vectors = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        for source in range(1,size):
            new_params_list, new_data_vector_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)
    return train_params, train_data_vectors

train_samples, train_data_vectors = get_data_vectors(params_train, comm, rank, label_train)

np.save(pjoin(config.traindir, f'data_vectors_test_OM.npy'),train_data_vectors)
np.save(pjoin(config.traindir, f'samples_test_OM.npy'), train_samples)
