import sys, platform, os
sys.path.insert(0, os.environ['ROOTDIR'] + 
                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
                   +os.environ['PYTHON_VERSION'])

#print(sys.executable)

from os.path import join as pjoin
import numpy as np
#import torch
from cocoa_emu import Config, get_params_list, CocoaModel, get_lhs_params_list
import emcee

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model

from cobaya.conventions import packages_path_input
from cobaya.run import run

import time

start_time = time.process_time()

configfile = sys.argv[1]
config = Config(configfile)

end_time = time.process_time()

print(f"to load config file: {end_time - start_time} seconds")

rank = int(sys.argv[2])
size = int(sys.argv[3])


if(rank==0):
    print("Initializing configuration space data vector dimension!")
    print("N_xi_pm: %d"%(config.probe_size[0]))
    print("N_ggl: %d"%(config.probe_size[1]))
    print("N_w: %d"%(config.probe_size[2]))
    print("N_gk: %d"%(config.probe_size[3]))
    print("N_sk: %d"%(config.probe_size[4]))
    print("N_kk: %d"%(config.probe_size[5]))

label_train = 'lhs_1000'

params_train = np.load(pjoin(config.traindir,f'total_samples_{label_train}.npy'),allow_pickle=True)

# ================== Calculate data vectors ==========================
start_time = time.process_time()

cocoa_model = CocoaModel(configfile, config.likelihood)

end_time = time.process_time()

print(f"to initialize cocoa model: {end_time - start_time} seconds")

def get_local_data_vector_list(params_list, rank, label):
    ''' Evaluate data vectors dispatched to the local process
    Input:
    ======
        - params_list: 
            full parameters to be evaluated. Parameters dispatched is a subset of the full parameters
        - rank: 
            the rank of the local process
    Outputs:
    ========
        - train_params: model parameters of the training sample
        - train_data_vectors: data vectors of the training sample
    '''
    # print results real time 
    print(f'rank: {rank}')
    print(f'size: {size}')
    dump_file = pjoin(config.traindir, f'dump/{label}_{rank}-{size}_cs.txt')
    fp = open(dump_file, "w")
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        print(f'running {i}')
        #print(params_list[i])
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}/{N_local}...')
        if type(params_list[i]) != dict:
            _p = {k:v for k,v in zip(config.running_params, params_list[i])}
        else:
            _p = params_list[i]
        params_arr  = np.array([_p[k] for k in config.running_params])
        #print(params_list[i])
        # Here it calls cocoa to calculate data vectors at requested parameters
        start_time = time.process_time()
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        end_time = time.process_time()
        print(f"to calculate 1 datavector: {end_time - start_time} seconds")
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        context = ' '.join([f'{num:e}' for num in np.hstack([params_arr, data_vector])])
        fp.write(context+"\n")
        fp.flush()
    fp.close()
    return train_params_list, train_data_vector_list

local_params_list, local_data_vector_list = get_local_data_vector_list(params_train,rank,label_train)