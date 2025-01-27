import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee
from argparse import ArgumentParser
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('config', type=str, help='Configuration file')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite existing model files')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Turn on debugging mode')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_num_interop_threads(40) # Inter-op parallelism
    torch.set_num_threads(40) # Intra-op parallelism
torch.set_default_dtype(torch.double)
print('Using device: ',device)

#===============================================================================
config = Config(args.config)

print(config.probe_mask)
print(config.probe_size)


print(f'\n>>> Start Emulator Training\n')
if config.init_sample_type == "lhs":
    #label_train = f'{config.init_sample_type}_{config.n_lhs}'
    label_train = 'test_1000'
    label_valid = label_train
    N_sample_train = config.n_lhs
    N_sample_valid = 100
else:
    iss = f'{config.init_sample_type}'
    label_train = iss+f'_t{config.gtemp_t}_{config.gnsamp_t}'
    label_valid = iss+f'_t{config.gtemp_v}_{config.gnsamp_v}'
    N_sample_train = config.gnsamp_t
    N_sample_valid = config.gnsamp_v
#================== Loading Training & Validating Data =========================
print(f'Loading training data!')
#train_samples = np.load(pjoin(config.traindir, f'samples_{label_train}.npy'))
train_samples = np.load(pjoin(config.traindir, f'params_{label_train}.npy'))
train_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_train}.npy'))
#train_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_train}.npy'))
print(f'Training dataset dimension: {train_samples.shape}')
print(f'Loading validating data!')
#valid_samples = np.load(pjoin(config.traindir, f'samples_{label_valid}.npy'))
valid_samples = np.load(pjoin(config.traindir, f'params_{label_valid}.npy'))
valid_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_valid}.npy'))
#valid_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_valid}.npy'))
print(f'Validation dataset dimension: {valid_samples.shape}')
train_samples = torch.Tensor(train_samples)
train_data_vectors = torch.Tensor(train_data_vectors)
#train_sigma8 = torch.Tensor(train_sigma8)
valid_samples = torch.Tensor(valid_samples)
valid_data_vectors = torch.Tensor(valid_data_vectors)
#valid_sigma8 = torch.Tensor(valid_sigma8)
#================= Training emulator ===========================================
# switch according to probes
#print(config.probe_mask)
#probes = ["xi_pm", "gammat", "wtheta", "wgk", "wsk", "Ckk"]
#probes = ["xi_pm", "gammat", "wtheta"]
probes = ['xi_pm']
#probes = ["real.roman_real_3x2pt"]
#probes = ['3x2pt']
for i in range(len(probes)):
    print("============= Training %s Emulator ================="%(probes[i]))
    l, r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
    emu = NNEmulator(config.n_dim, config.probe_size[i], 
        config.dv_lkl[l:r], config.dv_std[l:r], 
        config.inv_cov[l:r,l:r],
        mask=config.mask_lkl[l:r], param_mask=config.probe_params_mask[i], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double",print_summary=True)
    emu_fn = pjoin(config.modeldir, f'{probes[i]}_nn{config.nn_model}')
    if (not os.path.exists(emu_fn)) or args.overwrite:
        emu.train(train_samples, train_data_vectors[:,l:r],
                valid_samples, valid_data_vectors[:,l:r],
                batch_size=config.batch_size, n_epochs=config.n_epochs, 
                loss_type=config.loss_type)
        emu.save(emu_fn)



