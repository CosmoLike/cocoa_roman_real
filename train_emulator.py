import numpy as np
import torch
from cocoa_emu.nn_emulator import nn_emulator
import os
from sklearn.model_selection import train_test_split
from cocoa_emu.config import cocoa_config

def prepare_data(data_dir, config_file, sample_temp, cosmo, test_size=0.2, random_state=42, chi2_threshold=1e4):
    """
    Load and prepare data for training, filtering by chi^2 to fiducial datavector
    """
    # Load data
    samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_'+str(sample_temp)+'_'+cosmo+'.npy'))
    datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_'+str(sample_temp)+'_'+cosmo+'.npy'))

    config = cocoa_config(config_file) #This whole preparation can be done on the full 3x2 data vector
    dv_fid = config.dv_fid
    cov = config.cov
    mask = config.mask
    
    # Apply scale cuts via mask to both datavectors and covariance matrix
    print(f"Original datavector shape: {datavectors.shape}")
    print(f"Original covariance shape: {cov.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of unmasked elements: {np.sum(mask)}")
    
    # Apply mask to datavectors - keep only the masked elements
    datavectors_masked = datavectors[:, mask]
    dv_fid_masked = dv_fid[mask]
    
    # Apply mask to covariance matrix - keep only the masked elements
    cov_masked = cov[mask][:, mask]
    
    print(f"Masked datavector shape: {datavectors_masked.shape}")
    print(f"Masked covariance shape: {cov_masked.shape}")
    
    # Compute inverse covariance for masked data
    inv_cov_masked = config.inv_cov_masked
    
    # Compute chi^2 for each datavector using masked data
    diff = datavectors_masked - dv_fid_masked
    chi2 = np.einsum('ij,jk,ik->i', diff, inv_cov_masked, diff)


    print(np.mean(chi2))
    print(np.median(chi2))
    print(np.min(chi2))
    print(np.max(chi2))

    chi2_threshold = 1e5
    
    # Filter by chi^2 threshold
    mask_chi2 = chi2 < chi2_threshold
    filtered_samples = samples[mask_chi2]
    filtered_datavectors = datavectors_masked[mask_chi2]
    
    # Split into training and validation sets
    train_samples, valid_samples, train_dv, valid_dv = train_test_split(
        filtered_samples, filtered_datavectors, test_size=test_size, random_state=random_state
    )


    print(f"Total samples loaded: {len(samples)}")
    print(f"Samples after chi^2 cut: {np.sum(mask_chi2)}")
    print(f"Train set after cut: {len(train_samples)}")
    print(f"Valid set after cut: {len(valid_samples)}")
    
    return train_samples, valid_samples, train_dv, valid_dv 

def main():
    # Set paths
    full_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real_emu'
    data_dir = full_dir+'/chains/training_data'

    cosmo = 'lcdm'
    sample_temp = 512


    config_main = full_dir+'/DC1_3x2_'+cosmo.upper()+'_PARAMS.yaml'
    full_param_list = np.loadtxt(data_dir+'/dc1_3x2_param_names_'+str(sample_temp)+'_'+cosmo+'.txt',dtype='U12')

    #probes = ['3x2']
    version = 'testing'
    #probes = ['xi','GGL','W']
    probes = ['W']
    model_save_dir = full_dir+'/chains/emulator'
    os.makedirs(model_save_dir, exist_ok=True)

    emulator_type = {'3x2': '3x2_restrf', 'xi': 'xi_restrf', 'GGL': 'xi_restrf', 'W': 'xi_restrf'}

    # Training parameters
    n_epochs = 250
    batch_size = 256
    learning_rate = 1e-3
    reduce_lr = True
    weight_decay = 0


    # Check CUDA availability and set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    for probe in probes:
        config_file = full_dir+'/DC1_'+probe+'_'+cosmo.upper()+'_PARAMS.yaml'

        train_samples_file = os.path.join(data_dir, 'train_samples_'+probe+'_'+str(sample_temp)+'_'+cosmo+'.npy')
        train_dv_file = os.path.join(data_dir, 'train_datavectors_'+probe+'_'+str(sample_temp)+'_'+cosmo+'.npy') 
        valid_samples_file = os.path.join(data_dir, 'valid_samples_'+probe+'_'+str(sample_temp)+'_'+cosmo+'.npy')
        valid_dv_file = os.path.join(data_dir, 'valid_datavectors_'+probe+'_'+str(sample_temp)+'_'+cosmo+'.npy')

        if os.path.exists(train_samples_file):
            #Load the existing set of files
            print("Loading previously processed data...")
            train_samples = np.load(train_samples_file)
            train_dv = np.load(train_dv_file)
            valid_samples = np.load(valid_samples_file)
            valid_dv = np.load(valid_dv_file)

        else:
            #Prepare the data
            print("Preparing data...")
            train_samples, valid_samples, train_dv, valid_dv = prepare_data(data_dir,config_file,sample_temp,cosmo)

            # Save data files
            np.save(train_samples_file, train_samples)
            np.save(train_dv_file, train_dv)
            np.save(valid_samples_file, valid_samples)
            np.save(valid_dv_file, valid_dv)
            print("Data saved to file...")
    
        # Initialize emulator with correct output dimension
        print("Initializing emulator...")

        config = cocoa_config(config_file)

        emulator = nn_emulator(preset=emulator_type[probe], output_dim=len(train_dv[0]), input_dim=len(train_samples[0]))
        
        # Train emulator
        print(f"Training emulator on {device}...")
        emulator.train(
            device=device,
            config_file=config_file,
            x_train=train_samples,
            y_train=train_dv,
            x_valid=valid_samples,
            y_valid=valid_dv,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            reduce_lr=reduce_lr,
            weight_decay=weight_decay,
            save_losses=True
        )
        # Save trained model
        model_path = os.path.join(model_save_dir, 'dc1_'+probe+'_'+str(sample_temp)+'_emulator_'+cosmo+'_'+version+'.pt')
        print(f"Saving model to {model_path}")
        emulator.save(model_path)
        print("Training for "+probe+" complete!")




if __name__ == "__main__":
    main() 