import numpy as np
import torch
from nn_emulator import nn_emulator
import os
from cocoa_emu import cocoa_config
import yaml
import matplotlib.pyplot as plt

def load_config(config_file):
    """Load parameter configuration from YAML file"""
    config = cocoa_config(config_file)
    return config

def calculate_chi2(true_values, predicted_values, inv_cov=None):
    """Calculate chi^2 between true and predicted values"""
    residuals = true_values - predicted_values
    if inv_cov is None:
        # If no covariance matrix provided, assume diagonal with unit variance
        return np.sum(residuals**2, axis=1)
    else:
        # With covariance matrix
        if np.shape(residuals)==(1080,):
            chi2 = residuals @ inv_cov @ residuals
        else:
            chi2 = np.zeros(len(residuals))
            for i in range(len(residuals)):
                chi2[i] = residuals[i] @ inv_cov @ residuals[i]
            #chi2 = np.sum(chi2)
        return chi2

def get_true_samples(data_dir, config_file, sample_temp, cosmo):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_'+str(sample_temp)+'_'+cosmo+'.npy'))
    datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_'+str(sample_temp)+'_'+cosmo+'.npy'))

    return samples, datavectors

def get_pred_samples(model_dir, model_name, test_samples, test_dv_size, preset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = os.path.join(model_dir, model_name)
    model = nn_emulator(preset=preset, input_dim=test_samples.shape[1], output_dim=test_dv_size)
    model.load(model_path, config_file, device)

    # Predict datavectors
    test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
    predicted_datavectors = model.predict(test_samples_tensor)

    return predicted_datavectors

def plot_chi_squared_distribution(chi2_values, bins=100):
    """Plot the chi^2 distribution"""
    plt.hist(chi2_values, bins=np.logspace(np.log10(chi2_values.min()),np.log10(chi2_values.max())), density=True, alpha=0.5)
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('Density')
    plt.xlim(1e-4,100)
    plt.xscale('log')
    #plt.savefig(os.path.join(output_dir, 'chi2_distribution_3x2_lcdm.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'chi2_distribution_W_lcdm.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Configuration
    full_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real_emu'
    data_dir = os.path.join(full_dir, 'chains/training_data')
    model_dir = os.path.join(full_dir, 'chains/emulator')
    config_file = os.path.join(full_dir, 'DC1_W_LCDM_PARAMS.yaml')
    model_name = 'dc1_W_512_emulator_lcdm_testing.pt'  # Update with actual model filename
    sample_temp = 256
    cosmo = 'lcdm'
    preset = 'xi_restrf'
    
    # Load configuration
    config = load_config(config_file)
    mask = config.mask
    inv_cov = config.cov_inv_masked

    # Get true samples and datavectors
    true_samples, true_datavectors_full = get_true_samples(data_dir, config_file, sample_temp, cosmo)

    # Apply mask to datavectors
    true_datavectors = true_datavectors_full[:, mask]

    # Get predicted datavectors from the model
    predicted_datavectors = get_pred_samples(model_dir, model_name, true_samples, true_datavectors.shape[1], preset)

    delta_chi2 = calculate_chi2(true_datavectors, predicted_datavectors, inv_cov)

    # Print summary statistics
    print("Chi^2 Summary Statistics:")
    print(f"Mean: {np.mean(delta_chi2):.2e}, Median: {np.median(delta_chi2):.2e}, Min: {np.min(delta_chi2):.2e}, Max: {np.max(delta_chi2):.2e}")

    # Plot chi^2 distribution
    output_dir = model_dir
    plot_chi_squared_distribution(delta_chi2)