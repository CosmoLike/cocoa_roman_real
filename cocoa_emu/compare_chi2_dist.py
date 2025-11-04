import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from nn_emulator_v2 import nn_emulator
from cocoa_emu import cocoa_config

def calculate_chi2(true_values, predicted_values, covariance=None):
    """Calculate chi^2 between true and predicted values"""
    residuals = true_values - predicted_values
    if covariance is None:
        # If no covariance matrix provided, assume diagonal with unit variance
        return np.sum(residuals**2, axis=1)
    else:
        # With covariance matrix
        inv_cov = np.linalg.inv(covariance)
        if np.shape(residuals)==(1080,):
            chi2 = residuals @ inv_cov @ residuals
        else:
            chi2 = np.zeros(len(residuals))
            for i in range(len(residuals)):
                chi2[i] = residuals[i] @ inv_cov @ residuals[i]
        return chi2

def get_samples(data_dir, model_dir, model, config_file, dataset_size):
    """Get samples and calculate chi^2 values for a specific dataset size"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = cocoa_config(config_file)
    mask = config.mask
    masked_size = sum(mask)

    # Load data based on dataset size
    if dataset_size == 256:
        samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_8_1_256.npy'))
        true_datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_8_1_256.npy'))
    elif dataset_size == 128:
        samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_8_1_128_w0wa.npy'))
        true_datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_8_1_128_w0wa.npy'))
    elif dataset_size == 64:
        samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_8_1_64_w0wa.npy'))
        true_datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_8_1_64_w0wa.npy'))
    else:
        raise ValueError(f"Dataset size {dataset_size} not supported. Use 64, 128, or 256.")


    if type(model) == int:
        model1_path = os.path.join(model_dir, 'dc1_xi_emulator_w0wa_'+str(model)+'.pt')
        emulator1 = nn_emulator(preset='ind_restrf', output_dim=1080, input_dim=25)
        emulator1.load(model1_path, device)

        model2_path = os.path.join(model_dir, 'dc1_GGL_emulator_w0wa_'+str(model)+'.pt')
        emulator2 = nn_emulator(preset='ind_restrf', output_dim=769, input_dim=25)
        emulator2.load(model2_path, device)

        model3_path = os.path.join(model_dir, 'dc1_W_emulator_w0wa_'+str(model)+'.pt')
        emulator3 = nn_emulator(preset='ind_restrf', output_dim=101, input_dim=25)
        emulator3.load(model3_path, device)

        # Make predictions
        with torch.no_grad():
            predicted_xi = emulator1.predict(torch.tensor(samples, dtype=torch.float32).to(device))
            print(np.shape(predicted_xi))
            predicted_gamma = emulator2.predict(torch.tensor(samples, dtype=torch.float32).to(device))
            predicted_w = emulator3.predict(torch.tensor(samples, dtype=torch.float32).to(device))
            predicted_datavectors = np.concatenate((predicted_xi,predicted_gamma,predicted_w),axis=1)

    else:
        emulator = nn_emulator(preset='3x2_restrf',output_dim=masked_size)
        emulator.load(model_path, device)
        emulator.model.eval()
        # Make predictions
        with torch.no_grad():
            predicted_datavectors = emulator.predict(torch.tensor(samples, dtype=torch.float32).to(device))
    

    # Load parameter configuration
    config = cocoa_config(config_file)
    #covmat = torch.as_tensor(config.cov[0:1080,0:1080], dtype=torch.float64)
    covmat = torch.as_tensor(config.cov[mask][:,mask],dtype=torch.float64)

    # Calculate chi^2 values
    #chi2_values = calculate_chi2(true_datavectors[:,0:1080], predicted_datavectors, covmat)
    chi2_values = calculate_chi2(true_datavectors[:,mask],predicted_datavectors,covmat)

    # Print summary statistics
    print(f"Chi^2 statistics for dataset size {dataset_size}:")
    print(f"Mean: {np.mean(chi2_values):.2e}")
    print(f"Median: {np.median(chi2_values):.2e}")
    print(f"Std: {np.std(chi2_values):.2e}")
    print(f"Min: {np.min(chi2_values):.2e}")
    print(f"Max: {np.max(chi2_values):.2e}")
    print()

    return samples, true_datavectors, predicted_datavectors, chi2_values

def plot_chi2_comparison(chi2_64, chi2_128, chi2_256, output_dir):
    """Plot chi^2 distributions for both datasets on the same plot"""
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Determine bin range that covers both datasets
    #min_chi2 = np.min(chi2_64)
    #max_chi2 = np.min(chi2_64)
    min_chi2 = min([np.min(chi2_128), np.min(chi2_64)])
    max_chi2 = max([np.max(chi2_128), np.max(chi2_64)])
    #min_chi2 = min([np.min(chi2_128), np.min(chi2_64), np.min(chi2_256)])
    #max_chi2 = max([np.max(chi2_128), np.max(chi2_64), np.max(chi2_256)])
    
    # Create log-spaced bins
    bins = np.logspace(np.log10(min_chi2), np.log10(max_chi2), 50)
    
    # Plot histograms with thick black outlines and no individual bar outlines
    plt.hist(chi2_64, bins=bins, density=True, alpha=0.5, 
             label='T=128', color='tab:green', edgecolor='black', linewidth=2, histtype='stepfilled', hatch='x')
    plt.hist(chi2_128, bins=bins, density=True, alpha=0.5, 
             label='T=256', color='tab:blue', edgecolor='black', linewidth=2, histtype='stepfilled')
    #plt.hist(chi2_256, bins=bins, density=True, alpha=0.5, 
    #         label='T=512', color='tab:orange', edgecolor='black', linewidth=2, histtype='stepfilled', hatch='//')
    
    # Customize plot
    plt.xlabel(r'$\Delta \chi^2$', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.xscale('log')
    #plt.xlim(1e-3, 1)
    plt.legend(fontsize=24, frameon=False)
    
    # Make x-axis tick labels larger and remove y-axis elements
    plt.xticks(fontsize=20)
    plt.yticks([])  # Remove y-axis tick labels
    plt.ylabel('')  # Remove y-axis label
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'chi2_distribution_comparison_w0wa.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {os.path.join(output_dir, 'chi2_distribution_comparison_w0wa.png')}")

if __name__ == "__main__":
    # Set paths
    full_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real'
    data_dir = os.path.join(full_dir, 'chains/training_data_gaussian')
    model_dir = os.path.join(full_dir, 'chains/emulator')
    config_file = os.path.join(full_dir, 'DC1_3x2_w0wa_PARAMS.yaml')
    output_dir = os.path.join(full_dir, 'chains/emulator/performance_plots')
    
    # Model paths
    #model_512 = os.path.join(model_dir, 'dc1_3x2_emulator_512.pt')
    #model_256 = os.path.join(model_dir, 'dc1_3x2_emulator_w0wa_256.pt')
    #model_128 = os.path.join(model_dir, 'dc1_3x2_emulator_w0wa_128.pt')

    model_256 = 256
    model_128 = 128
    #model = 64

    #print("Processing DC1 512 emulator with 256 dataset...")
    # Get chi^2 values for 512 emulator on 256 dataset
    #samples_256, true_dv_256, pred_dv_256, chi2_256 = get_samples(data_dir, model_512, config_file, 256)
    chi2_256 = 0
    #chi2_128 = 0
    #chi2_64 = 0

    print("Processing DC1 256 emulator with 128 dataset...")
    # Get chi^2 values for 256 emulator on 128 dataset
    samples_128, true_dv_128, pred_dv_128, chi2_128 = get_samples(data_dir, model_dir, model_256, config_file, 128)
    
    print("Processing DC1 emulator with 64 dataset...")
    # Get chi^2 values for 64 emulator on 64 dataset
    samples_64, true_dv_64, pred_dv_64, chi2_64 = get_samples(data_dir, model_dir, model_128, config_file, 64)
    
    #samples_64, true_dv_64, pred_dv_64, chi2_64 = get_samples(data_dir, model_dir, model, config_file, 64)

    print("Creating comparison plot...")
    # Create comparison plot
    plot_chi2_comparison(chi2_64, chi2_128, chi2_256, output_dir)
    

    print("Done!") 