import numpy as np
import torch
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from nn_emulator import nn_emulator
import os
from cocoa_emu import cocoa_config
import yaml
import getdist.plots as gplot
from getdist import loadMCSamples

def load_config(config_file):
    """Load parameter configuration from YAML file"""
    config = cocoa_config(config_file)
    return config

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
            #chi2 = np.sum(chi2)
        return chi2

#Get range of values for each parameter from chains
from getdist import MCSamples
from getdist import loadMCSamples
# Analysis settings
analysis_settings = {
    'smooth_scale_1D': 0.35,
    'smooth_scale_2D': 0.35,
    'ignore_rows': u'0.5',
    'range_confidence': u'0.01'
}

def load_and_process_chains(chain_path, settings):
    """Load and process MCMC chains with derived parameters"""
    samples = loadMCSamples(chain_path, settings=settings)
    p = samples.getParams()
    
    # Add derived parameter SS8
    #samples.addDerived(p.s8omegamp5/0.5477225575, name='SS8', label='{S_8}')
    
    return samples

def calculate_parameter_constraints(samples, param_name):
    """Calculate best-fit and confidence intervals for a parameter"""
    stats = samples.getMargeStats()
    param_stats = stats.parWithName(param_name)
    
    best_fit = param_stats.mean
    sigma1_lower = param_stats.limits[0].lower
    sigma1_upper = param_stats.limits[0].upper
    sigma5_lower = param_stats.limits[2].lower
    sigma5_upper = param_stats.limits[2].upper
    
    return {
        'best_fit': best_fit,
        '1sigma': (sigma1_lower, sigma1_upper),
        '5sigma': (sigma5_lower, sigma5_upper)
    }

def chi_squared_distribution(chi2_values, bins=100):
    """Plot the chi^2 distribution"""
    plt.hist(chi2_values, bins=np.logspace(np.log10(chi2_values.min()),np.log10(chi2_values.max())), density=True, alpha=0.5)
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('Density')
    plt.xlim(1e-4,1)
    plt.xscale('log')
    plt.savefig(os.path.join(output_dir, 'chi2_distribution_3x2_lcdm.png'), dpi=300, bbox_inches='tight')
    #plt.savefig(os.path.join(output_dir, 'chi2_distribution_3x2_w0wa.png'), dpi=300, bbox_inches='tight')
    plt.close()

def make_heatmap(x, y, z, bins, median=True):
    x_new = np.linspace(np.amin(x), np.amax(x), bins+1)
    y_new = np.linspace(np.amin(y), np.amax(y), bins+1)
    z_new = np.zeros((bins,bins))
    for i in range(bins):
        for j in range(bins):
            if median == True: z_new[j,i] = np.median(z[(x >= x_new[i]) & (x < x_new[i+1]) & (y >= y_new[j]) & (y < y_new[j+1])])
            else: z_new[j,i] = np.mean(z[(x >= x_new[i]) & (x < x_new[i+1]) & (y >= y_new[j]) & (y < y_new[j+1])])
    
    return x_new, y_new, z_new

def make_diagonal(x, y, bins, median=True):
    x_new = np.linspace(np.amin(x), np.amax(x), bins+1)
    y_new = np.zeros(bins)
    for i in range(bins):
        if median == True: y_new[i] = np.median(y[(x >= x_new[i]) & (x <= x_new[i+1])])
        else: y_new[i] = np.mean(y[(x >= x_new[i]) & (x <= x_new[i+1])])
    return x_new[:bins], y_new

def get_samples(data_dir,model_path,config_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #emulator = nn_emulator(preset='xi_restrf')
    
    # Load parameter configuration
    config = cocoa_config(config_file)
    mask = config.mask
    masked_size = sum(mask)
    #emulator = nn_emulator(preset='3x2_restrf',output_dim=masked_size)
    #emulator.load(model_path, device)
    #emulator.model.eval()
    sample_temp=256
    #model1_path = os.path.join(model_path, 'dc1_3x2_512_emulator_lcdm_9_28.pt')
    #emulator1 = nn_emulator(preset='ind_restrf_v2', output_dim=1950, input_dim=15)
    #emulator1.load(model1_path, config_file, device,norm=True)


    #model1_path = os.path.join(model_path, 'dc1_xi_emulator_w0wa_lhc.pt')
    model1_path = os.path.join(model_path, 'dc1_xi_512_emulator_lcdm_testing.pt')
    #model1_path = os.path.join(model_path, 'dc1_xi_256_emulator_w0wa.pt')
    #model1_path = os.path.join(model_path, 'dc1_xi_256_emulator_lcdm.pt')
    #model1_path = os.path.join(model_path, 'dc1_xi_512_emulator_lcdm_resnet.pt')
    #emulator1 = nn_emulator(preset='ind_restrf', output_dim=1950, input_dim=15)
    emulator1 = nn_emulator(preset='xi_restrf', output_dim=1080, input_dim=15)
    #emulator1 = nn_emulator(preset='ind_restrf_v2', output_dim=1080, input_dim=25)
    #emulator1 = nn_emulator(preset='resnet', output_dim=1950, input_dim=15)
    #model1 = torch.load(model1_path, device)
    #emulator1 = nn_emulator(model=model1)
    emulator1.load(model1_path, config_file, device)
    #print(emulator1)
    #emulator1.model.eval()

    #model2_path = os.path.join(model_path, 'dc1_GGL_512_emulator_lcdm_resnet.pt')
    model2_path = os.path.join(model_path, 'dc1_GGL_512_emulator_lcdm_testing.pt')
    #model2_path = os.path.join(model_path, 'dc1_GGL_emulator_w0wa_lhc.pt')
    emulator2 = nn_emulator(preset='xi_restrf', output_dim=769, input_dim=15)
    #emulator2 = nn_emulator(preset='resnet_v2', output_dim=769, input_dim=15)
    #emulator2.load(model2_path, device)
    #model2 = torch.load(model2_path, device)
    #emulator2 = nn_emulator(model=model2)
    emulator2.load(model2_path, config_file, device)
    #emulator2.model.eval()

    
    model3_path = os.path.join(model_path, 'dc1_W_512_emulator_lcdm_testing.pt')
    #model3_path = os.path.join(model_path, 'dc1_W_512_emulator_lcdm_resnet.pt')
    #model3_path = os.path.join(model_path, 'dc1_W_emulator_w0wa_lhc.pt')
    emulator3 = nn_emulator(preset='xi_restrf', output_dim=101, input_dim=15)
    #emulator3 = nn_emulator(preset='w_restrf', output_dim=101, input_dim=15)
    #emulator3.load(model3_path, device)
    #model3 = torch.load(model3_path, device)
    #emulator3 = nn_emulator(model=model3)
    emulator3.load(model3_path, config_file, device)
    #emulator3.model.eval()
    
    
    #samples = np.load(os.path.join(data_dir, 'partial_params_v2.npy'))[0:10000]
    #true_datavectors = np.load(os.path.join(data_dir, 'partial_datavectors_v2.npy'))[0:10000]
    #samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_9_12_128_lcdm.npy'))
    #true_datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_9_12_128_lcdm.npy'))
    samples = np.load(os.path.join(data_dir, 'dc1_3x2_samples_256_lcdm.npy'))
    true_datavectors = np.load(os.path.join(data_dir, 'dc1_3x2_datavectors_256_lcdm.npy'))
    print(np.shape(samples))
    print(np.shape(true_datavectors))

     # Load parameter configuration
    #config = cocoa_config(config_file)
    #mask = config.mask
    #covmat = torch.as_tensor(config.cov[0:1080,0:1080],dtype=torch.float64)
    covmat = torch.as_tensor(config.cov[mask][:,mask],dtype=torch.float64)

    # Make predictions
    with torch.no_grad():
        #predicted_xi = emulator1.predict(torch.tensor(samples[:,:-8], dtype=torch.float32).to(device))
        predicted_xi = emulator1.predict(torch.tensor(samples, dtype=torch.float32).to(device))
        #print(np.shape(predicted_xi))
        predicted_gamma = emulator2.predict(torch.tensor(samples, dtype=torch.float32).to(device))
        predicted_w = emulator3.predict(torch.tensor(samples, dtype=torch.float32).to(device))
        predicted_datavectors = np.concatenate((predicted_xi,predicted_gamma,predicted_w),axis=1)
        #predicted_datavectors = emulator1.predict(torch.tensor(samples, dtype=torch.float32).to(device))

    # Calculate chi^2 values
    #chi2_values = calculate_chi2(true_datavectors[:,0:1080], predicted_xi,covmat)
    #chi2_values = calculate_chi2(true_datavectors[:,mask][:,1080:1849],predicted_datavectors[:,1080:1849],covmat[1080:1849,1080:1849])
    #chi2_values = calculate_chi2(true_datavectors[:,mask][:,1849:1950],predicted_datavectors[:,1849:1950],covmat[1849:1950,1849:1950])
    chi2_values = calculate_chi2(true_datavectors[:,mask],predicted_datavectors,covmat)
    # Print summary statistics
    print(f"Chi^2 statistics:")
    print(f"Mean: {np.mean(chi2_values):.2e}")
    print(f"Median: {np.median(chi2_values):.2e}")
    print(f"Std: {np.std(chi2_values):.2e}")
    print(f"Min: {np.min(chi2_values):.2e}")
    print(f"Max: {np.max(chi2_values):.2e}")

    return samples, true_datavectors, predicted_datavectors, chi2_values

def plot_heatmap(samples, chi2_values, config_file, output_dir, label=r"$\Delta \chi^2$", extents=[1e0, 1e2], 
cmap='viridis', log_scale=True, median=True, save_str="", chain_data=None):
    # Load parameter configuration
    config = cocoa_config(config_file)
    pnames = list(config.running_params)
    print(pnames)
    print(samples.shape)
    bounds = [[np.min(samples[:,i]), np.max(samples[:,i])] for i in range(len(pnames))]

    #names = ["As_1e9", "ns", "H0", "omegab", "omegam", "w0pwa", "w"]
    names = ["As_1e9", "ns", "H0", "omegab", "omegam"]
    
    labels = {
        'As_1e9': r'$A_s\times 10^9$',
        'ns': r'$n_s$',
        'H0': r'$H_0$',
        'omegab': r'$\Omega_b$',
        'omegam': r'$\Omega_m$',
        'w': r'$w_0$',
        'w0pwa': r'$w_0+w_a$'
    }

    params = samples.copy()
    fig, axs = plt.subplots(len(names),len(names), figsize=(12,12), sharex="col")
    for i in range(len(names)):
        for j in range(len(names)):
            idx_i = pnames.index(names[i])
            idx_j = pnames.index(names[j])
            if i < j:
                axs[i][j].axis("off")
                continue
            if i == j:
                x, y = make_diagonal(params[:,idx_j], chi2_values, 25, median)
                axs[i][j].plot(x, y, color='black', lw=3)
                # Show y-axis ticklabels on the right side for all diagonal plots
                axs[i][j].yaxis.tick_right()  # Move ticks to right side
                #axs[i][j].set_ylim(2e-3,5e-2)
                axs[i][j].set_ylim(1e-2,1)
                axs[i][j].set_yscale('linear')
                # Only add y-axis label for the top-left diagonal plot
                if i == 0 and j == 0:
                    axs[i][j].yaxis.set_label_position('left')  # Keep label on left for top-left plot
                    axs[i][j].set_ylabel(label, fontsize=12)
            else:
                X, Y, Z = make_heatmap(params[:,idx_j], params[:,idx_i], chi2_values, 25, median)
                if log_scale == True: img = axs[i,j].imshow(Z, aspect="auto", extent=(X[0], X[-1], Y[0], Y[-1]), cmap=cmap, 
                                                      norm=colors.LogNorm(vmin=extents[0], vmax=extents[1]))
                else: img = axs[i,j].imshow(Z, aspect="auto", extent=(X[0], X[-1], Y[0], Y[-1]), cmap=cmap, vmin=extents[0], vmax=extents[1])
                axs[i,j].set_xlim(X[0] - (X[-1] - X[0]) * 0.05, X[-1] + (X[-1] - X[0]) * 0.05)
                axs[i,j].set_ylim(Y[0] - (Y[-1] - Y[0]) * 0.05, Y[-1] + (Y[-1] - Y[0]) * 0.05)

                axs[i,j].xaxis.set_ticks_position('both')
                axs[i,j].yaxis.set_ticks_position('both')
                
                # Add chain contours if chain_data is provided
                if chain_data is not None:
                    try:
                        # Get the parameter indices for this plot
                        param_i = names[i]
                        param_j = names[j]
                        
                        # Get the 2D marginalized statistics
                        stats = chain_data.getMargeStats()
                        param1_stats = stats.parWithName(param_j)
                        param2_stats = stats.parWithName(param_i)
                        
                        # Get the 2D contours
                        contours_2d = chain_data.get2DContour(param_j, param_i)
                        
                        if contours_2d is not None:
                            # Plot the contours manually
                            for level in contours_2d:
                                if len(level) > 0:
                                    axs[i,j].plot(level[:, 0], level[:, 1], 'k--', linewidth=3.0, alpha=0.8)
                                    
                    except Exception as e:
                        print(f"Could not plot contours for {param_i} vs {param_j}: {e}")
            
            axs[i,j].tick_params(direction="in")
            for item in ([axs[i,j].xaxis.label, axs[i,j].yaxis.label]):
                item.set_fontsize(15)

                    
            if i == len(names) - 1: axs[i][j].set_xlabel(labels[names[j]])
            if j == 0 and i != 0: axs[i][j].set_ylabel(labels[names[i]])

            if j!=0 and i!=j: axs[i,j].yaxis.set_ticklabels([])
            #if i!=len(names) - 1: axs[i,j].xaxis.set_ticklabels([])

    cbar_ax = fig.add_axes([0.99, 0.14, 0.039, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(label, size=22)
    cbar.ax.tick_params(labelsize=18) 
    plt.subplots_adjust(wspace=0, hspace=0, right=0.95)

    if save_str!="": plt.savefig(save_str, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_performance(samples, chi2_values, config_file, output_dir):
    config = cocoa_config(config_file)
    param_names = list(config.running_params)

    # Create individual parameter chi^2 plots
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 4*len(param_names)))
    if len(param_names) == 1:
        axes = [axes]

    for i, (param, ax) in enumerate(zip(param_names, axes)):
        # Use scatter with density-based coloring for better visualization with log-scale
        # Calculate point density using 2D histogram
        x_bins = np.linspace(samples[:, i].min(), samples[:, i].max(), 50)
        y_bins = np.logspace(np.log10(chi2_values.min()), np.log10(chi2_values.max()), 50)
        
        H, xedges, yedges = np.histogram2d(samples[:, i], chi2_values, bins=[x_bins, y_bins])
        
        # Find the bin index for each point
        x_indices = np.digitize(samples[:, i], x_bins) - 1
        y_indices = np.digitize(chi2_values, y_bins) - 1
        
        # Get density for each point
        densities = []
        for x_idx, y_idx in zip(x_indices, y_indices):
            if 0 <= x_idx < H.shape[0] and 0 <= y_idx < H.shape[1]:
                densities.append(H[x_idx, y_idx])
            else:
                densities.append(0)
        
        densities = np.array(densities)
        
        # Normalize densities for coloring
        if np.max(densities) > 0:
            normalized_densities = densities / np.max(densities)
        else:
            normalized_densities = np.zeros_like(densities)
        
        # Scatter plot with color based on density
        scatter = ax.scatter(samples[:, i], chi2_values, c=normalized_densities, 
                           cmap='viridis', alpha=0.6, s=10, edgecolors='none')
        
        ax.set_xlabel(param)
        ax.set_ylabel('χ²')
        ax.set_yscale('log')
        
        # Calculate and plot median chi^2 in bins
        bins = np.linspace(samples[:, i].min(), samples[:, i].max(), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        digitized = np.digitize(samples[:, i], bins)
        median_chi2 = [np.median(chi2_values[digitized == j]) for j in range(1, len(bins))]
        ax.plot(bin_centers, median_chi2, 'r-', linewidth=2, label='Median χ²')
        ax.legend()
        
        # Add colorbar for the last plot
        #if i == len(param_names) - 1:
        #    cbar = plt.colorbar(scatter, ax=ax)
        #    cbar.set_label('Relative Point Density', size=12)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'chi2_vs_params_dc1_3x2_lcdm.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_indiviual_datavectors(samples, true_datavectors, predicted_datavectors, config_file, output_dir,datavector_index):
    """Plot individual datavectors"""

    # Load parameter configuration
    config = cocoa_config(config_file)
    #covmat = config.cov[0:1080,0:1080]
    mask = config.mask
    covmat = config.cov[mask][:,mask]
    dv_fid = config.dv_fid[mask]
    errors = np.sqrt(np.diagonal(covmat))

    true_datavector = true_datavectors[datavector_index]
    predicted_datavector = predicted_datavectors[datavector_index]
    #chi2_value = calculate_chi2(true_datavector[0:1080], predicted_datavector[0:1080],covmat)
    chi2_value = calculate_chi2(true_datavector, predicted_datavector, covmat)
    # Plot the datavector

    theta_min = np.log10(2.5)
    theta_max = np.log10(250.)
    ntheta = 15
    thetas = np.logspace(theta_min,theta_max,ntheta)

    import matplotlib.gridspec as gridspec

    # Parameters
    n_bins = 8  # Number of bins

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n_bins, n_bins, figsize=(15, 15), sharex=True, sharey=True)

    start = 0
    # Plot each bin combination
    for i in range(n_bins):
        for j in range(n_bins):
            row = 2 * (j - i) 
            if j>=i:
                ax = axes[7-i,j-i]

                delta_predicted = (predicted_datavector[start:start+15]-dv_fid[start:start+15])/dv_fid[start:start+15]
                delta_target = (true_datavector[start:start+15]-dv_fid[start:start+15])/dv_fid[start:start+15]

                ax.plot(thetas, delta_predicted, color='blue', lw=3)
                ax.plot(thetas, delta_target, color='red', lw=3, ls='dashed')
                ax.axhline(1, color='gray', linestyle='solid', lw=1)
                ax.errorbar(thetas, delta_target, xerr=None,
                            yerr=errors[start:start+15]/dv_fid[start:start+15],
                            color='gray', ls='dashed', marker = 'None', markersize=2,capsize=2)
            
                ax.set_xscale("log")
                ax.set(xlim=(2,275),ylim=(-0.3,0.3))
            
                # Add label in the top-left corner
                ax.text(0.1, 0.85, f"{i+1},{j+1}", transform=ax.transAxes,
                        fontsize=14, fontweight='bold', color='black')

            # Adjust ticks to point inwards
            
            #ax.tick_params(axis='both', direction='in', which='both', length=2)
            #if (j-i) == 0:
            #    ax.set_ylabel(r'$\frac{\Delta\xi_+ }{\xi_+^{fid}}$', fontsize=10)
            #    ax.tick_params(labelleft=True)
            #if i == 0:
            #    ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=10)
        
                start = start+15

            else:
                # Hide unused subplots
                ax = axes[7-i,j-i]
                ax.axis('off')

    fig.text(0.5, 0.06, r'$\theta$ [arcmin]', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Cosmic Shear \n' + r'Fractional Difference from Fiducial ($\frac{\Delta\xi_+ }{\xi_+^{fid}}$)', ha='center', va='center', rotation='vertical', fontsize=20)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #axs[0].plot(true_datavector[0:15],marker='o',linestyle='none',label='True')
    #axs[0].plot(predicted_datavector[0:15],marker='none',linestyle='solid',label='Predicted')
    #axs[0].set_title(f'Chi^2: {chi2_value:.2e}')
    #axs[0].legend()
    #axs[0].set_xlabel('Theta')
    #axs[1].plot((true_datavector[0:15]-predicted_datavector[0:15])/true_datavector[0:15])
    plt.savefig(os.path.join(output_dir, f'datavector_{datavector_index}_dc1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(chi2_value)
        
        


if __name__ == "__main__":
    # Set paths
    full_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real_emu'
    data_dir = os.path.join(full_dir, 'chains/training_data')
    model_dir = os.path.join(full_dir, 'chains/emulator')
    #config_file = os.path.join(full_dir, 'LHC0_PARAMS.yaml')
    #config_file = os.path.join(full_dir, 'DC1_3x2_PARAMS.yaml')
    config_file = os.path.join(full_dir, 'DC1_3x2_LCDM_PARAMS.yaml')
    #config_file = os.path.join(full_dir, 'DC1_3x2_w0wa_PARAMS.yaml')
    output_dir = os.path.join(full_dir, 'chains/emulator/performance_plots')
    
    #model_path = os.path.join(model_dir, 'LHC0_emulator12.pt')
    #model_path = os.path.join(model_dir, 'dc1_3x2_emulator_w0wa_128.pt')
    
    # Create performance plots
    samples, true_datavectors, predicted_datavectors, chi2_values = get_samples(data_dir, model_dir, config_file)
    #plot_performance(samples, chi2_values, config_file, output_dir) 
    #chain_dir = full_dir
    #cs_chain = load_and_process_chains(os.path.join(chain_dir, 'chains/lhc0_emu_cs2'), analysis_settings)
    #parameters = [u'As_1e9', u'ns', u'H0', u'omegab', u'omegam', u'w', u'w0pwa', u'roman_A1_1', u'roman_A1_2']

    plot_heatmap(samples, chi2_values, config_file, output_dir,extents=[5e-2, 5],
     save_str=os.path.join(output_dir, 'chi2_3x2_heatmap_lcdm.png'))
    #plot_indiviual_datavectors(samples, true_datavectors, predicted_datavectors, config_file, output_dir,0)
    #plot_indiviual_datavectors(samples, true_datavectors, predicted_datavectors, config_file, output_dir,1)
    chi_squared_distribution(chi2_values,bins=100)