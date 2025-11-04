import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# GENERAL PLOT OPTIONS
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '2.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'png'

#colors4 = ['k','#21908C','#FDE725','#3B528B']
colors6 = ['k','#440154','#3B528B','#21908C','#FDE725','#F97306']

params_cosmo = ['omegam', 'SS8']

# Parameters to analyze
params_cosmo_lcdm = ['As_1e9', 'ns', 'H0', 'omegab', 'omegam']
params_cosmo_w0wa = ['As_1e9', 'ns', 'H0', 'omegab', 'omegam', 'w', 'wa']
params_cosmo_lcdm_s8 = [u'omegam', u'sigma8', u'ns', u'SS8', u'omegab', u'H0']
params_cosmo_w0wa_s8 = [u'omegam', u'sigma8', u'ns', u'omegab', u'H0', u'w', u'wa']

IA_params = ['ROMAN_A1_1', 'ROMAN_A1_2']

basic_nuissance_cs = ['ROMAN_A1_1', 'ROMAN_A1_2', 'ROMAN_DZ_S1', 'ROMAN_DZ_S8',
 'ROMAN_M1', 'ROMAN_M8']
basic_nuissance_params = ['ROMAN_A1_1', 'ROMAN_A1_2', 'ROMAN_DZ_S1', 'ROMAN_DZ_S8',
 'ROMAN_M1', 'ROMAN_M8', 'ROMAN_B1_1', 'ROMAN_B1_8']

params_shear_calib = ['ROMAN_M1', 'ROMAN_M2', 'ROMAN_M3', 'ROMAN_M4', 
'ROMAN_M5', 'ROMAN_M6', 'ROMAN_M7', 'ROMAN_M8']

params_src_pz = ['ROMAN_DZ_S1', 'ROMAN_DZ_S2', 'ROMAN_DZ_S3', 'ROMAN_DZ_S4',
'ROMAN_DZ_S5', 'ROMAN_DZ_S6', 'ROMAN_DZ_S7', 'ROMAN_DZ_S8']

params_gbias_ml = ["ROMAN_B1_1", "ROMAN_B1_2", "ROMAN_B1_3", "ROMAN_B1_4",
"ROMAN_B1_5", "ROMAN_B1_6", "ROMAN_B1_7", "ROMAN_B1_8"]

# True values for parameters
true_values_dc1 = {
    'As_1e9': 2.5,
    'omegam': 0.25,
    'sigma8': 0.77532,
    'ns': 0.97,
    'SS8': 0.70776,
    'omegab': 0.05,
    'H0': 71.0,
    'w': -1.0,
    'wa': 0.0,
    'ROMAN_A1_1': 0.5,
    'ROMAN_A1_2': 0.0,
    'ROMAN_DZ_S1': 0.001414,
    'ROMAN_DZ_S8': -0.001002,
    'ROMAN_B1_1': 1.18,
    'ROMAN_B1_8': 3.44,
    'ROMAN_M1': 0.00203,
    'ROMAN_M8': 0.00278,
    
}

# True values for other cosmology (and for Fourier space)
true_values_dc1_v2 = {
    'As_1e9': 2.1,
    'omegam': 0.3156,
    'sigma8': 0.81107594,
    'ns': 0.9645,
    'SS8': 0.831897,
    'omegab': 0.0492,
    'H0': 67.27,
    'w': -1.0,
    'wa': 0.0,
    'ROMAN_A1_1': 0.606102,
    'ROMAN_A1_2': -1.51541,
    'ROMAN_DZ_S1': 0.001414,
    'ROMAN_DZ_S8': -0.001002,
    'ROMAN_B1_1': 1.18,
    'ROMAN_B1_8': 3.44,
    'ROMAN_M1': 0.00203,
    'ROMAN_M8': 0.00278,
    
}

latex_labels = { 
    'As_1e9': 'A_{s} \\times 10^9',
    'sigma8': '\sigma_8',
    'SS8': 'S_8',
    'ns': 'n_s',
    'H0': 'H_0',
    'omegab': '\Omega_b',
    'omegam': '\Omega_m',
    'w': 'w_0',
    'wa': 'w_a',
    'w0pwa': 'w_0+w_a',
    'ROMAN_A1_1': 'A_\mathrm{TA}', 
    'ROMAN_A1_2': '\eta_\mathrm{TA}',
    'ROMAN_DZ_S1': '\Delta z_{src}^{(1)}', 
    'ROMAN_DZ_S2': '\Delta z_{src}^{(2)}',
    'ROMAN_DZ_S3': '\Delta z_{src}^{(3)}', 
    'ROMAN_DZ_S4': '\Delta z_{src}^{(4)}', 
    'ROMAN_DZ_S5': '\Delta z_{src}^{(5)}', 
    'ROMAN_DZ_S6': '\Delta z_{src}^{(6)}',
    'ROMAN_DZ_S7': '\Delta z_{src}^{(7)}', 
    'ROMAN_DZ_S8': '\Delta z_{src}^{(8)}', 
    'ROMAN_DZ_L1': '\Delta z_{lens}^{(1)}', 
    'ROMAN_DZ_L2': '\Delta z_{lens}^{(2)}', 
    'ROMAN_DZ_L3': '\Delta z_{lens}^{(3)}', 
    'ROMAN_DZ_L4': '\Delta z_{lens}^{(4)}', 
    'ROMAN_DZ_L5': '\Delta z_{lens}^{(5)}', 
    'ROMAN_DZ_L6': '\Delta z_{lens}^{(6)}', 
    'ROMAN_DZ_L7': '\Delta z_{lens}^{(7)}', 
    'ROMAN_DZ_L8': '\Delta z_{lens}^{(8)}', 
    'ROMAN_B1_1': 'b_1^{(1)}', 
    'ROMAN_B1_2': 'b_1^{(2)}', 
    'ROMAN_B1_3': 'b_1^{(3)}', 
    'ROMAN_B1_4': 'b_1^{(4)}', 
    'ROMAN_B1_5': 'b_1^{(5)}', 
    'ROMAN_B1_6': 'b_1^{(6)}', 
    'ROMAN_B1_7': 'b_1^{(7)}', 
    'ROMAN_B1_8': 'b_1^{(8)}', 
    'ROMAN_M1': 'm^{(1)}', 
    'ROMAN_M2': 'm^{(2)}', 
    'ROMAN_M3': 'm^{(3)}', 
    'ROMAN_M4': 'm^{(4)}', 
    'ROMAN_M5': 'm^{(5)}', 
    'ROMAN_M6': 'm^{(6)}', 
    'ROMAN_M7': 'm^{(7)}', 
    'ROMAN_M8': 'm^{(8)}', 
    }

# Parameter ranges for plotting, may need to be adjusted based on the chains being plotted
pars_range = {
    #"omegam": [0.21, 0.32],
    "omegam": [0.235,0.265], 
    "As_1e9": [2.2, 2.8], 
    #"ns": [0.87, 1.07],
    "ns": [0.94,1.00], 
    "H0": [63, 76], 
    "sigma8": [0.8,0.94],
    "SS8": [0.815,0.835],
    "omegab": [0.04, 0.055], 
    "ROMAN_A1_1": [-5, 5], 
    "ROMAN_A1_2": [-5, 5], 
    "ROMAN_B1_1": [0.8, 3.0],
    "ROMAN_B1_2": [0.8, 3.0],
    "ROMAN_B1_3": [0.8, 3.0],
    "ROMAN_B1_4": [0.8, 3.0],
    "ROMAN_B1_5": [0.8, 3.0],
    "ROMAN_B1_6": [0.8, 3.0],
    "ROMAN_B1_7": [0.8, 3.0],
    "ROMAN_B1_8": [0.8, 4.0],
}

# Analysis settings - This is what I've been using as default
analysis_settings = {
    'smooth_scale_1D': 0.35,
    'smooth_scale_2D': 0.35,
    'ignore_rows': u'0.5',
    'range_confidence': u'0.01'
}

def load_and_process_chains(chain_path, settings, parameters):
    """Load and process MCMC chains with derived parameters"""
    samples = loadMCSamples(chain_path, settings=settings)
    p = samples.getParams()
    
    # Add derived parameter SS8
    if hasattr(p, "omegam") and hasattr(p, "sigma8"):
        samples.addDerived(p.sigma8*(p.omegam/0.3)**0.5, name='SS8', label='S_8')

    if hasattr(p, "w0pwa") and not hasattr(p, "wa"):
        samples.addDerived(p.w0pwa-p.w, name='wa', label='w_a')

    for pname in parameters:
        samples.paramNames.parWithName(pname).label = latex_labels[pname]
    
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

def create_triangle_plot(samples_list, params, labels, true_values, print_bf=False, print_bf_chains=None, output_filename=None):
    """Create a triangle plot with optional normalization"""
    g = gplot.getSubplotPlotter(width_inch=12.5)
    g.settings.axis_tick_x_rotation = 65
    g.settings.lw_contour = 1.2
    g.settings.legend_rect_border = False
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 20
    g.settings.legend_fontsize = 40
    g.settings.alpha_filled_add = 0.85
    g.settings.lab_fontsize = 30
    
    g.triangle_plot(
        samples_list,
        params,
        line_args=[
            {'lw': 6.0, 'ls': 'solid', 'color': colors6[0]},
            {'lw': 6.0,'ls': 'dashed', 'color': colors6[3]},
            {'lw': 6.0, 'ls': 'dashed', 'color': colors6[5]},
            {'lw': 6.0,'ls': 'dashed', 'color': colors6[2]},
            #{'lw': 1.6,'ls': 'dotted', 'color':'black'},
            #{'lw': 1.0,'ls': 'dashdot', 'color':'purple'}
        ],
        contour_colors=[colors6[0],colors6[3],colors6[5],colors6[2]],
        filled=[False,True,True,True],
        contour_ls=['solid', 'dashed', 'dashed', 'dashed'],
        contour_lws=[4.0, 4.0, 2.0, 2.0],
        legend_labels=labels,
        legend_loc=(0.5, 0.65),
        #markers = true_values,
        #title_limit = 1,
        #param_limits = pars_range,
    )
    
    # Now adjust subplot spacing, after the plot is created
    g.fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.12, wspace=0.12)
    
    
    # Make true value lines more visible and grey
    for i, name_i in enumerate(params):
        for j, name_j in enumerate(params):
            ax = g.subplots[i, j]
            if ax is not None:
                # Make true value lines more visible
                if name_j in true_values:
                    true_val = true_values[name_j]
                    ax.axvline(true_val, color='grey', linestyle='-', linewidth=3, alpha=0.8, zorder=10)
                if name_i in true_values and i != j:
                    true_val = true_values[name_i]
                    ax.axhline(true_val, color='grey', linestyle='-', linewidth=3, alpha=0.8, zorder=10)

    # Manually align all x labels at the same y position
    for j, name_j in enumerate(params):
        ax = g.subplots[len(params)-1, j]
        if ax is not None:
            ax.xaxis.set_label_coords(0.5, -0.6)  # adjust as needed for evenness
    # Manually align all y labels at the same x position
    for i, name_i in enumerate(params):
        ax = g.subplots[i, 0]
        if ax is not None:
            ax.yaxis.set_label_coords(-0.35, 0.5)  # adjust as needed for evenness

    # Add best fit values with 1σ constraints above diagonal plots (not inside, no box, no overlap)
    if print_bf:
        n_params = len(params)
        for i, name_i in enumerate(params):
            if name_i in params:
                ax = g.subplots[i, i]
                if ax is not None:
                    # Calculate constraints for all chains desired (indexed by print_bf_chains)
                    constraints_list = []
                    for chain in print_bf_chains:
                        samples = samples_list[chain]
                        try:
                            constraints = calculate_parameter_constraints(samples, name_i)
                            constraints_list.append(constraints)
                        except:
                            constraints_list.append(None)
                    # Get axis position in figure coordinates
                    bbox = ax.get_position()
                    x_center = (bbox.x0 + bbox.x1) / 2
                    y_top = bbox.y1
                    # Place the true value above the constraints text
                    if name_i in true_values:
                        true_val = true_values[name_i]
                        true_val_str = f"True: {true_val:.3f}"
                        g.fig.text(x_center, y_top + 0.04, true_val_str, ha='center', va='bottom', fontsize=16, color='grey')
                    # Prepare text lines
                    text_lines = []
                    for idx, constraints in enumerate(constraints_list):
                        if constraints is not None:
                            best_fit = constraints['best_fit']
                            sigma1_lower = np.abs(constraints['1sigma'][0]-constraints['best_fit'])
                            sigma1_upper = np.abs(constraints['1sigma'][1]-constraints['best_fit'])
                            color = ['royalblue','black','lightcoral','gray'][print_bf_chains[idx]]
                            chain_name = labels[print_bf_chains[idx]]

                            best_fit_str = f"{best_fit:.3f}"
                            sigma_lower_str = f"{sigma1_lower:.3f}"
                            sigma_upper_str = f"{sigma1_upper:.3f}"
                            #text = f"{chain_name}: {best_fit_str}$^{{+{sigma_upper_str}}}_{{-{sigma_lower_str}}}$"
                            text = f"Best Fit: {best_fit_str}$^{{+{sigma_upper_str}}}_{{-{sigma_lower_str}}}$"
                            text_lines.append((text, color))
                    # Place the texts above the plot, offsetting to avoid overlap, using the original placement
                    for k, (text, color) in enumerate(text_lines):
                        y_text = y_top + 0.0 + 0.025 * (len(text_lines)-k-1)
                        g.fig.text(x_center, y_text, text, ha='center', va='bottom', fontsize=16, color=color)
    
    if output_filename:
        # Save with extra padding to avoid cutting off text
        g.fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.7)


def plot_alts(chain_dir,chain_names,chain_labels,analysis_settings,params,output_filename):
    """Plot alternate dark energy models from chains in chain_dir"""
    chains = []
    for chain_name in chain_names:
        chain_path = os.path.join(chain_dir, chain_name)
        samples = load_and_process_chains(chain_path, analysis_settings, params)
        chains.append(samples)
    
    create_triangle_plot(chains, params, chain_labels, true_values_dc1, output_filename=output_filename,print_bf=False)

def print_marge_stats(samples, param_name):
    """Print marginalized statistics for a parameter"""
    stats = samples.getMargeStats()
    param_stats = stats.parWithName(param_name)
    
    print(f"Parameter: {param_name}")
    print(f"Mean: {param_stats.mean}")
    print(f"1σ limits: {param_stats.limits[0].lower} to {param_stats.limits[0].upper}")
    print(f"5σ limits: {param_stats.limits[2].lower} to {param_stats.limits[2].upper}")

def print_all_marge_stats(samples, param_names):
    """Print marginalized statistics for multiple parameters"""
    for param_name in param_names:
        print_marge_stats(samples, param_name)
        print("")  # Blank line between parameters

def main():
    # Set chain directory
    chain_dir = '/groups/behroozi/hbowden/cocoa/Cocoa/projects/roman_real/'
    chain_dir2 = os.path.dirname(os.path.abspath(__file__))
    
    # Define chains to use

    ## w0wa chains
    #cs_chain = 'chains/dc1_cs_w0wa'
    #cs_chain_planck = 'chains/dc1_cs_w0wa_planck'
    #threex2_chain = 'chains/dc1_3x2_w0wa'
    #threex2_chain_planck = 'chains/dc1_3x2_w0wa_planck'

    #LCDM chains
    #threex2_chain = 'chains/dc1_3x2'
    #threex2_chain_mask2 = 'chains/dc1_mask2_mcmc'

    
    v1_chain = load_and_process_chains(os.path.join(chain_dir, 'chains/dc1_3x2'), analysis_settings, params_cosmo_lcdm)
    v1_chain_emu = load_and_process_chains(os.path.join(chain_dir2, 'chains/dc1_emu_3x2_mcmc_v4'), analysis_settings,
     params_cosmo_lcdm)



    create_triangle_plot([v1_chain,v1_chain_emu], params_cosmo_lcdm, ['Baseline','Emulator'], true_values_dc1, print_bf=False, print_bf_chains=[0],
    output_filename='dc1_emu_test.png')


if __name__ == "__main__":
    main() 