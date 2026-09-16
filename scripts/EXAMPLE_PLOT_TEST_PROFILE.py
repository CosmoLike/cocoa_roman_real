import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

samples = []
scriptdir  = os.environ['ROOTDIR'] + "/projects/roman_real/scripts/"

# ------------------------------------------------------------------------------
data = np.loadtxt(scriptdir+"roman_real_w0wa_b_taka_valid_parameters.txt") 
#data = np.loadtxt(scriptdir+"roman_real_w0wa_b_taka_train_parameters.txt") 
names = ['As_1e9', 'ns', 'H0', 'omegab', 'omegam', 'w0pwa', 'w',
         'roman_DZ_S1', 'roman_DZ_S2', 
         'roman_DZ_S3', 'roman_DZ_S4', 'roman_DZ_S5', 
         'roman_DZ_S6', 'roman_DZ_S7', 'roman_DZ_S8', 
         'roman_A1_1', 'roman_A1_2'
]
labels = [
    r"10^{9} A_s", r"n_s", r"H_0", r"\Omega_b", r"\Omega_m", r"w_0+w_a", r"w_0",
    r"\Delta z_\mathrm{s,roman}^1", r"\Delta z_\mathrm{s,roman}^2",
    r"\Delta z_\mathrm{s,roman}^3", r"\Delta z_\mathrm{s,roman}^4", r"\Delta z_\mathrm{s,roman}^5",
    r"\Delta z_\mathrm{s,roman}^6", r"\Delta z_\mathrm{s,roman}^7", r"\Delta z_\mathrm{s,roman}^8",
    r"A_\mathrm{1IA,roman}^1", r"A_\mathrm{1IA,roman}^2"
]
flat_priors = {
  0: (0.5, 5),   
  1: (0.87, 1.07),
  2: (55.0, 91.0),
  3: (0.03, 0.07),
  4: (0.1, 0.9),
  5: (-5 , -0.01),
  6: (-3,  -0.01),
  15: (-5.0, 5.0),
  16: (-5.0, 5.0)
}
ranges = {names[j]: [lo, hi] for j, (lo, hi) in flat_priors.items()}
x = MCSamples(samples=data[:,0:17], names=names, ranges=ranges, labels=labels)
#x.thin(factor = int(len(x.weights)/10000))
samples.append(x)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
params   = [u'As_1e9', u'ns', u'H0', u'omegab', u'omegam', u'w0pwa', u'w']
def darken(color, amount=0.20):
    r, g, b = mcolors.to_rgb(color)
    return (r*(1-amount), g*(1-amount), b*(1-amount))
params_color = [
  darken('cornflowerblue', 0.20),
  darken('maroon',         0.15),
  darken('lightcoral',     0.25),
  darken('black',          0.00),
  darken('indigo',         0.15),
  darken('teal',           0.20),  
  darken('slategray',      0.20)
]
chaindir = os.environ['ROOTDIR'] + "/projects/roman_real/chains/"

analysissettings={'smooth_scale_1D':0.15,
                  'smooth_scale_2D':0.15,
                  'ignore_rows': u'0.0',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 256}

g=gplot.getSubplotPlotter(chain_dir=chaindir,
                          analysis_settings=analysissettings,
                          width_inch=10.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour=1.0
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 15.0
g.settings.legend_fontsize = 20.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

g.triangle_plot(samples,
  params=params,
  plot_3d_with_param=None,
  line_args=[ {'lw': 1.0,'ls': 'solid', 'color': 'cornflowerblue'},
            ],
  contour_colors=['cornflowerblue'],
  contour_ls=['solid'], 
  contour_lws=[1.0],
  filled=[True],
  shaded=False,
  legend_labels=[
    'w0wa Halofit Validation Points (emulator)',
  ],
  legend_loc=(0.32, 0.875))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
axarr = np.asarray(g.subplots)
def pname(p):
  return getattr(p, "name", str(p))  # ParamInfo.name is what you want
n = axarr.shape[0]

for i in range(7):
  root  = os.environ['ROOTDIR'] + "/projects/roman_real/chains/EXAMPLE_EMUL_PROFILE2."
  idx   = {name: i+2 for i, name in enumerate(params)}
  data  = np.loadtxt(root + params[i] + '.txt', comments="#",)
  for r in range(n):
    for c in range(n):
      ax = axarr[r, c]
      if ax is None:
          continue
      pars = getattr(ax, "getdist_params", None)
      if not pars:
          continue
      if len(pars) == 2:
          for j in range(15):
            ax.plot(data[j, idx[pname(pars[0])]], data[j, idx[pname(pars[1])]], 
                    marker='o',
                    markersize=4,
                    color=params_color[i])
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
fig = g.fig
shown = names
x0, y0 = 0.7, 0.85 
dy = 0.03            
for i in range(7):
    y = y0 - i*dy
    # colored square
    fig.text(x0, y, "■", color=params_color[i],
             transform=fig.transFigure,
             ha="center", 
             va="top", 
             fontsize=15)
    fig.text(x0 + 0.05, y, f"{params[i]}",
             transform=fig.transFigure,
             color="black", 
             ha="center", 
             va="top", 
             fontsize=15)
# ----------------------------------------------------
# ----------------------------------------------------
g.export(os.path.join(chaindir,"example_test_profile.pdf"))