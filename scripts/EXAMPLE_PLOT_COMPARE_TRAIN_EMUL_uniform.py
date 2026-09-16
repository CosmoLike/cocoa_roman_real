import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

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

parameter = [u'As_1e9', u'ns', u'H0', u'omegam', u'omegab', u'w0pwa', u'w',
             u'roman_A1_1', u'roman_A1_2', u'roman_DZ_S1']
chaindir  = os.environ['ROOTDIR'] + "/projects/roman_real/chains/"

analysissettings={'smooth_scale_1D':0.25,
                  'smooth_scale_2D':0.25,
                  'ignore_rows': u'0.0',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 256}

root_chains = (
  'w0wa_takahashi_params_train_cs_32',
  'w0wa_takahashi_params_train_cs_unifs'
)
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.saveAsText(chaindir + '/.VM_P1_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + root_chains[1],settings=analysissettings)
p = samples.getParams()
samples.saveAsText(chaindir + '/.VM_P1_TMP2')
# --------------------------------------------------------------------------------
#samples=loadMCSamples(chaindir + root_chains[2],settings=analysissettings)
#p = samples.getParams()
#samples.saveAsText(chaindir + '/.VM_P1_TMP3')
# --------------------------------------------------------------------------------

#GET DIST PLOT SETUP
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

g.triangle_plot(
  params=parameter,
  roots=[chaindir + '/.VM_P1_TMP1',
         chaindir + '/.VM_P1_TMP2'],
         #chaindir + '/.VM_P1_TMP3'],
  plot_3d_with_param=None,
  line_args=[ {'lw': 1.0,'ls': 'solid', 'color': 'cornflowerblue'},
              {'lw': 2.1,'ls': '--', 'color': 'maroon'},
              {'lw': 1.2,'ls': 'dotted', 'color': 'black'},
              {'lw': 1.6,'ls': '-.', 'color': 'indigo'}
            ],
  contour_colors=['cornflowerblue','maroon','black','indigo'],
  contour_ls=['solid', '--', 'solid','dotted','-.'], 
  contour_lws=[1.0,2.1,1.0,1.2,1.6],
  filled=[True,False,True,False,True],
  shaded=False,
  legend_labels=[
    'cosmic shear emul equivalent training - T=32',
    'cosmic shear emul equivalent training - Unif',
    'cosmic shear emul equivalent training - T=64',
  ],
  legend_loc=(0.32, 0.875))

# ----------------------------------------------------
# ----------------------------------------------------
axarr = g.subplots
# ----------------------------------------------------
axarr[2,0].set_xlim([1.3,2.8])
# ----------------------------------------------------
# ----------------------------------------------------

g.export(os.path.join(chaindir,"example_compare_train_emul.pdf"))