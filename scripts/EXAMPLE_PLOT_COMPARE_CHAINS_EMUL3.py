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

parameter = [u'As_1e9', u'omegam', u'ns', u'H0', u'omegab', u'chi2v2']
chaindir  = os.environ['ROOTDIR'] + "/projects/roman_real/chains/"

analysissettings={'smooth_scale_1D':0.25,
                  'smooth_scale_2D':0.25,
                  'ignore_rows': u'0.3',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 256}

analysissettings2={'smooth_scale_1D':0.25,
                   'smooth_scale_2D':0.25,
                   'ignore_rows': u'0.0',
                   'range_confidence' : u'0.005',
                   'fine_bins_2D': 1024,
                   'fine_bins_1D': 256}

root_chains = (
  'EXAMPLE_EMUL_MCMC3',
  'EXAMPLE_EMUL_NAUTILUS1',
  'EXAMPLE_EMUL_EMCEE1',
  'EXAMPLE_EMUL_POLY1',
)

# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P3_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[1], settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P3_TMP2')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[2],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P3_TMP3')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[3],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P3_TMP4')
# --------------------------------------------------------------------------------

#GET DIST PLOT SETUP
g=gplot.getSubplotPlotter(chain_dir=chaindir,
                          analysis_settings=analysissettings2,
                          width_inch=10.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour=1.0
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 15.0
g.settings.legend_fontsize = 15.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

g.triangle_plot(
  params=parameter,
  roots=[chaindir + '/.VM_P3_TMP1',
         chaindir + '/.VM_P3_TMP2',
         chaindir + '/.VM_P3_TMP3',
         chaindir + '/.VM_P3_TMP4'],
  plot_3d_with_param=None,
  line_args=[{'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
              {'lw': 1.2,'ls': '--', 'color':'black'},
              {'lw': 2.1,'ls': 'dotted', 'color': 'maroon'},
              {'lw': 1.6,'ls': '-.', 'color': 'indigo'}
            ],
  contour_colors=['lightcoral','black','maroon', 'indigo'],
  contour_ls=['solid','--','dotted','-.'],
  contour_lws=[1.0,1.2,2.1,1.6],
  filled=[True,False,False,True],
  shaded=False,
  legend_labels=[
    'MH, 4-walkers, burn-in=0.3 (HMCODE)',
    'Nautilus, $n_{\\rm live}=1024$, $n_{\\rm eff} \\sim 15,000$',
    'EMCEE $n_{\\rm walkers}=3 \\times ndim$, $n_{\\rm eval} \\sim 2,000,000$',
    'PolyChord $n_{\\rm live}=512$, $n_{\\rm repeat}=3D$',
  ],
  legend_loc=(0.3, 0.85))
g.export(os.path.join(chaindir,"example_compare_chains_emul3.pdf"))