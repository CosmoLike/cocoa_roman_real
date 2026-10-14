"""Cobaya likelihood base class for the Roman real-space two-point functions.

The five likelihoods of this folder (cosmic_shear, combo_xi_ggl, combo_xi_gg,
combo_2x2pt, combo_3x2pt) inherit from _cosmolike_prototype_base and differ
only in the probe set they name. The data vector concatenates, for pairs of
tomographic (redshift) bins and for angular bins theta:

  xi_+(theta), xi_-(theta)  cosmic shear: correlations of galaxy shapes
                            between two source bins;
  gamma_t(theta)            galaxy-galaxy lensing: tangential shear of a
                            source bin around the galaxies of a lens bin;
  w(theta)                  angular clustering of the galaxies of a lens bin.

The three together are the "3x2pt" analysis. The likelihood is Gaussian,
ln L = -chi^2/2 with chi^2 = (t - d)^T C^-1 (t - d), where t is the theory
vector, d the data vector and C the covariance, all restricted to the
entries the mask keeps.

How the pieces fit:

- Cobaya, the sampler framework, calls initialize() once, get_requirements()
  to learn what its theory blocks must compute (CAMB or emulator products),
  and logp(**params) at every sampled point.
- cosmolike, a C library, computes the theory vector. Python reaches it
  through the compiled module cosmolike_roman_real_interface (imported as
  ci; source in interface/interface.cpp). cosmolike keeps its state in C
  global variables: the ci.init_* calls of initialize() fix the survey,
  binning, accuracy and model choices for the whole run, and the ci.set_*
  calls of every evaluation hand over the cosmology and the nuisance
  parameters.
- The yaml key use_emulator selects the theory path:
    0 (or False)  CAMB supplies P(k, z), the growth factor and distances;
                  cosmolike computes every two-point function.
    1             neural-network emulators return the two-point functions
                  without shear calibration and point-mass term; cosmolike
                  adds both and applies the mask
                  (internal_get_datavector_emulator).
    2 (hybrid)    emulators replace CAMB for the distances and the matter
                  power spectra; cosmolike computes the projections as for 0.

Units at the boundary with cosmolike: k in h/Mpc (CAMB works in 1/Mpc),
P(k) in (Mpc/h)^3, comoving distances chi in Mpc/h.
"""
# A __future__ import must come before every other statement (only the module
# docstring and comments may precede it). Under Python 3 these three behaviors
# are already the default, so the line changes nothing.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools

# Cobaya's base class for likelihoods configured by a .dataset file and its
# error type (LoggedError, an exception that also writes its message to the
# log); GetDist's IniFile reads the "key = value" lines of the .dataset file.
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2: the nonlinear boost P_nl/P_lin used when non_linear_emul = 1
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time spent inside a `with timer(label):` block.

  @contextmanager turns this generator (a function containing yield) into a
  context manager: the code before `yield` runs when the with-block starts
  and the code after it when the block ends. No code in the project calls
  it; it is a tool for manual timing.

  Arguments:
    label = text printed before the elapsed time

  Side effects: prints "<label>: <seconds>s".
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# The compiled cosmolike module: scripts/start_roman_real.sh puts interface/,
# where the .so lives, on PYTHONPATH.
import cosmolike_roman_real_interface as ci

# OpenMP threads for cosmolike's parallel loops: the user's OMP_NUM_THREADS,
# read once when this module is imported (1 when the variable is unset).
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Return fn wrapped so that cosmolike's OpenMP thread count is reset first.

    cosmolike's hot loops are OpenMP parallel regions sized by
    omp_get_max_threads(), which starts at OMP_NUM_THREADS. Some Python
    libraries call omp_set_num_threads(1) without saying so; the setting is
    process-wide, so afterwards every cosmolike parallel loop would run on
    one core. The wrapper calls ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
    before each call of fn (the binding also keeps OpenBLAS at one thread,
    so BLAS workers do not compete with the OpenMP threads).

    A decorator is a function that receives a function and returns its
    replacement: writing @with_omp_threads above a method makes the method
    name refer to the wrapper. functools.wraps copies fn's name and
    docstring onto the wrapper; *args and **kwargs pass every positional and
    keyword argument through unchanged.

    Arguments:
      fn = the function or method to wrap

    Returns:
      wrapper, a function with fn's arguments and return value
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Reset cosmolike's OpenMP thread count, then call fn unchanged."""
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name in params_source.yaml and
# params_lens.yaml: roman_M1, roman_DZ_S1, roman_B1_1, roman_PM1, ...
survey = "roman"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Gaussian likelihood of the Roman real-space data vector, via cosmolike.

  A name that starts with _ marks the class as internal: input yamls name
  its children (roman_real.cosmic_shear, roman_real.combo_3x2pt, ...), which
  only pass their probe set to initialize.

  Before calling initialize, Cobaya copies every option of the likelihood
  block, merged over the defaults in <likelihood>.yaml, into an attribute of
  the same name (self.accuracyboost, self.use_emulator, ...). Main options:

    data_file, path        the .dataset file and its folder
    accuracyboost          density of the z and k grids and of cosmolike's
                           internal tables
    use_emulator           0/False = CAMB, 1 = data-vector emulators,
                           2 = hybrid (emulated distances and P(k))
    non_linear_emul        1 = EuclidEmulator2 boost below z = 10,
                           2 = the nonlinear P(k) of the theory code
    IA_model               intrinsic alignments: 0 = NLA (nonlinear
                           alignment), 1 = TATT (tidal alignment and tidal
                           torquing)
    IA_code                one-loop spectra from 0 = cosmolike's C FAST-PT,
                           1 = the Python FAST-PT theory block (FAST-PT is
                           an FFT-based perturbation-theory code)
    external_nz_modeling   re-send n(z) at every sample (set_*_related)
    baryon options         external_baryon_suppression, add_baryons_on_dv,
                           use_baryon_pca, create_baryon_pca (initialize)
    print_datavector       write every theory vector to
                           print_datavector_file
  """

  def initialize(self, probe):
    """Read the .dataset file, build the sampling grids and set up cosmolike.

    Steps:
      1. read the .dataset file (named by the yaml key data_file): the data
         vector, covariance, mask and n(z) file names, the number of
         tomographic bins and the theta binning;
      2. build the z and k grids on which P(k, z), growth and distances are
         handed to cosmolike (their sizes grow with accuracyboost);
      3. fix cosmolike's configuration through the ci.init_* calls and load
         the data vector, mask and covariance;
      4. set up the optional baryon modeling.

    Arguments:
      probe = "xi", "xi_ggl", "xi_gg", "2x2pt" or "3x2pt": the two-point
              functions of the data vector (xi = xi_+ and xi_-, ggl =
              gamma_t, gg = w(theta), 2x2pt = gamma_t and w, 3x2pt = all)

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.

    Side effects: sets cosmolike's global configuration; reads the data,
    covariance, mask, n(z) and optional baryon files.
    """
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    # ------------------------------------------------------------------------   
    # z_interp_1D: the redshifts of the distance table chi(z) and of the
    # growth table. Three blocks of 0.80, 0.40 and 0.10 times
    # (1000 + 250 boost) nodes (at least 100, 100 and 50): 0 <= z < 3, where
    # the galaxies are (dz = 0.003 at boost 1), 3 <= z < 50.1, and
    # 1070 <= z <= 1100 around the last-scattering redshift, which only
    # CMB-lensing probes need.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # z_interp_2D: the z nodes of the P(k, z) tables handed to cosmolike.
    # cosmolike interpolates linearly in z between exactly these nodes (it
    # indexes the uniform blocks of the handed grid directly; there is no
    # internal regridding). Linear interpolation leaves a sawtooth-shaped
    # O(dz^2) residual that vanishes at the nodes, so two grids that share
    # no nodes disagree by the full residual amplitude, and a node count
    # that varies freely with the boost moves chi2 without converging
    # (measured with such grids: order-unity chi2 jitter in roman_kl's
    # clustering vector, smaller shifts in this project). The grid therefore
    # refines each uniform block by the integer factor m = 2^ceil(log2(boost)),
    # capped at 16 (m = 1 for boost <= 1), keeping the block endpoints:
    #   (a) every block stays uniform (cosmolike keeps its two-segment
    #       direct indexing, no search);
    #   (b) the nodes of a coarser grid are a subset of those of every finer
    #       grid, so a larger boost is a true refinement: the residual falls
    #       like 1/m^2 and the sawtooth never shifts;
    #   (c) m = 1 gives 105 + 35 = 140 nodes, the redshifts of
    #       z_interp_2D_camb below.
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/(105 m)); the high block multiplies its interval count (endpoint
    # included: 35 nodes = 34 intervals -> 34 m + 1 nodes). The grid stops
    # at z = 49.99, inside the z <= 50 range of the hybrid emulators; above
    # the galaxy redshifts it serves CMB-lensing kernels and the growth
    # normalization at z_interp_2D[-1] (set_cosmo_related).
    #
    # pk_z_refinement (a positive integer) multiplies m on top of the
    # accuracy boost. A Fourier-space data vector reads P(k, z) at fixed
    # multipoles, where the residual of the linear z interpolation does not
    # average out as it does in real space: roman_fourier's 3x2pt chi2 moves
    # by 0.25, 0.030 and 0.002 from m = 1 to 2, 4 and 8; roman_real's and
    # lsst_y1's move by at most 0.004 from m = 1 to 2.
    #
    # getattr(self, name, default) returns the yaml option when the
    # likelihood's yaml defines it and default otherwise.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement is this boost-independent 140-node grid (the m = 1 grid
    # above). The denser nested nodes only re-evaluate the smooth z-spline
    # CAMB builds through these transfer redshifts when the cosmolike
    # tables are filled: raising the boost refines exactly cosmolike's
    # linear z interpolation, and the CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    # log10 of k in 1/Mpc (CAMB's unit), from 10^-4.99 to 100/Mpc, with
    # 1250 + 250 boost nodes (1500 at boost 1); set_cosmo_related converts
    # k to h/Mpc for cosmolike.
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # cosmolike configuration, fixed for the whole run. initial_setup resets
    # cosmolike's C globals to their defaults; then come the probe set, the
    # angular binning (n_theta bins between theta_min and theta_max, in
    # arcmin) and the lens-source pairs left out of galaxy-galaxy lensing
    # (ggl_exclude: [lens, source] pairs counted from 0, flattened to
    # lens0, source0, lens1, source1, ... as cosmolike expects).
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    ci.init_ggl_exclude(np.array(self.ggl_exclude).flatten())

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # n(z) conventions: how the n(z) table is interpolated (0 = cubic
    # spline, 1 = linear, 2+ = Steffen, a monotone spline) and what its z
    # column holds (0 = left bin edges, 1 = sample points)
    # Select the reconstruction before loading either histogram.
    histogram_akima = getattr(self, "photoz_histogram_akima", 0)
    if histogram_akima not in (0, 1):
      raise LoggedError(self.log, "photoz_histogram_akima must be 0 or 1")
    ci.init_photoz_histogram_akima(enabled=int(histogram_akima))

    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    # cosmolike's C FAST-PT: density of its internal convolution grid
    # relative to its output grid
    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # the non-Limber FFTLog chi grid, refined on top of the accuracy boost
    # (narrow lens bins need it: see init_nonlimber_accuracy_boost)
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # Limber approximation for galaxy-galaxy lensing (gs) and galaxy
    # clustering (gg): 0 = exact (non-Limber) projection below l = 150,
    # 1 = Limber at every multipole
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 0)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model galaxy power from an HOD
    # (halo occupation distribution: mean galaxy count in a halo of mass M);
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics use the cold dark matter + baryon spectrum P_cb. The
    # hybrid emulators have no P_cb, so get_neutrino_inputs uses its
    # small-scale limit P_lin/(1 - f_nu)^2.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # use_emulator = 1: the emulators return the two-point functions, and
    # cosmolike computes only the point-mass term. It needs the n(z), the
    # data, mask and covariance (for chi2) and a low accuracy setting.
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      # lmax: the largest multipole of the C_l tables that the Legendre sums
      # turn into real-space correlation functions; is_linear=False selects
      # the nonlinear matter power spectrum.
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling: Python keeps the n(z) tables (self.lens_nz,
      # self.source_nz) and re-sends them at every sample (set_*_related),
      # so cosmolike is told only the number of bins here.
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 
      
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA needs no one-loop IA spectra: use cosmolike's C FAST-PT
        # (IA_code = 0), so get_requirements, which Cobaya calls after
        # initialize, does not request the Python FAST-PT theory block.
        self.IA_code = 0
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # bias_model: one redshift-evolution code per bias parameter, in the
        # order (b1, b2, bs2, b3, bmag, bK); 0 = one amplitude per bin
        ci.init_bias(bias_model=self.bias_model)

      # non_linear_emul = 1: EuclidEmulator2 supplies the nonlinear boost
      # (set_cosmo_related); the emulator object is built once and reused.
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # Baryonic feedback options. Precedence: create_baryon_pca, then
      # external_baryon_suppression, then add_baryons_on_dv; the flags below
      # switch off what a higher option excludes.
      #   create_baryon_pca: compute principal components (PCs) of the
      #     baryonic data-vector changes of the simulations listed in
      #     baryon_pca_select_sims and save them (internal_get_datavector);
      #   external_baryon_suppression: a theory block supplies the
      #     suppression S(k, z) = P_baryons/P_dark-matter-only
      #     (set_cosmo_related);
      #   add_baryons_on_dv: cosmolike multiplies P(k) by the suppression of
      #     the simulation which_bsims_add_on_dv (a contaminated theory);
      #   use_baryon_pca: add sum_i Q_i PC_i to the theory vector, with the
      #     sampled amplitudes roman_BARYON_Q1..Q4 (off under the first two).
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    # The PC file has one row per data-vector entry and one column per
    # component; npcs = 4 matches the four sampled amplitudes
    # roman_BARYON_Q1..Q4 of params_source.yaml.
    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Return the products Cobaya's theory blocks must compute for logp.

    Each key names a quantity a theory block provides (a parameter such as
    H0, a CAMB product such as Pk_interpolator, an emulator output such as
    cosmic_shear); its value holds the options of the request (None = no
    options).

      use_emulator = 1   the data-vector emulators of the probe set, plus H0
                         and chi(z) for the sets that contain gamma_t
      use_emulator = 2   the emulated distances and P(k), mnu for
                         get_neutrino_inputs, the Python FAST-PT tables when
                         IA_code = 1 and the EuclidEmulator2 inputs when
                         non_linear_emul = 1
      otherwise (CAMB)   P(k, z) of the total matter and of cold dark matter
                         + baryons on z_interp_2D_camb, chi(z), omnuh2, the
                         optional FAST-PT and EuclidEmulator2 inputs as
                         above, and S(k, z) when external_baryon_suppression
                         is set

    Pk_interpolator asks for both the nonlinear and the linear spectrum
    ("nonlinear": (True, False)) up to k_max = kmax_boltzmann * accuracyboost
    in 1/Mpc; distances come in Mpc.

    Returns:
      dict mapping requirement names to option dicts or None
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
      # Also need Python FAST-PT if IA_code == 1
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 (non_linear_emul = 1) takes omegab, mnu, w and wa
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # mnu gives Omega_nu h^2 on this path (get_neutrino_inputs)
      _requirements_["mnu"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The baryon-suppression theory block evaluates S(k, z) on the
      # likelihood's own grid: z_interp_2D and k = 10^log10k_interp_2D in
      # 1/Mpc (the theory block converts to h/Mpc if its model needs it).
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # Also need Python FAST-PT if IA_code == 1
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 (non_linear_emul = 1) takes omegab, mnu, w and wa
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Keep both fields available to the likelihood and direct halo readers.
      # CAMB obtains them from the same transfer-function calculation.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Hand the cosmology of the current sample to cosmolike.

    Reads from the theory blocks (CAMB or the hybrid emulators) the linear
    and nonlinear matter power spectra, the growth factor and the comoving
    distance, converts them to cosmolike's units and calls ci.set_cosmology;
    with IA_code = 1 it also installs the Python FAST-PT tables. With
    use_emulator = 1 it sets nothing.

    Hot path, run once per evaluation. Units: k from 1/Mpc to h/Mpc
    (log10 k - log10 h), P from Mpc^3 to (Mpc/h)^3 (ln P + ln h^3), chi
    from Mpc to Mpc/h. Shape flow of the power spectra:

      PKL.logP(z_interp_2D, k)  [n_z2D, n_k]
        -> flatten(order='F')   [n_z2D * n_k], z index fastest, the layout
                                of set_cosmology's lnP_linear

      legend: n_z2D = len(z_interp_2D), n_k = len(log10k_interp_2D)

    Raises:
      an exception when non_linear_emul is neither 1 nor 2.

    Side effects: replaces cosmolike's cosmology tables (and, with
    IA_code = 1, its FAST-PT tables).
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      # delta_tot: total matter, massive neutrinos included. Cobaya's
      # interpolator extrapolates ln P beyond the k range CAMB computed; the
      # limits 1e-6 and 250 boost (1/Mpc) enclose log10k_interp_2D.
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      
      
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EuclidEmulator2 covers z < 10 and 8.73e-3 <= k <= 9.4 h/Mpc (the k
        # list 10^-2.0589 ... 10^0.973 below). ln(boost) is interpolated
        # linearly in log10 k onto the cosmolike k grid (h/Mpc), extrapolated
        # above 9.4 h/Mpc and set to 0 (boost 1) below 8.73e-3 h/Mpc, where
        # P(k) is linear.
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # The nonlinear P(k) of the theory code covers every redshift; at
        # z < 10 it is replaced below by P_lin times the EuclidEmulator2
        # boost.
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # (z_interp_2D < 10)[:, None] has shape [n_z2D, 1]: np.where
        # broadcasts it along k and picks, row by row, the EE2 or the theory
        # spectrum; ravel(order='F') restores the z-fastest flat layout.
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", self.non_linear_emul)

      # G(z) = D(z)(1 + z) = D/a: the linear growth factor D relative to its
      # matter-era behavior D ~ a, measured as sqrt(P(z, k)/P(0, k)) (1 + z)
      # at the single wavenumber k = growth_k.
      #
      # G is sampled on the dense 1D z grid, clipped to the P(k)
      # interpolator range: cosmolike reads G linearly in z, and on the
      # coarse 2D grid (dz ~ 0.03) the linear read misses D by up to 9e-5
      # and the growth rate f = 1 - (1+z) dlnG/dz (the slope of the table)
      # by 1%; on the 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a cubic
      # spline in z through CAMB's transfer redshifts, so this asks CAMB
      # for no extra redshifts (about 0.1 ms per evaluation). The table is
      # divided by G at the last z_2D node (z_growth ends below it);
      # cosmolike's growfac divides by G(0), so D(z=0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is sampled at growth_k (default 0.05/Mpc), a sub-horizon scale.
      # At k = 5e-4/Mpc (about 2 H0/c) CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every
      # reader of G (IA amplitudes, one-loop D^4, sigma(M, z), the growth
      # rate f) describes sub-horizon modes; with 0.06 eV neutrinos the
      # growth varies by 0.03% above 0.05/Mpc (cosmolike_core skill,
      # references/growth_factor_measurements.md)
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)

      # external_baryon_suppression: the theory block returns a dict that
      # maps each requested z to S(k) on the requested k grid (with its own
      # calibration-range masking applied). ln S is added to the nonlinear
      # ln P: lnPNL is z-fastest, so the slice [i :: n_z2D] is the k row of
      # the i-th redshift (enumerate yields the pairs i, z). A z missing
      # from the dict is skipped with a warning; any exception is logged as
      # an error and the evaluation continues with the suppression applied
      # so far (none when the retrieval itself fails).
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: the one-loop IA and galaxy-bias spectra come from the
      # Python FAST-PT theory block instead of cosmolike's C FAST-PT. Each
      # table is an [n_rows, N] array on one k grid; row -2 holds k in
      # h/Mpc, so FPTIA[-2, 0] and FPTIA[-2, -1] are its ends. set_IA_PS and
      # set_bias_PS copy the tables in, flattened row after row (order='C').
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra far above the neutrino free-streaming
        scale, an approximation on cluster scales. Its measured size is
        in projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Hand the source-galaxy nuisance parameters of one sample to cosmolike.

    For each source bin i = 1..source_ntomo: the multiplicative shear bias
    roman_M<i> (shear calibration), the photo-z shift roman_DZ_S<i> and the
    intrinsic-alignment amplitudes roman_A1_<i> (tidal alignment, the NLA
    amplitude), roman_A2_<i> (tidal torquing) and roman_BTA_<i> (density
    weighting of the tidal-alignment term); A2 and BTA enter TATT only. A
    name missing from params counts as 0. With use_emulator = 1 only the
    shear calibration is set: the other source parameters are emulator
    inputs.

    Arguments:
      params = the sample's parameter values by name; **params collects
               every keyword argument Cobaya passes (roman_M1=..., ...)
               into this dictionary

    Side effects: sets cosmolike's source nuisance parameters; with
    external_nz_modeling, re-sends the source n(z).
    """
    ntomo = self.source_ntomo
    # Each list below holds one value per bin: the inner comprehension builds
    # the names roman_M1 ... roman_M<ntomo>, and the outer one reads each
    # value from params (the second argument of get is the value used when
    # the name is absent).
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # The source n(z) is re-sent at every sample so that a user function
        # can modify it with sampled parameters (for example, adding an
        # outlier population). The change acts on a copy, so self.source_nz
        # keeps the fiducial n(z); a user function goes between the copy and
        # set_source_sample:
        #   source_nz_local = f(source_nz_local, nuisance parameters)
       
        source_nz_local = self.source_nz.copy()

        ci.set_source_sample(source_nz_local)

        # The photo-z shifts still apply on top of the re-sent n(z); a user
        # function that models them itself would drop this call.
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Hand the lens-galaxy nuisance parameters of one sample to cosmolike.

    For each lens bin i = 1..lens_ntomo: the point-mass amplitude
    roman_PM<i> (the mass enclosed below the smallest gamma_t scales, which
    gamma_t still feels at larger radii; in 10^13 Msun/h), the galaxy-bias
    parameters roman_B1_<i> (linear; 1 when absent), roman_B2_<i>
    (quadratic), roman_BMAG_<i> (magnification), roman_B3NL_<i>
    (third-order nonlocal) and roman_BK_<i> (nonlocal), and the photo-z
    shift roman_DZ_L<i>. Other missing names count as 0. With
    use_emulator = 1 only the point mass is set.

    Arguments:
      params = the sample's parameter values by name (as in
               set_source_related)

    Side effects: sets cosmolike's lens nuisance parameters; with
    external_nz_modeling, re-sends the lens n(z).
    """
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # The lens n(z) is re-sent at every sample so that a user function
        # can modify it with sampled parameters (for example, adding an
        # outlier population). The change acts on a copy, so self.lens_nz
        # keeps the fiducial n(z); a user function goes between the copy and
        # set_lens_sample:
        #   lens_nz_local = f(lens_nz_local, nuisance parameters)
       
        lens_nz_local = self.lens_nz.copy()

        ci.set_lens_sample(lens_nz_local)

        # The photo-z shifts still apply on top of the re-sent n(z); a user
        # function that models them itself would drop this call.
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def compute_logp(self, datavector):
    """Return ln L = -chi^2/2 of a theory data vector.

    ci.compute_chi2 compares the vector with the data on the unmasked
    entries, using the inverse of the masked covariance that init_data_real
    loaded.

    Arguments:
      datavector = full-length theory vector (masked entries included),
                   float64 numpy array

    Returns:
      float, the log-likelihood up to an additive constant
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def logp(self, **params):
    """Return the log-likelihood of one sample; Cobaya calls this method.

    Arguments:
      params = the sample's parameter values by name

    Returns:
      float, ln L = -chi^2/2
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the full-length theory data vector of one sample.

    Dispatches to internal_get_datavector_emulator (use_emulator = 1) or to
    internal_get_datavector (CAMB and hybrid paths).

    Arguments:
      params = the sample's parameter values by name

    Returns:
      float64 numpy array in the layout [xi_+ and xi_- | gamma_t | w], one
      entry per data point, masked entries set to 0
    """
    if self.use_emulator == 1:
      dv = self.internal_get_datavector_emulator(**params)
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    """Assemble the theory vector from the data-vector emulators.

    The emulator theory blocks return each two-point block without shear
    calibration or point-mass term. This method places them in the 3x2pt
    layout (the blocks of absent probes stay 0); then cosmolike multiplies
    by the (1 + m) shear-calibration factors, adds the point-mass term to
    gamma_t when some roman_PM<i> is nonzero, and zeroes the masked
    entries. With use_baryon_pca it also adds the baryon PCs.

      sizes = [n_xi, n_gammat, n_w]   (n_xi counts xi_+ and xi_-)
      dv    = [xi | gamma_t | w]      length n_xi + n_gammat + n_w

    Arguments:
      params = the sample's parameter values by name

    Returns:
      float64 numpy array, the full-length theory vector

    Raises:
      ValueError when an emulator block has the wrong length or the probe
      name is unknown.

    Side effects: with print_datavector, writes (index, value) rows to
    print_datavector_file.
    """
    # ---------------------------------------------------------------
    # The shear calibration m and the point-mass amplitudes are not emulator
    # inputs: cosmolike applies them below. The point-mass term needs the
    # lens parameters and the cosmology, so set_lens_related and
    # set_cosmo_related run only when the probe set contains gamma_t and some
    # PM is nonzero (all(v == 0 for v in PM) is True when every PM is 0).
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    if not self.use_baryon_pca: 
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    # print_datavector: two columns, the entry index and the value; fmt is
    # the tuple ('%d', '%1.8e'), one format per column.
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the theory vector with cosmolike (CAMB and hybrid paths).

    Hands the cosmology and the nuisance parameters of the sample to
    cosmolike (the lens parameters only when the probe set has galaxies),
    then computes the full-length vector with masked entries set to 0. With
    create_baryon_pca it first computes the baryon principal components and
    saves them to filename_baryon_pca; with use_baryon_pca it adds
    sum_i Q_i PC_i, where Q_i = roman_BARYON_Q<i>.

    Arguments:
      params = the sample's parameter values by name

    Returns:
      the theory vector as a list of floats (get_datavector converts it to
      a numpy array)

    Side effects: writes filename_baryon_pca (create_baryon_pca) and
    print_datavector_file (print_datavector).
    """
    self.set_cosmo_related()
    
    if self.probe != "xi":
      self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else: 
      datavector = ci.compute_data_vector_masked()

    # print_datavector: same two-column file as in
    # internal_get_datavector_emulator
    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
