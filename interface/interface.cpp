#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <map>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <boost/algorithm/string.hpp>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/recompute.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"


#include <carma.h>
#include <armadillo>
#include "cosmolike/generic_interface.hpp"
#include "cosmolike/cosmo2D_wrapper.hpp"

// Why STL vectors?
// The conversion between STL vector and python np array is cleaner
// arma:Col is cast to 2D np array with 1 column (not as nice!)

using vector = arma::Col<double>;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INIT FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_baryons_contamination(std::string sim)
{
  cosmolike_interface::init_baryons_contamination(sim);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_data_3x2pt_real_space(
    std::string cov, 
    std::string mask, 
    std::string data
  )
{
  arma::Col<int>::fixed<3> order = {0, 1, 2};
  cosmolike_interface::init_data_3x2pt_real_space(cov, mask, data, order);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_distances(std::vector<double> io_z, std::vector<double> io_chi)
{
  cosmolike_interface::init_distances(
      arma::conv_to<vector>::from(io_z),
      arma::conv_to<vector>::from(io_chi)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_growth(std::vector<double> io_z, std::vector<double> io_G)
{
  cosmolike_interface::init_growth(
      arma::conv_to<vector>::from(io_z),
      arma::conv_to<vector>::from(io_G)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_linear_power_spectrum(
    std::vector<double> io_log10k,
    std::vector<double> io_z, 
    std::vector<double> io_lnP
  )
{
  cosmolike_interface::init_linear_power_spectrum(
      arma::conv_to<vector>::from(io_log10k),
      arma::conv_to<vector>::from(io_z),
      arma::conv_to<vector>::from(io_lnP)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_non_linear_power_spectrum(
    std::vector<double> io_log10k,
    std::vector<double> io_z, 
    std::vector<double> io_lnP
  )
{
  cosmolike_interface::init_non_linear_power_spectrum(
      arma::conv_to<vector>::from(io_log10k),
      arma::conv_to<vector>::from(io_z),
      arma::conv_to<vector>::from(io_lnP)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SET FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_baryon_pcs(arma::Mat<double> eigenvectors)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_baryon_pcs");

  cosmolike_interface::BaryonScenario::get_instance().set_pcs(eigenvectors);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_baryon_pcs");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_cosmological_parameters(
    const double omega_matter,
    const double hubble, 
    const bool is_cached_cosmology
  )
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmological_parameters");

  if(!is_cached_cosmology)
  {
    // Cosmolike should not need parameters from inflation or dark energy.
    // because Cobaya provides P(k,z), H(z), D(z), Chi(z)...
    // It may require H0 to set scales and \Omega_M to set the halo model

    // cosmolike c interface
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0-omega_matter;
    // Cosmolike only needs to know that there are massive neutrinos (>0)
    cosmology.Omega_nu = 0.1;
    cosmology.h0 = hubble/100.0; // assuming H0 in km/s/Mpc
    cosmology.MGSigma = 0.0;
    cosmology.MGmu = 0.0;

    // Technical Problem: we want Cosmolike to calculate the data vector when
    // Cobaya request (no cache). To avoid cache in Cosmolike, we use a
    // random number generators to set cosmology.random
    cosmology.random = cosmolike_interface::RandomNumber::get_instance().get();
    cosmology.is_cached = 0;
  }
  else
  {
    cosmology.is_cached = 1;
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmological_parameters");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_cosmology(
    const double omega_matter,
    const double hubble,
    std::vector<double> io_log10k_2D,
    std::vector<double> io_z_2D, 
    std::vector<double> io_lnP_linear,
    std::vector<double> io_lnP_nonlinear,
    std::vector<double> io_G,
    std::vector<double> io_z_1D,
    std::vector<double> io_chi
  )
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmology");

  cosmology.Omega_m = omega_matter;
  cosmology.Omega_v = 1.0-omega_matter;

  // Cosmolike only needs to know that there are massive neutrinos (>0)
  cosmology.Omega_nu = 0.1;
  cosmology.h0 = hubble/100.0; // assuming H0 in km/s/Mpc
  cosmology.MGSigma = 0.0;
  cosmology.MGmu = 0.0;

  // Technical Problem: we want Cosmolike to calculate new data vector when
  // we update distances / matter power spectrum / Growth factor.
  // To avoid cache, we use a random number generator to set cosmology.random
  cosmology.random = cosmolike_interface::RandomNumber::get_instance().get();
  cosmology.is_cached = 0;

  cosmolike_interface::init_linear_power_spectrum(
    arma::conv_to<vector>::from(io_log10k_2D),
    arma::conv_to<vector>::from(io_z_2D),
    arma::conv_to<vector>::from(io_lnP_linear)
  );

  cosmolike_interface::init_non_linear_power_spectrum(
    arma::conv_to<vector>::from(io_log10k_2D),
    arma::conv_to<vector>::from(io_z_2D),
    arma::conv_to<vector>::from(io_lnP_nonlinear)
  );

  cosmolike_interface::init_growth(
      arma::conv_to<vector>::from(io_z_2D),
      arma::conv_to<vector>::from(io_G)
    );

  cosmolike_interface::init_distances(
      arma::conv_to<vector>::from(io_z_1D),
      arma::conv_to<vector>::from(io_chi)
    );

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmology");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_IA(
    std::vector<double> A1, 
    std::vector<double> A2,
    std::vector<double> BTA
  )
{
  cosmolike_interface::set_nuisance_IA(
      arma::conv_to<vector>::from(A1),
      arma::conv_to<vector>::from(A2),
      arma::conv_to<vector>::from(BTA)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_shear_calib(std::vector<double> M)
{
  cosmolike_interface::set_nuisance_shear_calib(
      arma::conv_to<vector>::from(M)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_shear_photoz(std::vector<double> SP)
{
  cosmolike_interface::set_nuisance_shear_photoz(
      arma::conv_to<vector>::from(SP)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_clustering_photoz(std::vector<double> CP)
{
  cosmolike_interface::set_nuisance_clustering_photoz(
      arma::conv_to<vector>::from(CP)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_bias(
    std::vector<double> B1, 
    std::vector<double> B2, 
    std::vector<double> BMAG
  )
{
  cosmolike_interface::set_nuisance_bias(
      arma::conv_to<vector>::from(B1),
      arma::conv_to<vector>::from(B2),
      arma::conv_to<vector>::from(BMAG)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_pm(std::vector<double> PM)
{  
  cosmolike_interface::PointMass::get_instance().set_pm_vector(
      arma::conv_to<vector>::from(PM)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// COMPUTE FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double compute_chi2(std::vector<double> datavector)
{
  return cosmolike_interface::IP::get_instance().get_chi2(
      arma::conv_to<vector>::from(datavector)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> compute_baryon_pcas_3x2pt(std::string scenarios)
{
  arma::Col<int>::fixed<3> order = {0, 1, 2};

  // Init BaryonScenario Class --------------------------------------------
  cosmolike_interface::BaryonScenario::get_instance().set_scenarios(scenarios);

  return cosmolike_interface::compute_baryon_pcas_3x2pt_real(order);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

std::vector<double> compute_theory_data_vector_masked()
{
  arma::Col<int>::fixed<3> order = {0, 1, 2};
  return arma::conv_to<std::vector<double>>::from(
      cosmolike_interface::compute_data_vector_3x2pt_real_masked_any_order(order)
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

std::vector<double> compute_theory_data_vector_masked_with_baryon_pcs(
    std::vector<double> Q  // PC amplitudes
  )
{
  arma::Col<int>::fixed<3> order = {0, 1, 2};
  return arma::conv_to<std::vector<double>>::from(
      cosmolike_interface::compute_data_vector_3x2pt_real_masked_any_order(
          Q,
          order
        )
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

std::vector<double> get_binning_real_space()
{  
  return arma::conv_to<std::vector<double>>::from(
      cosmolike_interface::get_binning_real_space()
    );
}

// ---------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER -------------------------------
// ---------------------------------------------------------------------------

PYBIND11_MODULE(cosmolike_roman_interface, m)
{
  m.doc() = "CosmoLike Interface for roman-Y1 3x2pt Module";

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // INIT FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("init_accuracy_boost",
      &cosmolike_interface::init_accuracy_boost,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      py::arg("accuracy_boost").none(false),
      py::arg("sampling_boost").none(false),
      py::arg("integration_accuracy").none(false)
    );

  m.def("init_baryons_contamination",
      &init_baryons_contamination,
      "Init data vector contamination (on the matter power spectrum) with baryons",
      py::arg("sim").none(false)
    );

  m.def("init_binning",
      &cosmolike_interface::init_binning_real_space,
      "Init Bining related variables",
      py::arg("ntheta_bins").none(false).noconvert(),
      py::arg("theta_min_arcmin").none(false),
      py::arg("theta_max_arcmin").none(false)
    );

  m.def("init_cosmo_runmode",
      &cosmolike_interface::init_cosmo_runmode,
      "Init Run Mode (should we force the matter power spectrum to be linear)",
      py::arg("is_linear").none(false)
    );

  m.def("init_data_real",
      &init_data_3x2pt_real_space,
      "Load covariance matrix, mask (as a vector of 0s and 1s) and data vector"
      "hold their values",
      py::arg("COV").none(false),
      py::arg("MASK").none(false),
      py::arg("DATA").none(false)
    );

  m.def("init_distances",
      &init_distances,
      "Load comoving distance, chi(z), as a function  of redshift to Cosmolike",
      py::arg("z").none(false),
      py::arg("chi").none(false)
    );

  m.def("init_growth",
      &init_growth,
      "Load Growth Factor (G(z) = D/a) as a function of redshift to Cosmolike",
      py::arg("z").none(false),
      py::arg("G").none(false)
    );

  m.def("init_IA",
      &cosmolike_interface::init_IA,
      "Init IA related options",
      py::arg("ia_model").none(false).noconvert(),
      py::arg("ia_redshift_evolution").none(false).noconvert()
    );

  m.def("init_lens_sample",
      &cosmolike_interface::init_lens_sample,
      "Init Lens Sample",
      py::arg("filename").none(false),
      py::arg("ntomo_bins").none(false).noconvert()
    );

  m.def("init_linear_power_spectrum",
      &init_linear_power_spectrum,
      "Load Linear Matter Power Spectrum from Cobaya to Cosmolike",
      py::arg("log10k").none(false),
      py::arg("z").none(false),
      py::arg("lnP").none(false)
    );

  m.def("init_non_linear_power_spectrum",
      &init_non_linear_power_spectrum,
      "Load Matter Power Spectrum from Cobaya to Cosmolike",
      py::arg("log10k").none(false),
      py::arg("z").none(false),
      py::arg("lnP").none(false)
    );

  m.def("init_probes",
      &cosmolike_interface::init_probes,
      "Init Probes (cosmic shear or 2x2pt or 3x2pt...)",
      py::arg("possible_probes").none(false)
    );

  m.def("initial_setup",
      &cosmolike_interface::initial_setup,
      "Initialize Cosmolike Variables to their Default Values"
    );

  m.def("init_source_sample",
    &cosmolike_interface::init_source_sample,
      "Init source sample from File",
      py::arg("filename").none(false),
      py::arg("ntomo_bins").none(false).noconvert()
    );

  m.def("init_survey_parameters",
      &cosmolike_interface::init_survey,
      "Init Survey Parameters",
      py::arg("surveyname").none(false),
      py::arg("area").none(false),
      py::arg("sigma_e").none(false)
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // SET FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("set_cosmology",
      &set_cosmology,
      "Set Cosmological Paramters, Distance, Matter Power Spectrum, Growth Factor",
       py::arg("omegam").none(false),
       py::arg("H0").none(false),
       py::arg("log10k_2D").none(false),
       py::arg("z_2D").none(false),
       py::arg("lnP_linear").none(false),
       py::arg("lnP_nonlinear").none(false),
       py::arg("G").none(false),
       py::arg("z_1D").none(false),
       py::arg("chi").none(false)
    );

  m.def("set_baryon_pcs",
      &set_baryon_pcs,
      "Load baryonic principal components from numpy array",
       py::arg("eigenvectors").none(false)
    );

  m.def("set_nuisance_ia",
      &set_nuisance_IA,
      "Set nuisance Intrinsic Aligment (IA) amplitudes",
      py::arg("A1").none(false),
      py::arg("A2").none(false),
      py::arg("B_TA").none(false)
    );

  m.def("set_nuisance_bias",
      &set_nuisance_bias,
      "Set nuisance Bias Parameters",
      py::arg("B1").none(false),
      py::arg("B2").none(false),
      py::arg("B_MAG").none(false)
    );

  m.def("set_nuisance_shear_calib",
      &set_nuisance_shear_calib,
      "Set nuisance shear calibration amplitudes",
      py::arg("M").none(false)
    );

  m.def("set_nuisance_clustering_photoz",
      &set_nuisance_clustering_photoz,
      "Set nuisance clustering shear photo-z bias amplitudes",
      py::arg("bias")
    );

  m.def("set_nuisance_shear_photoz",
      &set_nuisance_shear_photoz,
      "Set nuisance shear photo-z bias amplitudes",
      py::arg("bias").none(false)
    );

  m.def("set_cosmological_parameters",
      &set_cosmological_parameters,
      "Set cosmological parameters",
      py::arg("omega_matter").none(false),
      py::arg("hubble").none(false),
      py::arg("is_cached").none(false)
    );

  m.def("set_point_mass",
      &set_pm,
       "Set the point mass amplitudes",
      py::arg("PMV").none(false)
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // reset FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  m.def("reset_bary_struct",
      &reset_bary_struct,
       "Set the Baryon Functions to not contaminate the MPS w/ Baryon effects"
    );
  
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("compute_data_vector_masked",
      &compute_theory_data_vector_masked,
      "Compute theoretical data vector. Masked dimensions are filled w/ zeros",
      py::return_value_policy::move
    );

  m.def("compute_data_vector_masked_with_baryon_pcs",
      &compute_theory_data_vector_masked_with_baryon_pcs,
      "Compute theoretical data vector, including contributions from baryonic"
      " principal components. Masked dimensions are filled w/ zeros",
      py::arg("Q").none(false),
      py::return_value_policy::move
    );

  m.def("compute_chi2",
      &compute_chi2,
      "Compute $\\chi^2$ given a theory data vector input",
      py::arg("datavector").none(false),
      py::return_value_policy::move
    );

  m.def("compute_baryon_pcas",
      &compute_baryon_pcas_3x2pt,
      "Compute baryonic principal components given a list of scenarios" 
      "that contaminate the matter power spectrum",
      py::arg("scenarios").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // Theoretical Cosmolike Functions
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("get_binning_real_space",
      &get_binning_real_space,
      "Get real space binning (theta bins)"
    );

  m.def("xi_pm_tomo",
      &cosmolike_interface::xi_pm_tomo_cpp,
      "Compute cosmic shear (real space) data vector at all tomographic"
      " and theta bins"
    );

  m.def("w_gammat_tomo",
      &cosmolike_interface::w_gammat_tomo_cpp,
      "Compute galaxy-galaxy lensing (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("w_gg_tomo",
      &cosmolike_interface::w_gg_tomo_cpp,
      "Compute galaxy-galaxy clustering (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("C_ss_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("C_ss_tomo_limber",
      py::overload_cast<vector>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("int_for_C_ss_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::int_for_C_ss_tomo_limber_cpp
      ),
      "Compute integrand for shear-shear (fourier - limber) data vector"
      " at a single tomographic bin and ell value",
      py::arg("a").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("ni").none(false).noconvert()
    );

  m.def("int_for_C_ss_tomo_limber",
      py::overload_cast<vector, vector>(
        &cosmolike_interface::int_for_C_ss_tomo_limber_cpp
      ),
      "Compute integrand shear-shear (fourier - limber) data vector at all" 
      " tomographic bins and many scale factor and ell (vectorized)",
      py::arg("a").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("C_gs_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_gs_tomo_limber_cpp
      ),
      "Compute shear-position (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("nl").none(false).noconvert(),
      py::arg("ns").none(false).noconvert()
    );

  m.def("C_gs_tomo_limber",
      py::overload_cast<vector>(
        &cosmolike_interface::C_gs_tomo_limber_cpp
      ),
      "Compute shear-position (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("int_for_C_gs_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::int_for_C_gs_tomo_limber_cpp
      ),
      "Compute integrand for shear-position (fourier - limber) data vector"
      " at a single tomographic bin and ell value",
      py::arg("a").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("nl").none(false).noconvert(),
      py::arg("ns").none(false).noconvert()
    );

  m.def("int_for_C_gs_tomo_limber",
      py::overload_cast<vector, vector>(
        &cosmolike_interface::int_for_C_gs_tomo_limber_cpp
      ),
      "Compute integrand shear-shear (fourier - limber) data vector at all" 
      " tomographic bins and many scale factor and ell (vectorized)",
      py::arg("a").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("C_gg_tomo_limber",
      py::overload_cast<const double, const int>(
        &cosmolike_interface::C_gg_tomo_limber_cpp
      ),
      "Compute position-position (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("nz").none(false).noconvert()
    );

  m.def("C_gg_tomo_limber",
      py::overload_cast<vector>(
        &cosmolike_interface::C_gg_tomo_limber_cpp
      ),
      "Compute position-position (fourier - limber) data vector at all"
      " tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("int_for_C_gg_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::int_for_C_gg_tomo_limber_cpp
      ),
      "Compute integrand for position-position (fourier - limber) data vector"
      " at a single tomographic bin and ell value",
      py::arg("a").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("int_for_C_gg_tomo_limber",
      py::overload_cast<vector, vector>(
        &cosmolike_interface::int_for_C_gg_tomo_limber_cpp
      ),
      "Compute integrand position-position (fourier - limber) data vector" 
      " at all tomographic bins and many scale factor and ell (vectorized)",
      py::arg("a").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("C_gg_tomo",
      py::overload_cast<vector>(
        &cosmolike_interface::C_gg_tomo_cpp
      ),
      "Compute position-position (fourier - non-limber/limber) data vector"
      " at all tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int main()
{
  std::cout << "GOODBYE" << std::endl;
  exit(1);
}
