"""Unit test 5: the notebook-style direct cosmolike interface.

The EXAMPLE_EVALUATE1.ipynb notebook drives cosmolike WITHOUT cobaya:
it runs CAMB itself, hands the resulting power spectra and distances
to the compiled interface through set_cosmology, and calls
compute_data_vector_masked directly (the functions interface.cpp binds
from the cosmo2D wrapper layer). Tests 1-4 evaluate through the cobaya
likelihood, so they never touch that call path, and an interface
change can break every notebook while the yaml pipeline keeps passing.
Two failures of exactly that kind motivated this test: init_IA grew a
required third argument (ia_code), and the grid-monotonicity check
added to the C layer aborts the whole process - in Jupyter, a dead
kernel - when it receives a chi(z) grid with duplicated nodes. The
test here:

  5. rebuilds the notebook's call sequence against the frozen example1
     dataset at the frozen fiducial point and checks the chi2 stays
     within CHI2_TOLERANCE (0.2) of the frozen cobaya reference.

The two pipelines are close but deliberately not identical: the
notebook computes its own CAMB run and interpolation grids while the
cobaya prototype uses its internal ones, so the chi2 values differ at
the 0.01 level. That sits far inside the 0.2 tolerance and far from
the failure modes this test exists to catch (a TypeError from a
changed binding, a process abort, or an order-unity chi2 shift).

Like every physics evaluation in this suite, the computation runs in
a worker subprocess: initializing this configuration next to another
of different dimensions would abort the process (see cocoa_test_utils
for the full explanation), and an abort inside a worker leaves a
readable error instead of killing pytest.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_real/tests/test_notebook_interface.py
"""

import json
import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# __file__ is this file's own path; insert(0, ...) puts its folder
# FIRST in the search order, ahead of any same-named module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"

# =============================================================================
# WORKER LAYER (mirrors cocoa_test_utils Section 5, for this file's
# computation; the shared _worker only knows the cobaya evaluations)
# =============================================================================
# __file__ is this file's own path; the worker driver imports the
# file by that absolute path (see _WORKER_DRIVER)
_THIS_FILE = os.path.abspath(__file__)

# A flag distinct from the shared harness's, so a notebook-interface
# worker is never mistaken for a cobaya-evaluation worker.
_WORKER_FLAG = "COCOA_TESTS_NOTEBOOK_WORKER"

# The driver handed to `python -c` inside the worker: load THIS file
# by path and call its _worker with the result path.
_WORKER_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'test_notebook_interface_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(sys.argv[2])\n"
)


def _get_camb_cosmology(point, camb_args, kmax_boltzmann):
    """Run CAMB the way the notebook does and package the cosmology.

    This reproduces the notebook's get_camb_cosmology at its default
    accuracy (boost 1, so the dyadic factor m is 1): the interpolation
    grids, the CAMB configuration, and the unit conventions (cosmolike
    wants k in h/Mpc, P(k) in (Mpc/h)^3, distances in Mpc/h) all
    follow the notebook cell, with the numerical settings taken from
    the frozen configuration instead of the notebook's hardcoded
    values so the comparison targets exactly what the yaml ran.

    Arguments:
      point     = the frozen {parameter name: value} dictionary; the
                  cosmology (As_1e9, ns, H0, omegab, omegam, mnu, w,
                  w0pwa) is read from it.
      camb_args = the frozen theory camb extra_args dictionary
                  (halofit_version, AccuracyBoost, k_per_logint,
                  kmax, lens_potential_accuracy, ...).
      kmax_boltzmann = the frozen likelihood kmax_boltzmann; paired
                  with camb_args["kmax"], the one physical cutoff.

    Returns:
      (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth,
       z_interp_1D, chi), the exact tuple set_cosmology consumes.
    """
    import camb
    from camb import model
    import numpy as np

    h = point["H0"] / 100.0
    # neutrino correction as in the notebook: omegach2 subtracts the
    # massive-neutrino contribution from the cold component. ** is
    # python's power operator (h**2 = h squared); ^ would be XOR
    omegabh2 = point["omegab"] * h**2
    omegach2 = (point["omegam"] - point["omegab"]) * h**2 \
        - (point["mnu"] * (3.046 / 3)**0.75) / 94.0708
    # .get(key, fallback) returns the fallback when the frozen camb
    # block carries no kmax of its own
    kmax = float(camb_args.get("kmax", kmax_boltzmann))

    # chi(z) support grid: interior segments drop their endpoint
    # (cosmolike aborts on non-monotone grids), same construction as
    # the notebook and likelihood/_cosmolike_prototype_base.py.
    # np.linspace(a, b, n) is n evenly spaced values from a to b;
    # endpoint=False leaves the upper edge out, so the next segment
    # can start there without duplicating a node
    tmp = int(1000 + 250 * 1.0)
    z_interp_1D = np.concatenate(
        (np.linspace(0.0, 3.0, max(100, int(0.80 * tmp)), endpoint=False),
         np.linspace(3.0, 50.1, max(100, int(0.40 * tmp)), endpoint=False),
         np.linspace(1070, 1100, max(50, int(0.10 * tmp)))), axis=0)

    # 2D table grids at boost 1: the m = 1 member of the nested dyadic
    # family (105 + 35 = 140 z nodes), and the 1500-node log10k grid
    z_interp_2D = np.concatenate(
        (np.linspace(0, 3.0, 105, endpoint=False),
         np.linspace(3.0, 49.99, 35)), axis=0)
    log10k_interp_2D = np.linspace(-4.99, 2.0, int(1250 + 250 * 1.0))

    # every .get(name, default) below falls back to the notebook's
    # default when the frozen extra_args leave that key out
    pars = camb.set_params(
        H0=point["H0"],
        ombh2=omegabh2,
        omch2=omegach2,
        mnu=point["mnu"],
        omk=0,
        tau=0.06,
        As=1e-9 * point["As_1e9"],
        ns=point["ns"],
        halofit_version=camb_args.get("halofit_version", "takahashi"),
        lmax=10,
        AccuracyBoost=float(camb_args.get("AccuracyBoost", 1.0)),
        lens_potential_accuracy=float(
            camb_args.get("lens_potential_accuracy", 1.0)),
        num_massive_neutrinos=1,
        nnu=3.046,
        accurate_massive_neutrino_transfers=bool(
            camb_args.get("accurate_massive_neutrino_transfers", False)),
        k_per_logint=int(camb_args.get("k_per_logint", 10)),
        kmax=kmax)
    pars.set_dark_energy(w=point["w"], wa=point["w0pwa"] - point["w"],
                         dark_energy_model="ppf")
    pars.NonLinear = model.NonLinear_both
    # the z list handed to CAMB stays at the 140-node grid (CAMB caps
    # requested transfer redshifts at 256; at boost 1 the table grid
    # and the request grid coincide anyway)
    pars.set_matter_power(redshifts=z_interp_2D, kmax=kmax, silent=True)

    results = camb.get_results(pars)
    PKL = results.get_matter_power_interpolator(
        var1="delta_tot", var2="delta_tot", nonlinear=False,
        extrap_kmax=2.5e2, hubble_units=False, k_hunit=False)
    PKNL = results.get_matter_power_interpolator(
        var1="delta_tot", var2="delta_tot", nonlinear=True,
        extrap_kmax=2.5e2, hubble_units=False, k_hunit=False)

    # flatten(order='F') serializes the (z, k) tables the way the
    # interface expects; the log(h^3) converts P(k) to (Mpc/h)^3
    lnPL = np.log(PKL.P(z_interp_2D, np.power(10.0, log10k_interp_2D))
                  .flatten(order="F")) + np.log(h**3)
    # the frozen configuration runs non_linear_emul = 2: the nonlinear
    # spectrum is CAMB's halofit, no emulator boost on top
    lnPNL = np.log(PKNL.P(z_interp_2D, np.power(10.0, log10k_interp_2D))
                   .flatten(order="F")) + np.log(h**3)
    # log10k shifts to h/Mpc units after the tables were evaluated
    log10k_interp_2D = log10k_interp_2D - np.log10(h)

    # growth from the linear P(k) at a fixed large scale, normalized
    # to its highest-z entry, exactly as the notebook builds it
    # (np.sqrt, the 1+z factor, and the division by a single entry
    # all act elementwise on the whole array)
    G_growth = np.sqrt(PKL.P(z_interp_2D, 0.0005)
                       / PKL.P(0, 0.0005)) * (1 + z_interp_2D)
    G_growth = G_growth / G_growth[len(G_growth) - 1]

    chi = results.comoving_radial_distance(z_interp_1D) * h

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth,
            z_interp_1D, chi)


def _notebook_chi2_impl():
    """Worker-side body: the notebook call sequence, start to finish.

    Mirrors the notebook's setup cell followed by its get_chi2: dataset
    descriptor via getdist's IniFile, the init_* sequence on the
    compiled interface, one CAMB run, set_cosmology plus the nuisance
    setters, then the masked data vector and its chi2. Every numeric
    setting (lmax, IA choices, ggl_exclude, CAMB accuracy) comes from
    the frozen example1 configuration, and the dataset comes from the
    frozen data copy, so the number is comparable to the frozen cobaya
    reference and the live project can change freely.

    Returns:
      chi2 = -2 ln L against the frozen shipped data vector, float.
    """
    from getdist import IniFile
    import numpy as np

    import cosmolike_roman_real_interface as ci

    # load_frozen_info resolves the likelihood path to the absolute
    # frozen/data location and returns the complete configuration
    info = u.load_frozen_info(EXAMPLE, tatt=False)
    like = info["likelihood"]["roman_real.cosmic_shear"]
    camb_args = info["theory"]["camb"]["extra_args"]
    point = u.load_frozen_point(EXAMPLE)

    dataset = os.path.join(like["path"], like["data_file"])
    print(f"    dataset: {dataset}", flush=True)
    # normpath collapses ./ and ../ segments in the path text
    ini = IniFile(os.path.normpath(dataset))

    # --- the notebook's init sequence (EXAMPLE_EVALUATE1.ipynb) ---
    ci.initial_setup()
    ci.init_ggl_exclude(np.array(like["ggl_exclude"]).flatten())
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName("nz_lens_file"),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName("nz_source_file"),
        source_ntomo=int(ini.int("source_ntomo")))
    # NLA runs the C FASTPT branch: the same fallback the likelihood
    # prototype applies when IA_model = 0
    ia_code = int(like["IA_code"])
    if int(like["IA_model"]) == 0 and ia_code == 1:
        ia_code = 0
    ci.init_IA(ia_model=int(like["IA_model"]),
               ia_redshift_evolution=int(like["IA_redshift_evolution"]),
               ia_code=ia_code)
    ci.init_probes(possible_probes="xi")
    ci.init_binning(int(ini.int("n_theta")),
                    ini.float("theta_min_arcmin"),
                    ini.float("theta_max_arcmin"))
    ci.init_data_real(ini.relativeFileName("cov_file"),
                      ini.relativeFileName("mask_file"),
                      ini.relativeFileName("data_file"))
    ci.init_ntable_lmax(int(like["lmax"]))
    ci.init_accuracy_boost(float(like["accuracyboost"]),
                           int(like["integration_accuracy"]))

    # --- one CAMB run at the frozen point, notebook grids ---
    print("    running CAMB (notebook-style grids)...", flush=True)
    # the returned 7-tuple unpacks by position into the named grids;
    # the backslash continues the statement on the next line
    (log10k_2D, z_2D, lnPL, lnPNL, G_growth, z_1D, chi) = \
        _get_camb_cosmology(point, camb_args, like["kmax_boltzmann"])

    ci.set_cosmology(omegam=point["omegam"],
                     H0=point["H0"],
                     log10k_2D=log10k_2D,
                     z_2D=z_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_1D=z_1D,
                     chi=chi)

    # nuisance vectors ordered by tomographic bin, zero-padded to the
    # 8 source bins exactly as the notebook passes them. The
    # comprehension collects roman_M1..roman_M8 in bin order:
    # range(1, 9) runs from 1 and stops BEFORE 9
    ci.set_nuisance_shear_calib(
        M=[point[f"roman_M{i}"] for i in range(1, 9)])
    ci.set_nuisance_shear_photoz(
        bias=[point[f"roman_DZ_S{i}"] for i in range(1, 9)])
    # A2/BTA come from the frozen point too; under NLA (IA_model = 0)
    # the C code ignores them, matching the cobaya reference run
    ci.set_nuisance_ia(
        A1=[point["roman_A1_1"], point["roman_A1_2"], 0, 0, 0, 0, 0, 0],
        A2=[point["roman_A2_1"], point["roman_A2_2"], 0, 0, 0, 0, 0, 0],
        B_TA=[point["roman_BTA_1"], 0, 0, 0, 0, 0, 0, 0])

    datavector = np.array(ci.compute_data_vector_masked())
    # float() converts the returned value to a plain python float;
    # :.6f below prints it with a fixed six decimals
    chi2 = float(ci.compute_chi2(datavector))
    print(f"    direct-interface chi2 = {chi2:.6f}", flush=True)
    if not np.isfinite(chi2):
        raise AssertionError("non-finite chi2 from the direct interface")
    return chi2


def _worker(result_path):
    """Worker-side entry: compute the chi2 and save it as json.

    Arguments:
      result_path = file the chi2 is written into; the parent reads
                    it back. Progress prints go to the inherited
                    stdout, so the terminal streams them.

    Returns:
      nothing; the result lands in result_path.
    """
    u.require_cocoa_environment()
    value = _notebook_chi2_impl()
    # json.dump writes the chi2 into the file as json text; the with
    # block closes the file even when the dump fails
    with open(result_path, "w") as f:
        json.dump(value, f)


def notebook_interface_chi2():
    """chi2 of the notebook call path, evaluated in a fresh worker.

    Returns:
      the chi2 as a float.

    Raises:
      RuntimeError when the worker dies without writing a result: the
      cosmolike C layer aborts the process on an internal
      inconsistency (a changed binding or a rejected grid) instead of
      raising, and that abort is precisely what this test watches for.
    """
    import subprocess
    import tempfile

    # .get returns None when the flag is absent, so a normal
    # (parent) process falls through and spawns the worker below
    if os.environ.get(_WORKER_FLAG) == "1":
        return _notebook_chi2_impl()
    # delete=False keeps the file when the with block closes it: only
    # a fresh unique NAME is needed; the worker writes the file and
    # the finally below removes it
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as tmp:
        result_path = tmp.name
    # dict(os.environ) is a COPY of the environment: the edits below
    # reach only the worker subprocess, never this process
    environment = dict(os.environ)
    environment[_WORKER_FLAG] = "1"
    environment["OMP_NUM_THREADS"] = u.REQUIRED_OMP_THREADS
    # subprocess.run starts the worker and BLOCKS until it exits;
    # env=environment hands the child the edited environment copy
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER_DRIVER, _THIS_FILE, result_path],
        env=environment)
    # the finally below runs on EVERY exit from the try, an exception
    # included, so the temporary file never outlives this call
    try:
        if completed.returncode != 0:
            raise RuntimeError(
                "notebook-interface worker exited with code "
                f"{completed.returncode} before writing a result; a "
                "cosmolike-level abort prints its reason (e.g. a grid "
                "rejection from basics.c) just above")
        # json.load parses the worker's file back into the number it
        # dumped; float() pins the type
        with open(result_path) as f:
            return float(json.load(f))
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


# =============================================================================
# THE TEST
# =============================================================================
class TestNotebookInterface(unittest.TestCase):
    """Test 5, sharing the suite's frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must
    fail loudly before any physics runs), and loads the frozen
    reference chi2 values.
    """

    # @classmethod hands the class itself in as cls; unittest calls
    # this once, before the first test of the class
    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_5_notebook_interface_chi2(self):
        """The direct (notebook-style) call path agrees with cobaya."""
        chi2 = notebook_interface_chi2()
        ref = self.reference[f"{EXAMPLE}_nla"]
        u.report_chi2_test(
            5, "example1 notebook-style direct interface (cosmic shear, "
               "NLA) chi2 vs frozen cobaya reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"direct-interface chi2 = {chi2:.6f} vs frozen cobaya "
                f"reference {ref:.6f} (|delta| >= {u.CHI2_TOLERANCE}); "
                "an interface-binding or grid-validation change likely "
                "broke the notebook call path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
