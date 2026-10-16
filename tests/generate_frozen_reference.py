"""Maintainer tool: (re)create the frozen state the unit tests run on.

Running this redefines what the tests protect, so it refuses to run
without the explicit --overwrite flag. Only run it when a change to
the data vectors, n(z), covariance, examples, or likelihood defaults
is deliberate, and review the printed chi2 values before committing:
they become the new references the tests compare against.

The machinery and its documentation live in
cosmolike_core/cocoa_testing_freeze.py (freeze_main lists the modes
and the stage order); this script keeps the project's data:
the generated-vector roster, the per-mask
descriptor variants and the halo.c reference step.

Usage (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/roman_real/tests/generate_frozen_reference.py --overwrite

Incremental modes on an existing frozen state (each verifies the
state against the manifest, then re-pins it):
  --tatt-masks rewrites the per-mask descriptors;
  --baryons regenerates the feedback vectors;
  --halo rewrites the halo.c reference.
"""

import os

# The references must be produced under the same OpenMP thread count
# the tests enforce, and before any cobaya/cosmolike import.
os.environ["OMP_NUM_THREADS"] = "4"

import sys

# __file__ is this script's own path; insert(0, ...) puts the tests/
# folder first on the module search path, so the import below finds
# the shim no matter where the script was launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# The data vectors --overwrite regenerates:
# {descriptor dataset: (generating example, TATT flag)},
# built from the shim's generator pairing.
VECTORS = {name: (example, True)
           for name, example in u.TATT_GENERATORS.items()}

# The --mask reruns of the comparison sweeps read one frozen TATT
# dataset descriptor per scale-cut mask: identical to the base TATT
# descriptor except for its mask_file line (the "ones" entry of the
# fastpt_masks of cocoa_test_utils). variant -> (base, mask).
TATT_MASK_VARIANTS = {
    "tatt_roman_real_ones.dataset": ("tatt_roman_real.dataset", "ones.mask"),
}

import json
import time



def generate_halo_reference(stamp):
    """Write frozen/halo_reference.json, the ground truth of test_halo.py.

    The probe grids, the configuration and the pinned HOD
    parameters all live in test_halo.py, which this function imports,
    so the generator and the tests can never describe different
    states. It builds that state (a model at the frozen fiducial
    point), evaluates every probe outside test_halo.HEAD_DEFECTS, and
    stores inputs and values together; the listed probes would abort
    the process and are left out (the _meta block records the list).
    Runs inside a
    --halo-one worker subprocess: it builds a model (see main's note).

    Arguments:
      stamp = the UTC time string written into the _meta block.

    Returns:
      nothing; frozen/halo_reference.json is written.

    Raises:
      whatever test_halo.build_halo_state raises when the model fails
      to build or evaluate.
    """
    # The halo test defines the probe grid and HOD state. Import it from
    # the data-vector sector; stored reference files remain in tests/frozen/.
    from data_vector import test_halo as th

    t0 = time.time()
    print("freezing the halo.c probes ...", flush=True)
    state = th.build_halo_state()
    probes = th.freeze_probes(state)
    # dict(a, **b) builds one dictionary from a's entries plus b's:
    # the generation record plus the settings the values depend on
    reference = {
        "_meta": dict({"generated_utc": stamp,
                       "omp_num_threads": os.environ["OMP_NUM_THREADS"]},
                      **th.frozen_meta()),
        "probes": probes,
    }
    with open(th.HALO_REFERENCE_FILE, "w") as f:
        # written like the chi2 reference file; json stores the floats
        # in their shortest exact form, so every value round-trips
        # bit for bit
        json.dump(reference, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"halo reference: {len(probes)} probes frozen "
          f"({time.time() - t0:.1f}s); left out (current halo.c "
          f"defects): {sorted(th.HEAD_DEFECTS)}", flush=True)


if __name__ == "__main__":
    sys.exit(u.freeze_main(
        u, script_file=__file__, usage=__doc__,
        vectors=VECTORS,
        mask_variants=TATT_MASK_VARIANTS,
        extra_steps=(("--halo", "--halo-one", True,
                  generate_halo_reference),),
    ))
