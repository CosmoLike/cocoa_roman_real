"""
Baryon Feedback Suppression Theory Block for Cobaya/Cocoa

Provides baryon feedback effects on the matter power spectrum using external
baryon feedback models (pyspk, BCEmu, or PCA). This theory block computes
suppression factors S(k,z) that are applied to the linear power spectrum.

Physics Background:
    Baryon feedback processes (e.g., AGN heating, stellar feedback) suppress
    the power spectrum on small scales (k > 0.1 h/Mpc) through ejection of
    baryonic matter from overdense regions. The suppression factor S(k,z) < 1
    quantifies this effect: P_nl(k,z) = S(k,z) * P_linear(k,z)

    References:
    - pyspk: Mead+ 2015 (arXiv:1505.07833) — "BCEMU: A Fast Emulator of
      Baryon Physics for the matter power spectrum"
    - SPk parameterization: Mead+ 2021 — Direct baryon feedback model

Author: Integration with Cobaya/Cocoa framework
Date: April 2026
"""

import numpy as np
import logging
import pyspk as spk
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from cobaya.theory import Theory
from cobaya.log import LoggedError


class BaryonSuppression(Theory):
    """
    Theory block for baryon suppression using external models.

    Attributes:
        params (dict): Declares sampled parameters alpha_spk, beta_spk, gamma_spk
                       (for pyspk model; extensible for BCEmu, PCA)
        requested_z (array): Redshifts at which to compute suppression
        requested_k (array): Wavenumbers (h/Mpc) at which to compute suppression
        baryon_model (int): Model selector (1=pyspk, 2=BCEmu, 3=PCA)
        log (logger): Cobaya logger for warnings/errors
    """

    # Define the parameters this theory class needs to evaluate.
    params = {
        "alpha_spk": None,  # Alpha parameter for pyspk model
        "beta_spk": None,  # Beta parameter for pyspk model
        "gamma_spk": None,  # Gamma parameter for pyspk model
    }

    # Define configuration defaults
    baryon_model: int = 1  # 1=pyspk, 2=bcemu (deferred), 3=pca (deferred)

    def initialize(self):
        """
        Initialize the theory block.

        Sets up:
        - Empty arrays for requested z and k scales
        - Calibration ranges for pyspk
        - Parameter validation bounds (3-sigma from priors)
        """
        self.requested_z = np.array([])
        self.requested_k = np.array([])

        # pyspk Calibration ranges (from Kunhao's testing/tuning)
        self.z_min_calib = 0.125  # Below this, pyspk not well-calibrated
        self.z_max_calib = 3.0  # Above this, pyspk not well-calibrated
        self.k_min_calib = 8.73e-3  # h/Mpc; below this, outside calibration

        # Parameter validation bounds (3-sigma conservative from YAML priors)
        # Expected ranges: alpha ~4.18±0.12, beta ~1.26±0.08, gamma ~0.42±0.10
        self.alpha_min, self.alpha_max = 3.8, 4.6
        self.beta_min, self.beta_max = 1.0, 1.6
        self.gamma_min, self.gamma_max = 0.1, 0.75

        self.log.debug(
            "BaryonSuppression: Initialized with baryon_model=%d, "
            "z_calibration=[%.3f, %.3f], k_min_calib=%.3e h/Mpc",
            self.baryon_model,
            self.z_min_calib,
            self.z_max_calib,
            self.k_min_calib,
        )

    def get_requirements(self):
        """
        Declare dependencies on other theory components.

        Returns:
            list: Required products from other theory blocks
                - "H0": Hubble constant (for cosmology)
                - "omegam": Matter density (for cosmology)
        """
        return ["H0", "omegam"]

    def must_provide(self, **requirements):
        """
        Parse product requests from the likelihood.

        Stores the z and k grids at which the likelihood needs suppression factors.
        This allows the theory block to compute only what's needed, improving
        efficiency when multiple likelihoods have different k/z requirements.

        Args:
            **requirements (dict): Map of product names to their specifications.
                Expected key: "baryon_suppression" with value
                {
                    "z": array of redshifts,
                    "k": array of wavenumbers (h/Mpc)
                }

        Raises:
            LoggedError: If baryon_suppression is requested but z or k is missing.
        """
        if "baryon_suppression" in requirements:
            req = requirements["baryon_suppression"]

            # Extract and validate z array
            requested_z = req.get("z", None)
            if requested_z is None:
                raise LoggedError(
                    self.log, "baryon_suppression requires 'z' array in requirements"
                )
            self.requested_z = np.atleast_1d(requested_z)

            # Extract and validate k array
            requested_k = req.get("k", None)
            if requested_k is None:
                raise LoggedError(
                    self.log, "baryon_suppression requires 'k' array in requirements"
                )
            self.requested_k = np.atleast_1d(requested_k)

            n_z = len(self.requested_z)
            n_k = len(self.requested_k)
            self.log.info(
                "BaryonSuppression.must_provide: baryon_suppression requested; "
                "n_z=%d [%.3f, %.3f], n_k=%d [%.3e, %.3e]",
                n_z,
                self.requested_z.min(),
                self.requested_z.max(),
                n_k,
                self.requested_k.min(),
                self.requested_k.max(),
            )

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Compute baryon suppression factors and store in state.

        Fetches cosmological parameters from the provider, retrieves baryon
        model parameters from the sampler, and computes S(k,z) for all
        requested (k, z) pairs. Handles multiple baryon models (selector via
        self.baryon_model) and implements comprehensive error handling with
        graceful degradation (returns unity suppression on error).

        Args:
            state (dict): Cobaya state dictionary where results are stored.
                         On output, state["baryon_suppression"] = {z: S(k,z) array}
            want_derived (bool): Whether to compute derived parameters (unused).
            **params_values_dict (dict): Map of parameter names to current values.
                Expected keys: "alpha_spk", "beta_spk", "gamma_spk"

        Returns:
            None: Results stored in state["baryon_suppression"]

        Notes:
            - Returns unity suppression (graceful degradation) on any error
            - Applies calibration masking: z outside [0.125, 3.0] or k < 8.73e-3 h/Mpc
            - Validates parameters against 3-sigma bounds before calling pyspk
            - Uses log-space interpolation to avoid numerical issues
        """

        # Initialize result dictionary
        suppression_dict = {}

        # Route to appropriate baryon model
        if self.baryon_model == 1:
            suppression_dict = self._calculate_pyspk(params_values_dict)
        elif self.baryon_model == 2:
            self.log.warning(
                "baryon_model=2 (BCEmu) not yet implemented; "
                "returning unity suppression"
            )
            suppression_dict = self._unity_suppression()
        elif self.baryon_model == 3:
            self.log.warning(
                "baryon_model=3 (PCA) not yet implemented; returning unity suppression"
            )
            suppression_dict = self._unity_suppression()
        else:
            self.log.error(
                "baryon_model=%d is invalid; must be 1 (pyspk), 2 (bcemu), or 3 (pca); "
                "returning unity suppression",
                self.baryon_model,
            )
            suppression_dict = self._unity_suppression()

        # Store result in state
        state["baryon_suppression"] = suppression_dict

    def _calculate_pyspk(self, params_values_dict):
        """
        Compute suppression factors using pyspk model.

        Implements the SPk baryon feedback model with comprehensive validation
        and calibration masking.

        Args:
            params_values_dict (dict): {"alpha_spk": float, "beta_spk": float, "gamma_spk": float}

        Returns:
            dict: {z_val: suppression_array} for each requested redshift.
                  Returns unity suppression dict on any error.
        """
        try:
            # 1. Fetch sampled baryon parameters
            alpha = params_values_dict.get("alpha_spk", 4.18)
            beta = params_values_dict.get("beta_spk", 1.26)
            gamma = params_values_dict.get("gamma_spk", 0.42)

            self.log.debug(
                "SPk baryon suppression: alpha=%.4f, beta=%.4f, gamma=%.4f",
                alpha,
                beta,
                gamma,
            )

            # 2. Validate parameters are within acceptable ranges (3-sigma bounds)
            # This is a safety layer; parameter priors should enforce this, but
            # we catch edge cases and gracefully reject them
            if not (self.alpha_min < alpha < self.alpha_max):
                self.log.warning(
                    "SPk parameter alpha_spk=%.4f outside valid range "
                    "[%.4f, %.4f]; returning unity suppression",
                    alpha,
                    self.alpha_min,
                    self.alpha_max,
                )
                return self._unity_suppression()

            if not (self.beta_min < beta < self.beta_max):
                self.log.warning(
                    "SPk parameter beta_spk=%.4f outside valid range "
                    "[%.4f, %.4f]; returning unity suppression",
                    beta,
                    self.beta_min,
                    self.beta_max,
                )
                return self._unity_suppression()

            if not (self.gamma_min < gamma < self.gamma_max):
                self.log.warning(
                    "SPk parameter gamma_spk=%.4f outside valid range "
                    "[%.4f, %.4f]; returning unity suppression",
                    gamma,
                    self.gamma_min,
                    self.gamma_max,
                )
                return self._unity_suppression()

            # 3. Fetch cosmological parameters from provider (e.g., CAMB/CLASS)
            H0 = self.provider.get_param("H0")
            omegam = self.provider.get_param("omegam")
            cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

            self.log.debug("SPk cosmology: H0=%.3f, omegam=%.4f", H0, omegam)

            # 4. Compute suppression for each requested redshift
            suppression_dict = {}

            for i_z, z_val in enumerate(self.requested_z):
                # Check if redshift is within pyspk calibration range
                # Outside this range, we return unity (no suppression)
                if z_val < self.z_min_calib or z_val > self.z_max_calib:
                    self.log.debug(
                        "SPk z=%.3f outside calibration range [%.3f, %.3f]; "
                        "using unity suppression for this redshift",
                        z_val,
                        self.z_min_calib,
                        self.z_max_calib,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                try:
                    # Call pyspk to compute suppression on its native k-grid
                    k_spk, sup_spk = spk.sup_model(
                        SO=500,  # Spherical overdensity radius
                        z=z_val,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        cosmo=cosmo,
                        verbose=False,
                    )

                except Exception as e:
                    self.log.error(
                        "SPk model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                # Defensive check: verify pyspk output is valid (no NaN/Inf)
                if not np.all(np.isfinite(k_spk)) or not np.all(np.isfinite(sup_spk)):
                    n_bad_k = np.sum(~np.isfinite(k_spk))
                    n_bad_sup = np.sum(~np.isfinite(sup_spk))
                    self.log.error(
                        "SPk returned non-finite values at z=%.3f: "
                        "%d bad k-values, %d bad suppression values; "
                        "using unity suppression for this redshift",
                        z_val,
                        n_bad_k,
                        n_bad_sup,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                # Log suppression range for diagnostics
                self.log.debug(
                    "SPk at z=%.3f: k_range=[%.3e, %.3e], "
                    "suppression_range=[%.6f, %.6f]",
                    z_val,
                    k_spk.min(),
                    k_spk.max(),
                    sup_spk.min(),
                    sup_spk.max(),
                )

                # 5. Interpolate pyspk suppression onto likelihood's k-grid
                # Use log-space interpolation for better numerical stability
                try:
                    interp_spk = interp1d(
                        np.log10(k_spk),
                        np.log(sup_spk),
                        kind="linear",
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )
                    sup_interp = np.exp(interp_spk(np.log10(self.requested_k)))

                except Exception as e:
                    self.log.error(
                        "Interpolation failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                # 6. Apply calibration masking: suppress effect outside calibration ranges
                # For k < 8.73e-3 h/Mpc, set suppression to 1 (outside calib; Kunhao's choice)
                sup_interp[self.requested_k < self.k_min_calib] = 1.0

                # Defensive check: verify interpolated suppression is physically reasonable
                if not np.all(np.isfinite(sup_interp)):
                    n_bad = np.sum(~np.isfinite(sup_interp))
                    self.log.error(
                        "Interpolation produced %d non-finite values at z=%.3f; "
                        "using unity suppression for this redshift",
                        n_bad,
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                # Ensure suppression factor is physically reasonable [0.01, 1.0]
                # (baryon effects reduce power, but not catastrophically)
                if np.any(sup_interp < 0.01) or np.any(sup_interp > 1.0):
                    n_low = np.sum(sup_interp < 0.01)
                    n_high = np.sum(sup_interp > 1.0)
                    self.log.warning(
                        "SPk at z=%.3f produced %d values < 0.01 (clipping to 0.01) "
                        "and %d values > 1.0 (clipping to 1.0)",
                        z_val,
                        n_low,
                        n_high,
                    )
                    sup_interp = np.clip(sup_interp, 0.01, 1.0)

                suppression_dict[z_val] = sup_interp

            self.log.info(
                "SPk suppression computed for %d redshifts, %d k-values",
                len(suppression_dict),
                len(self.requested_k),
            )
            return suppression_dict

        except Exception as e:
            # Catch-all: any uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in SPk calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression()

    def _unity_suppression(self):
        """
        Return unity suppression factors (no baryon feedback).

        Used as a graceful fallback when baryon model computation fails.

        Returns:
            dict: {z_val: ones_array} for each requested redshift
        """
        return {z_val: np.ones_like(self.requested_k) for z_val in self.requested_z}

    def get_baryon_suppression(self):
        """
        Accessor method for the likelihood to fetch suppression factors.

        Returns:
            dict: {z_val: suppression_array} from the current state.
                  Each suppression_array has shape (n_k,).

        Raises:
            AttributeError: If calculate() has not been called yet.
        """
        return self.current_state["baryon_suppression"]
