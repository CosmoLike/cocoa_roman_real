import numpy as np
import pyspk as spk
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from cobaya.theory import Theory


class BaryonSuppression(Theory):
    # Define the parameters this theory class needs to evaluate.
    params = {"alpha_spk": None, "beta_spk": None, "gamma_spk": None}

    def initialize(self):
        """Initialize empty arrays for the required scales and redshifts."""
        self.requested_z = []
        self.requested_k = []

    def get_requirements(self):
        """Declare dependencies on other theory components (e.g., CAMB/CLASS)."""
        # We need H0 and Omega_m to initialize the astropy cosmology for pyspk
        return ["H0", "Omega_m"]

    def must_provide(self, **requirements):
        """
        Parse requests from the likelihood.
        If 'baryon_suppression' is requested, store the required z and k.
        """
        if "baryon_suppression" in requirements:
            req = requirements["baryon_suppression"]
            self.requested_z = req.get("z", [])
            self.requested_k = req.get("k", [])

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Compute the suppression factor and store it in `state`."""

        # 1. Fetch cosmological parameters from the provider (e.g., CAMB/CLASS)
        H0 = self.provider.get_param("H0")
        Omega_m = self.provider.get_param("Omega_m")
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        # 2. Fetch the sampled pyspk parameters
        alpha = params_values_dict["alpha_spk"]
        beta = params_values_dict["beta_spk"]
        gamma = params_values_dict["gamma_spk"]

        suppression_dict = {}

        # 3. Compute the suppression factor for each requested redshift
        for z_val in self.requested_z:
            # pyspk returns its own k-array and the suppression factor S(k)
            k_spk, sup_spk = spk.sup_model(
                SO=500, z=z_val, alpha=alpha, beta=beta, gamma=gamma, cosmo=cosmo
            )

            # Interpolate onto the likelihood's requested k-grid
            interp_spk = interp1d(
                np.log10(k_spk),
                np.log(sup_spk),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            sup_interp = np.exp(interp_spk(np.log10(self.requested_k)))

            suppression_dict[z_val] = sup_interp

        # 4. Store the result in the state dictionary
        state["baryon_suppression"] = suppression_dict

    def get_baryon_suppression(self):
        """Accessor method so the likelihood can fetch the dict."""
        return self.current_state["baryon_suppression"]
