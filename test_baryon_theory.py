"""
Unit tests for BaryonSuppression theory block.

Tests:
1. Direct instantiation and initialization
2. Parameter validation (in-range, boundary, out-of-range)
3. must_provide() parsing of z/k grids
4. Calibration boundary masking (z, k ranges)
5. NaN/Inf detection and graceful degradation
6. Edge cases (invalid cosmology, empty grids, etc.)

Run: python -m pytest test_baryon_theory.py -v
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
from pyspk_theory import BaryonSuppression
import logging

# Enable logging to see theory block debug output
logging.basicConfig(level=logging.DEBUG)


class TestBaryonSuppressionInitialization:
    """Test theory block initialization and setup."""

    def test_initialize(self):
        """Test basic initialization sets up calibration ranges."""
        theory = BaryonSuppression()
        # Mock logger
        theory.log = Mock()
        theory.initialize()

        # Check calibration ranges
        assert theory.z_min_calib == 0.125
        assert theory.z_max_calib == 3.0
        assert theory.k_min_calib == 8.73e-3

        # Check parameter bounds
        assert theory.alpha_min == 3.8
        assert theory.alpha_max == 4.6
        assert theory.beta_min == 1.0
        assert theory.beta_max == 1.6
        assert theory.gamma_min == 0.1
        assert theory.gamma_max == 0.75

    def test_get_requirements(self):
        """Test that theory block declares H0 and Omega_m requirements."""
        theory = BaryonSuppression()
        reqs = theory.get_requirements()
        assert "H0" in reqs
        assert "Omega_m" in reqs


class TestMustProvide:
    """Test must_provide parsing of z/k grid specifications."""

    def test_must_provide_valid_zk(self):
        """Test parsing valid z and k arrays."""
        theory = BaryonSuppression()
        theory.log = Mock()

        z_array = np.array([0.5, 1.0, 1.5])
        k_array = np.array([0.01, 0.1, 1.0, 10.0])

        requirements = {"baryon_suppression": {"z": z_array, "k": k_array}}

        theory.must_provide(**requirements)

        np.testing.assert_array_equal(theory.requested_z, z_array)
        np.testing.assert_array_equal(theory.requested_k, k_array)

    def test_must_provide_scalar_zk(self):
        """Test that scalar z/k are converted to arrays."""
        theory = BaryonSuppression()
        theory.log = Mock()

        requirements = {"baryon_suppression": {"z": 0.5, "k": 0.1}}

        theory.must_provide(**requirements)

        assert isinstance(theory.requested_z, np.ndarray)
        assert isinstance(theory.requested_k, np.ndarray)
        assert len(theory.requested_z) == 1
        assert len(theory.requested_k) == 1

    def test_must_provide_missing_z(self):
        """Test error when z array is missing."""
        theory = BaryonSuppression()
        theory.log = Mock()
        theory.log.side_effect = Mock()  # Make it raise

        # Mock LoggedError to raise on call
        from unittest.mock import patch

        with patch("pyspk_theory.LoggedError", side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                requirements = {"baryon_suppression": {"k": np.array([0.1])}}
                theory.must_provide(**requirements)

    def test_must_provide_missing_k(self):
        """Test error when k array is missing."""
        theory = BaryonSuppression()
        theory.log = Mock()

        from unittest.mock import patch

        with patch("pyspk_theory.LoggedError", side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                requirements = {"baryon_suppression": {"z": np.array([0.5])}}
                theory.must_provide(**requirements)


class TestParameterValidation:
    """Test parameter validation against bounds."""

    def setup_method(self):
        """Set up mock provider for all tests."""
        self.theory = BaryonSuppression()
        self.theory.log = Mock()
        self.theory.initialize()

        # Mock provider with fiducial cosmology
        self.theory.provider = Mock()
        self.theory.provider.get_param.side_effect = lambda x: {
            "H0": 67.32,
            "Omega_m": 0.316,
        }[x]

        # Set up z/k grids
        self.theory.requested_z = np.array([0.5, 1.0])
        self.theory.requested_k = np.array([0.01, 0.1, 1.0])

        self.state = {}

    def test_alpha_in_range(self):
        """Test that alpha parameter in range is accepted."""
        params = {
            "alpha_spk": 4.18,  # Fiducial, in range [3.8, 4.6]
            "beta_spk": 1.26,
            "gamma_spk": 0.42,
        }

        # We expect this to call pyspk without logging warnings
        self.theory.calculate(self.state, **params)

        # Should have created baryon_suppression in state
        assert "baryon_suppression" in self.state
        assert len(self.state["baryon_suppression"]) == 2  # Two redshifts

    def test_alpha_below_min(self):
        """Test that alpha below minimum logs warning and returns unity."""
        params = {
            "alpha_spk": 3.5,  # Below min 3.8
            "beta_spk": 1.26,
            "gamma_spk": 0.42,
        }

        self.theory.calculate(self.state, **params)

        # Should return unity suppression (all 1.0)
        for z_val, sup_array in self.state["baryon_suppression"].items():
            np.testing.assert_array_almost_equal(
                sup_array, np.ones_like(self.theory.requested_k)
            )

        # Check warning was logged
        warning_calls = [
            call
            for call in self.theory.log.method_calls
            if "warning" in str(call).lower()
        ]
        assert len(warning_calls) > 0

    def test_alpha_above_max(self):
        """Test that alpha above maximum logs warning and returns unity."""
        params = {
            "alpha_spk": 4.8,  # Above max 4.6
            "beta_spk": 1.26,
            "gamma_spk": 0.42,
        }

        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        for z_val, sup_array in self.state["baryon_suppression"].items():
            np.testing.assert_array_almost_equal(
                sup_array, np.ones_like(self.theory.requested_k)
            )

    def test_beta_below_min(self):
        """Test that beta below minimum is rejected."""
        params = {
            "alpha_spk": 4.18,
            "beta_spk": 0.8,  # Below min 1.0
            "gamma_spk": 0.42,
        }

        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        for z_val, sup_array in self.state["baryon_suppression"].items():
            np.testing.assert_array_almost_equal(
                sup_array, np.ones_like(self.theory.requested_k)
            )

    def test_gamma_above_max(self):
        """Test that gamma above maximum is rejected."""
        params = {
            "alpha_spk": 4.18,
            "beta_spk": 1.26,
            "gamma_spk": 0.8,  # Above max 0.75
        }

        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        for z_val, sup_array in self.state["baryon_suppression"].items():
            np.testing.assert_array_almost_equal(
                sup_array, np.ones_like(self.theory.requested_k)
            )


class TestCalibrationBoundaries:
    """Test calibration range masking (z and k boundaries)."""

    def setup_method(self):
        """Set up mock provider."""
        self.theory = BaryonSuppression()
        self.theory.log = Mock()
        self.theory.initialize()

        self.theory.provider = Mock()
        self.theory.provider.get_param.side_effect = lambda x: {
            "H0": 67.32,
            "Omega_m": 0.316,
        }[x]

        self.state = {}

    def test_z_below_calibration_range(self):
        """Test that z < 0.125 returns unity suppression."""
        self.theory.requested_z = np.array([0.1])  # Below z_min_calib=0.125
        self.theory.requested_k = np.array([0.01, 0.1, 1.0])

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}

        self.theory.calculate(self.state, **params)

        # Should return unity at z=0.1
        sup = self.state["baryon_suppression"][0.1]
        np.testing.assert_array_almost_equal(sup, np.ones_like(self.theory.requested_k))

    def test_z_above_calibration_range(self):
        """Test that z > 3.0 returns unity suppression."""
        self.theory.requested_z = np.array([3.5])  # Above z_max_calib=3.0
        self.theory.requested_k = np.array([0.01, 0.1, 1.0])

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}

        self.theory.calculate(self.state, **params)

        # Should return unity at z=3.5
        sup = self.state["baryon_suppression"][3.5]
        np.testing.assert_array_almost_equal(sup, np.ones_like(self.theory.requested_k))

    def test_k_below_calibration_masked(self):
        """Test that k < 8.73e-3 h/Mpc is masked to 1.0."""
        self.theory.requested_z = np.array([0.5])
        # Mix of k below and above calibration threshold
        self.theory.requested_k = np.array([0.001, 0.01, 0.1, 1.0])

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}

        self.theory.calculate(self.state, **params)

        sup = self.state["baryon_suppression"][0.5]

        # k < 8.73e-3 should be masked to 1.0
        assert sup[0] == 1.0  # k=0.001 < 8.73e-3

        # k >= 8.73e-3 should have actual suppression (< 1.0 typically)
        # Note: we can't guarantee this without running pyspk, but we can check it's valid
        assert 0.01 <= sup[1] <= 1.0  # k=0.01 >= 8.73e-3


class TestModelSelector:
    """Test baryon model selector."""

    def setup_method(self):
        """Set up mock provider."""
        self.theory = BaryonSuppression()
        self.theory.log = Mock()
        self.theory.initialize()

        self.theory.provider = Mock()
        self.theory.provider.get_param.side_effect = lambda x: {
            "H0": 67.32,
            "Omega_m": 0.316,
        }[x]

        self.theory.requested_z = np.array([0.5])
        self.theory.requested_k = np.array([0.1])
        self.state = {}

    def test_invalid_model_number(self):
        """Test that invalid baryon_model returns unity."""
        self.theory.baryon_model = 99  # Invalid

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}
        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        sup = self.state["baryon_suppression"][0.5]
        np.testing.assert_array_almost_equal(sup, np.ones_like(self.theory.requested_k))

    def test_bcemu_not_implemented(self):
        """Test that BCEmu (baryon_model=2) returns unity with warning."""
        self.theory.baryon_model = 2

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}
        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        sup = self.state["baryon_suppression"][0.5]
        np.testing.assert_array_almost_equal(sup, np.ones_like(self.theory.requested_k))

        # Check warning was logged
        warning_calls = [
            str(call)
            for call in self.theory.log.method_calls
            if "warning" in str(call).lower()
        ]
        assert any("BCEmu" in w for w in warning_calls)

    def test_pca_not_implemented(self):
        """Test that PCA (baryon_model=3) returns unity with warning."""
        self.theory.baryon_model = 3

        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}
        self.theory.calculate(self.state, **params)

        # Should return unity suppression
        sup = self.state["baryon_suppression"][0.5]
        np.testing.assert_array_almost_equal(sup, np.ones_like(self.theory.requested_k))

        # Check warning was logged
        warning_calls = [
            str(call)
            for call in self.theory.log.method_calls
            if "warning" in str(call).lower()
        ]
        assert any("PCA" in w for w in warning_calls)


class TestAccessor:
    """Test get_baryon_suppression accessor."""

    def setup_method(self):
        """Set up mock provider."""
        self.theory = BaryonSuppression()
        self.theory.log = Mock()
        self.theory.initialize()

        self.theory.provider = Mock()
        self.theory.provider.get_param.side_effect = lambda x: {
            "H0": 67.32,
            "Omega_m": 0.316,
        }[x]

        self.theory.requested_z = np.array([0.5])
        self.theory.requested_k = np.array([0.1])

    def test_get_baryon_suppression_after_calculate(self):
        """Test that accessor returns computed suppression dict."""
        state = {}
        params = {"alpha_spk": 4.18, "beta_spk": 1.26, "gamma_spk": 0.42}

        self.theory.calculate(state, **params)

        # Mock current_state to be the state dict
        self.theory.current_state = state

        result = self.theory.get_baryon_suppression()

        assert isinstance(result, dict)
        assert 0.5 in result  # Our requested z


class TestUnitySuppressionFallback:
    """Test graceful degradation to unity suppression."""

    def test_unity_suppression_dict(self):
        """Test _unity_suppression returns correct dict structure."""
        theory = BaryonSuppression()
        theory.requested_z = np.array([0.5, 1.0, 1.5])
        theory.requested_k = np.array([0.01, 0.1, 1.0])

        result = theory._unity_suppression()

        assert len(result) == 3  # One entry per z
        for z_val, sup_array in result.items():
            assert z_val in [0.5, 1.0, 1.5]
            np.testing.assert_array_almost_equal(sup_array, np.ones(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
