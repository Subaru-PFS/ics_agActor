"""Tests for find_representative_spot.

Test values are derived from the 2026-05-13T17:12:30.695 AG log which
contained ~2095 guiding frames. Typical (non-spike) peaks were ~280,
fluxes ~13400, and sizes ~0.77 arcsec. A near-saturated source on CAM2
produced peaks ~24000, triggering 664x frame-to-frame spikes in the old
single-star implementation.
"""

import numpy as np
import pandas as pd
import pytest

from agActor.field_acquisition import find_representative_spot


def _make_detected(n, *, peak=280.0, flux=13400.0, moment_20=2.5, moment_02=2.3, moment_11=0.1):
    """Build a minimal detected_objects DataFrame with n rows of identical values."""
    return pd.DataFrame({
        "peak_intensity": np.full(n, peak),
        "image_moment_00_pix": np.full(n, flux),
        "central_image_moment_20_pix": np.full(n, moment_20),
        "central_image_moment_02_pix": np.full(n, moment_02),
        "central_image_moment_11_pix": np.full(n, moment_11),
    })


def _make_identified(ids):
    """Build a minimal identified_objects DataFrame from a list of detected_object_id values."""
    return pd.DataFrame({"detected_object_id": ids})


class TestFindRepresentativeSpot:
    """Tests derived from on-sky log 2026-05-13T17:12:30.695."""

    def test_empty_identified_returns_zeros(self):
        detected = _make_detected(5)
        identified = _make_identified([])
        assert find_representative_spot(detected, identified) == (0.0, 0.0, 0.0)

    def test_single_star(self):
        detected = _make_detected(1, peak=280.0, flux=13400.0)
        identified = _make_identified([0])
        flux, peak, size = find_representative_spot(detected, identified)
        assert peak == pytest.approx(280.0)
        assert flux == pytest.approx(13400.0)
        assert size > 0

    def test_uniform_population(self):
        """20 identical stars should return exactly those values."""
        detected = _make_detected(20, peak=280.0, flux=13400.0)
        identified = _make_identified(list(range(20)))
        flux, peak, size = find_representative_spot(detected, identified)
        assert peak == pytest.approx(280.0)
        assert flux == pytest.approx(13400.0)

    def test_outlier_does_not_dominate(self):
        """One near-saturated source (peak ~24000) among 19 normal stars (peak ~280).

        This is the core bug: the old implementation could return the outlier's
        peak (24000) instead of a value representative of the population (~280).
        With nanmedian, the returned peak must stay near 280.
        """
        n_normal = 19
        detected = _make_detected(n_normal + 1, peak=280.0, flux=13400.0)
        detected.loc[0, "peak_intensity"] = 24000.0
        detected.loc[0, "image_moment_00_pix"] = 2_200_000.0

        identified = _make_identified(list(range(n_normal + 1)))
        flux, peak, size = find_representative_spot(detected, identified)

        assert peak == pytest.approx(280.0, rel=0.1)
        assert flux == pytest.approx(13400.0, rel=0.1)

    def test_high_saturation_outlier(self):
        """Extreme outlier (peak ~58000, near satVal2=58300) among 19 normal stars.

        Derived from frame pairs 1024217/1024218 where raw max ~58k on CAM2
        caused a 664x spike in the old implementation.
        """
        n_normal = 19
        detected = _make_detected(n_normal + 1, peak=280.0, flux=13400.0)
        detected.loc[0, "peak_intensity"] = 58000.0
        detected.loc[0, "image_moment_00_pix"] = 4_000_000.0

        identified = _make_identified(list(range(n_normal + 1)))
        _, peak, _ = find_representative_spot(detected, identified)

        assert peak == pytest.approx(280.0, rel=0.1)

    def test_all_nan_peaks(self):
        """All-NaN inputs should not raise."""
        detected = _make_detected(5)
        detected["peak_intensity"] = np.nan
        detected["image_moment_00_pix"] = np.nan

        identified = _make_identified(list(range(5)))
        flux, peak, size = find_representative_spot(detected, identified)

        assert np.isnan(flux)
        assert np.isnan(peak)

    def test_size_is_positive(self):
        """Size must be non-negative even with edge-case moments."""
        detected = _make_detected(5, moment_20=0.0, moment_02=0.0, moment_11=0.0)
        identified = _make_identified(list(range(5)))
        _, _, size = find_representative_spot(detected, identified)
        assert size >= 0.0

    def test_negative_moment_product_does_not_raise(self):
        """Negative semi-axis product (from noisy moments) must not produce nan via sqrt."""
        detected = _make_detected(3, moment_20=0.5, moment_02=0.5, moment_11=1.0)
        identified = _make_identified(list(range(3)))
        _, _, size = find_representative_spot(detected, identified)
        assert np.isfinite(size)
        assert size >= 0.0

    def test_subset_of_detections(self):
        """Only matched stars contribute, not the full detected_objects table."""
        detected = _make_detected(10, peak=280.0, flux=13400.0)
        detected.loc[0, "peak_intensity"] = 58000.0
        detected.loc[0, "image_moment_00_pix"] = 4_000_000.0

        identified = _make_identified([1, 2, 3, 4, 5])
        _, peak, _ = find_representative_spot(detected, identified)

        assert peak == pytest.approx(280.0)

    def test_frame_to_frame_stability(self):
        """Simulates two consecutive frames where the matched population is stable
        but one star's position in the list shifts. The median must stay stable.

        In the old implementation, argpartition over pointing errors meant that
        even a small change in error ordering could jump from a normal star to
        the outlier, causing >100x spikes.
        """
        np.random.seed(42)
        n = 20

        frame_a = _make_detected(n, peak=280.0, flux=13400.0)
        frame_a.loc[0, "peak_intensity"] = 24000.0
        frame_a.loc[0, "image_moment_00_pix"] = 2_200_000.0

        frame_b = frame_a.copy()
        frame_b["peak_intensity"] += np.random.normal(0, 5, n)
        frame_b.loc[0, "peak_intensity"] = 24000.0

        identified = _make_identified(list(range(n)))

        _, peak_a, _ = find_representative_spot(frame_a, identified)
        _, peak_b, _ = find_representative_spot(frame_b, identified)

        ratio = max(peak_a, peak_b) / min(peak_a, peak_b)
        assert ratio < 1.1
