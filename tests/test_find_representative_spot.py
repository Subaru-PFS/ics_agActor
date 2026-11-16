"""Tests for find_representative_spot.

Test values are derived from the 2026-05-13T17:12:30.695 AG log which
contained ~2095 guiding frames. Typical (non-spike) peaks were ~280,
fluxes ~13400, and sizes ~0.77 arcsec (by the formula of the time). A near-saturated source on CAM2
produced peaks ~24000, triggering 664x frame-to-frame spikes in the old
single-star implementation.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from agActor import field_acquisition
from agActor.field_acquisition import find_representative_spot
from pfs.utils import agMetrics
from pfs.utils.datamodel.ag import SourceDetectionFlags
from pfs.utils.datamodel.ag import SourceMatchingFlags as M


def _make_detected(n, *, peak=280.0, flux=13400.0, moment_20=2.5, moment_02=2.3, moment_11=0.1):
    """Build a minimal detected_objects DataFrame with n rows of identical values."""
    return pd.DataFrame({
        "peak_intensity": np.full(n, peak),
        "image_moment_00_pix": np.full(n, flux),
        "central_image_moment_20_pix": np.full(n, moment_20),
        "central_image_moment_02_pix": np.full(n, moment_02),
        "central_image_moment_11_pix": np.full(n, moment_11),
        "flags": np.zeros(n, dtype=int),
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

    def test_cross_moment_does_not_enter_the_size(self):
        """The size is from the trace radius: a cross moment, even an impossible one, leaves it alone."""
        detected = _make_detected(3, moment_20=0.5, moment_02=0.5, moment_11=1.0)
        identified = _make_identified(list(range(3)))
        _, _, size = find_representative_spot(detected, identified)
        assert size == pytest.approx(agMetrics.momentsToFwhm(0.5, 0.5))
        # Negative control: the determinant radius has no value for these moments.
        assert np.isnan(agMetrics.momentsToFwhm(0.5, 0.5, 1.0, radius="determinant"))

    def test_no_usable_spot_sends_zero_not_nan(self):
        """With a negative moment on every spot there is no FWHM: the size sent to mlp1 is 0."""
        detected = _make_detected(3, moment_20=-1.0, moment_02=2.0)
        identified = _make_identified(list(range(3)))
        flux, peak, size = find_representative_spot(detected, identified)
        assert size == 0.0
        assert flux == pytest.approx(13400.0) and peak == pytest.approx(280.0)
        # Negative control: the image size itself has no value, which is what would otherwise be sent.
        assert np.isnan(field_acquisition.get_image_size(detected, identified).fwhm)

    def test_spot_with_a_negative_moment_is_left_out(self):
        """Spots with a negative second moment among normal ones do not change the size."""
        detected = _make_detected(5)
        identified = _make_identified(list(range(5)))
        _, _, clean_size = find_representative_spot(detected, identified)

        detected.loc[[0, 1, 2], "central_image_moment_20_pix"] = -5.0
        _, _, size = find_representative_spot(detected, identified)

        assert size == pytest.approx(clean_size)

    def test_size_is_the_fwhm_of_the_shared_definition(self):
        """The size is 2.9 times the radius of the stored moments, at 0.137 arcsec per pixel."""
        detected = _make_detected(6, moment_20=6.0, moment_02=2.0, moment_11=1.5)
        _, _, size = find_representative_spot(detected, _make_identified(list(range(6))))
        assert size == pytest.approx(agMetrics.momentsToFwhm(6.0, 2.0))
        assert size == pytest.approx(2.05 * np.sqrt(2) * 2.0 * 13 / 94.7, rel=1e-6)  # trace radius 2 pix
        # Negative controls: the formula used until now (twice the determinant radius at a 15 m focal
        # length), and the new one on the determinant radius.
        determinant = (6.0 * 2.0 - 1.5**2) ** 0.25
        assert size != pytest.approx(2 * determinant * 206.265 * 13 / 15000, rel=0.05)
        assert size != pytest.approx(agMetrics.momentsToFwhm(6.0, 2.0, 1.5, radius="determinant"), rel=0.05)

    def test_image_size_per_detector_half(self):
        """Spots flagged RIGHT are on the other half of the detector, with their own median."""
        detected = _make_detected(6, moment_20=4.0, moment_02=4.0, moment_11=0.0)
        detected.loc[[3, 4, 5], ["central_image_moment_20_pix", "central_image_moment_02_pix"]] = 9.0
        detected.loc[[3, 4, 5], "flags"] = int(SourceDetectionFlags.RIGHT)
        size = field_acquisition.get_image_size(detected, _make_identified(list(range(6))))
        assert size.fwhmLeft == pytest.approx(agMetrics.momentsToFwhm(4.0, 4.0))
        assert size.fwhmRight == pytest.approx(agMetrics.momentsToFwhm(9.0, 9.0))
        assert (size.nLeft, size.nRight) == (3, 3)
        assert size.fwhmLeft < size.fwhm < size.fwhmRight
        # Negative control: without the flag every spot is on the left half.
        detected["flags"] = 0
        unsplit = field_acquisition.get_image_size(detected, _make_identified(list(range(6))))
        assert (unsplit.nLeft, unsplit.nRight) == (6, 0) and np.isnan(unsplit.fwhmRight)

    def test_rows_are_selected_by_label(self):
        """detected_object_id holds labels of detected_objects, not positions."""
        detected = _make_detected(4)
        detected.index = [10, 11, 12, 13]
        detected.loc[13, "peak_intensity"] = 900.0
        _, peak, _ = find_representative_spot(detected, _make_identified([13]))
        assert peak == pytest.approx(900.0)
        # Negative control: the same number read as a position does not exist.
        with pytest.raises(IndexError):
            detected.iloc[[13]]

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


def _offsets_for(monkeypatch, match_flags):
    """get_guide_offsets on five detections: two star-like, three large and bright."""
    n = 5
    detected = pd.DataFrame(
        {
            "agc_camera_id": 0,
            "spot_id": range(n),
            "centroid_x_pix": 100.0 * (1 + np.arange(n)),
            "centroid_y_pix": 100.0 * (1 + np.arange(n)),
            "central_image_moment_11_pix": 0.1,
            "central_image_moment_20_pix": [2.5, 2.5, 30.0, 30.0, 30.0],
            "central_image_moment_02_pix": [2.3, 2.3, 30.0, 30.0, 30.0],
            "peak_intensity": [280.0, 300.0, 9000.0, 9000.0, 9000.0],
            "image_moment_00_pix": [13000.0, 14000.0, 9.0e5, 9.0e5, 9.0e5],
            "flags": 0,
        }
    )
    guide = pd.DataFrame({"agc_camera_id": [0, 0], "x": [101.0, 201.0], "y": [101.0, 201.0], "source_id": [1, 2]})
    match_results = np.zeros((n, 12))
    match_results[:, 9] = [0, 1, 0, 0, 0]
    match_results[:, 10] = 1
    match_results[:, 11] = match_flags

    def fake_calculate_offsets(*, guide_objects, detected_array, **_):
        return 0.0, 0.0, np.nan, np.nan, match_results, 0.0, np.ones(n, dtype=bool), guide_objects

    monkeypatch.setattr(field_acquisition, "calculate_offsets", fake_calculate_offsets)
    return detected, field_acquisition.get_guide_offsets(
        guide_objects=guide, detected_objects=detected, ra=0.0, dec=0.0,
        taken_at=datetime(2026, 11, 15, tzinfo=UTC), adc=0.0,
    )  # fmt: skip


def test_get_guide_offsets_takes_the_representative_spot_from_matched_detections(monkeypatch):
    """Unmatched detections pass the source filter too; they must not set the spot values."""
    detected, offsets = _offsets_for(
        monkeypatch, [M.GOOD_MATCH, M.GOOD_MATCH, M.NO_MATCH, M.NO_MATCH, M.BAD_RESIDUAL]
    )

    assert offsets.peak == pytest.approx(290.0)
    assert offsets.flux == pytest.approx(13500.0)
    # Negative control: the median over every row of identified_objects is an unmatched spot's.
    all_flux, all_peak, all_size = find_representative_spot(detected, offsets.identified_objects)
    assert all_peak == pytest.approx(9000.0) and all_flux == pytest.approx(9.0e5)
    assert offsets.size < all_size
    # The logged image size describes the same spots: the two matched ones, on the left half.
    assert offsets.image_size.fwhm == pytest.approx(offsets.size)
    assert (offsets.image_size.nLeft, offsets.image_size.nRight) == (2, 0)
    line = offsets.describe_image_size(1234)
    assert f"frame_id=1234 fwhm={offsets.size:.3f} arcsec" in line
    assert "(n=2) right=nan (n=0)" in line


def test_get_guide_offsets_spot_without_any_match_uses_every_detection(monkeypatch):
    """A frame with no good match keeps describing its detections, as it did before."""
    detected, offsets = _offsets_for(monkeypatch, [M.NO_MATCH] * 4 + [M.BAD_RESIDUAL])

    assert offsets.identified_objects["matched"].sum() == 0
    assert offsets.peak == pytest.approx(9000.0) and offsets.flux == pytest.approx(9.0e5)
    # Negative control: the spot of an empty selection is all zeros, which is not what is sent.
    assert find_representative_spot(detected, offsets.identified_objects.query("matched == 1")) == (0.0, 0.0, 0.0)
    assert offsets.size > 0.0
