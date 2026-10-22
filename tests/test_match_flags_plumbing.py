"""The SourceMatchingFlags reach agc_match through the matcher and get_guide_offsets (INSTRM-3038).

Two seams are pinned here: RADECInRShiftA returns the flag as column 11 of its match
result, and get_guide_offsets turns that column into ``identified_objects.match_flags``
and defines ``matched`` from it.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from agActor import field_acquisition
from agActor.coordinates.Subaru_POPT2_PFS_AG import PFS
from pfs.utils.datamodel.ag import SourceMatchingFlags as F

# A fabricated catalog basis: six stars 50 mm apart on the detector plane, with the
# RA/Dec Jacobians RADECInRShiftA expects in columns 3-6 (InR columns 7-8 unused).
STARS = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0], [100.0, 0.0], [0.0, 100.0]])
SHIFT = np.array([0.05, 0.0])  # the true pointing error, mm: exactly 0.5 of the RA perturbation


def _catalog():
    n = len(STARS)
    cat = np.zeros((n, 9))
    cat[:, 0:2] = STARS
    cat[:, 2] = 10.0  # mag
    cat[:, 3] = 0.1  # dx/dra
    cat[:, 6] = 0.1  # dy/dde
    return cat


def test_radec_inr_shift_returns_match_flags_as_column_11():
    cat = _catalog()
    detections = np.vstack(
        [
            STARS[:4] + SHIFT,  # four clean matches
            STARS[0] + SHIFT + [0.30, 0.0],  # second claim on star 0, farther: unused, and an outlier
            STARS[4] + SHIFT + [1.0, 0.0],  # near star 4 but 1 mm off: a bad residual
            STARS[0] + [25.0, 25.0],  # 35 mm from every star: no match
        ]
    )
    n = len(detections)
    ra_offset, de_offset, _inr, _scale, result = PFS().RADECInRShiftA(
        detections[:, 0], detections[:, 1], np.full(n, 1000.0), np.zeros(n),
        cat, cat, fit_inr=False, fit_scale=False, maxresid=0.5,
    )
    assert result.shape == (n, 12)
    flags = result[:, 11].astype(int)
    assert flags[:4].tolist() == [F.GOOD_MATCH] * 4
    assert flags[4] == F.BAD_RESIDUAL | F.UNUSED_MULTI_MATCH
    assert flags[5] == F.BAD_RESIDUAL
    assert flags[6] == F.NO_MATCH
    # The flag agrees with the matcher's own inlier column, and the duplicate claims star 0.
    assert np.array_equal(result[:, 8] == 1.0, flags == F.GOOD_MATCH)
    assert result[4, 9] == 0 and result[0, 9] == 0
    assert ra_offset == pytest.approx(0.5 * (1.0 / 3600.0)) and de_offset == pytest.approx(0.0)
    # Negative control: column 8 alone (the old flag) says nothing about the no-match row
    # beyond "not an inlier"; it cannot tell rows 5 and 6 apart, the flags can.
    assert result[5, 8] == result[6, 8] and flags[5] != flags[6]


def _frames():
    detected = pd.DataFrame(
        {
            "agc_camera_id": [0, 0, 0, 3],
            "spot_id": [0, 1, 2, 0],
            "centroid_x_pix": [100.0, 200.0, 300.0, 400.0],
            "centroid_y_pix": [100.0, 200.0, 300.0, 400.0],
            "central_image_moment_11_pix": 0.1,
            "central_image_moment_20_pix": 2.5,
            "central_image_moment_02_pix": 2.3,
            "peak_intensity": 280.0,
            "image_moment_00_pix": 13400.0,
            "flags": 0,
        }
    )
    guide = pd.DataFrame(
        {"agc_camera_id": [0, 0, 3], "x": [101.0, 201.0, 401.0], "y": [101.0, 201.0, 401.0], "source_id": [1, 2, 3]}
    )
    return detected, guide


def test_get_guide_offsets_carries_match_flags_and_defines_matched(monkeypatch):
    detected, guide = _frames()
    # What the matcher returns for the four detections: columns 0-7 positions and
    # residuals, 8 inlier, 9 catalog index, 10 camera_enabled, 11 match_flags.
    match_results = np.zeros((4, 12))
    match_results[:, 8] = [1, 1, 1, 0]  # the duplicate (row 2) is an inlier of the fit ...
    match_results[:, 9] = [0, 1, 1, 2]  # ... but claims star 1 after row 1
    match_results[:, 10] = [1, 1, 1, 0]
    match_results[:, 11] = [F.GOOD_MATCH, F.GOOD_MATCH, F.UNUSED_MULTI_MATCH, F.BAD_RESIDUAL]

    def fake_calculate_offsets(*, guide_objects, detected_array, **kwargs):
        valid = np.ones(len(detected_array), dtype=bool)
        return 0.0, 0.0, np.nan, np.nan, match_results, 0.0, valid, guide_objects

    monkeypatch.setattr(field_acquisition, "calculate_offsets", fake_calculate_offsets)
    offsets = field_acquisition.get_guide_offsets(
        guide_objects=guide, detected_objects=detected, ra=0.0, dec=0.0,
        taken_at=datetime(2026, 10, 22, tzinfo=UTC), adc=0.0,
    )
    ident = offsets.identified_objects
    assert ident["match_flags"].tolist() == [1, 1, int(F.UNUSED_MULTI_MATCH), int(F.BAD_RESIDUAL)]
    assert ident["match_flags"].dtype.kind == "i"
    assert ident["matched"].tolist() == [1, 1, 0, 0]
    assert ident["guide_object_id"].tolist() == [0, 1, 1, 2]
    # Negative control: `matched` is not the matcher's inlier column any more.
    assert ident["matched"].tolist() != match_results[:, 8].astype(int).tolist()
    assert "inlier" not in ident.columns
