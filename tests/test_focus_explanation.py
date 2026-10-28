"""Focus explains which cameras gave no value and why (INSTRM-2800)."""

import numpy as np
import pandas as pd
from pfs.utils.datamodel.ag import SourceDetectionFlags

from agActor.utils.focus import explain_missing_focus, focus

RIGHT = int(SourceDetectionFlags.RIGHT)
BAD_SHAPE = int(SourceDetectionFlags.BAD_SHAPE)


def spots(rows):
    """Detected objects from (zero-based camera, flags) pairs, round 2-3 pixel spots."""
    return pd.DataFrame(
        [
            dict(
                agc_camera_id=camera,
                spot_id=i,
                central_image_moment_11_pix=0.0,
                central_image_moment_20_pix=4.0 + 0.1 * i,
                central_image_moment_02_pix=4.0 + 0.1 * i,
                flags=flags,
            )
            for i, (camera, flags) in enumerate(rows)
        ]
    )


def test_single_star_gives_nan_and_a_reason():
    # The ticket's case: one usable star (on AG1's left half) used to raise an IndexError.
    detected = spots([(0, 0)])
    dz, dzs = focus(detected_objects=detected, enabledCameras=[1, 2, 3, 4, 5, 6])
    assert np.isnan(dz)
    assert np.isnan(dzs).all()
    assert explain_missing_focus(detected, dzs, [1, 2, 3, 4, 5, 6]) == (
        "AG1: 0 stars on the right half (1 on the left); AG2: no stars; AG3: no stars; "
        "AG4: no stars; AG5: no stars; AG6: no stars"
    )


def test_camera_with_both_halves_is_not_reported():
    detected = spots([(1, 0), (1, 0), (1, RIGHT), (1, RIGHT), (3, RIGHT)])
    dz, dzs = focus(detected_objects=detected, enabledCameras=[2, 4])
    # Negative control: AG2 has stars on both halves, so it has a value and no reason.
    assert np.isfinite(dzs[1])
    assert np.isfinite(dz)
    assert explain_missing_focus(detected, dzs, [2, 4]) == "AG4: 0 stars on the left half (1 on the right)"


def test_disabled_cameras_and_flagged_spots_are_left_out():
    # AG1 is disabled; AG3's only right-half spot is flagged BAD_SHAPE, which focus does not use.
    detected = spots([(0, 0), (0, RIGHT), (2, 0), (2, RIGHT | BAD_SHAPE)])
    _, dzs = focus(detected_objects=detected, enabledCameras=[3])
    assert explain_missing_focus(detected, dzs, [3]) == "AG3: 0 stars on the right half (1 on the left)"
    # Negative control: with AG1 enabled it has both halves and gets a value.
    _, dzs_all = focus(detected_objects=detected, enabledCameras=[1, 3])
    assert np.isfinite(dzs_all[0])
    assert (
        explain_missing_focus(detected, dzs_all, [1, 3]) == "AG3: 0 stars on the right half (1 on the left)"
    )


def test_nothing_to_explain_when_every_camera_has_a_value():
    detected = spots([(c, f) for c in range(6) for f in (0, RIGHT)])
    _, dzs = focus(detected_objects=detected)
    assert np.isfinite(dzs).all()
    assert explain_missing_focus(detected, dzs) == ""
