"""The focus error for second moments that are the spot's variance.

ics_agccActor used to store half the variance, and the focus slope was calibrated on those
moments. It now stores the variance, so the slope is half what it was and a given pair of
spots gives the focus error it always gave.
"""

import numpy as np
import pandas as pd
from agActor.utils.focus import focus

# The calibration made on moments stored at half the variance.
OLD_SLOPE_MM_PER_PIX2 = 0.0086
INTERCEPT_MM = -0.026


def _detected(variance_left: float, variance_right: float) -> pd.DataFrame:
    """Round spots on both halves of every camera, their moments the given variances (pix^2)."""
    rows = []
    for camera_id in range(6):
        for spot_id in range(6):
            right = spot_id % 2
            variance = variance_right if right else variance_left
            rows.append(
                {
                    "agc_camera_id": camera_id,
                    "spot_id": spot_id,
                    "central_image_moment_11_pix": 0.0,
                    "central_image_moment_20_pix": variance,
                    "central_image_moment_02_pix": variance,
                    "flags": right,
                }
            )
    return pd.DataFrame(rows)


def test_focus_error_unchanged_for_the_same_spots():
    variance_left, variance_right = 14.0, 9.0  # pix^2

    dz, dzs = focus(detected_objects=_detected(variance_left, variance_right))

    # What these spots gave when agcc stored half their variance: semi-axes a^2 + b^2 of a
    # round spot is twice its stored moment.
    old_difference = 2 * (variance_left / 2) - 2 * (variance_right / 2)
    expected = old_difference * OLD_SLOPE_MM_PER_PIX2 + INTERCEPT_MM
    np.testing.assert_allclose(dzs, np.full(6, expected), rtol=1e-12)
    np.testing.assert_allclose(dz, expected, rtol=1e-12)

    # Negative control: the old slope on moments at the variance gives twice the focus term.
    new_difference = 2 * variance_left - 2 * variance_right
    assert not np.allclose(dzs, new_difference * OLD_SLOPE_MM_PER_PIX2 + INTERCEPT_MM)
