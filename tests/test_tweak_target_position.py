"""tweak_target_position runs on pandas 2 and pandas 3 and leaves the catalog's parallax alone."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from agActor.utils.data import tweak_target_position


def test_tweak_target_position_does_not_write_into_the_catalog():
    guide_objects = pd.DataFrame(
        {
            "ra": [150.6, 149.4],
            "dec": [2.0, 2.0],
            "pm_ra": [1.0, -2.0],
            "pm_dec": [0.5, 0.0],
            "parallax": [0.5, 0.0],  # the second is below the 1e-5 mas floor pfs_utils applies
            "agc_camera_id": [0, 3],
        }
    )
    tweaked = tweak_target_position(guide_objects, 150.0, 2.0, 0.0, datetime(2026, 8, 31, 12, tzinfo=UTC))

    assert np.isfinite(tweaked[["ra", "dec", "x", "y"]].to_numpy()).all()
    # pfs_utils clamps the parallax array it is given in place. Handed the frame's own array, that
    # raises on pandas 3 (read-only) and rewrites the column on pandas 2; neither may happen.
    assert tweaked["parallax"].tolist() == [0.5, 0.0]
    assert guide_objects["parallax"].tolist() == [0.5, 0.0]
