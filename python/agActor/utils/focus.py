import logging

import numpy as np
import pandas as pd
from pfs.utils.datamodel.ag import SourceDetectionFlag

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_focus_errors
from agActor.utils.data import query_agc_data
from agActor.utils.math import semi_axes

logger = logging.getLogger(__name__)


def focus(
    *,
    frame_id: int | None = None,
    detected_objects: pd.DataFrame | None = None,
    max_ellipticity: float = 2.0e0,
    max_size: float = 1.0e12,
    min_size: float = -1.0e0,
    **kwargs,
):
    """Calculate focus error from detected objects in an auto-guider frame.

    Parameters
    ----------
    frame_id : int | None
        Frame ID to use for retrieving detected objects. Either frame_id or detected_objects must be provided.
    detected_objects : pd.DataFrame | None
        DataFrame of detected objects. Either frame_id or detected_objects must be provided.
    max_ellipticity : float
        Maximum ellipticity for source filtering, by default 2.0e0
    max_size : float
        Maximum size for source filtering, by default 1.0e12
    min_size : float
        Minimum size for source filtering, by default -1.0e0
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    dz : float
        Median focus error across all auto-guider cameras.
    dzs : NDArray[np.float64]
        Array of focus errors for each auto-guider camera.
    """
    if frame_id is None and detected_objects is None:
        raise ValueError("Either frame_id or detected_objects must be provided.")

    if frame_id is not None and detected_objects is None:
        logger.info(f"In focus, getting detected objects for {frame_id=}")
        detected_objects = query_agc_data(frame_id)

    logger.info(f"In focus with {max_ellipticity=}, {max_size=}, {min_size=}")

    _detected_objects = np.array(
        [
            (
                row["agc_camera_id"] + 1,  # camera_id (1-6)
                row["spot_id"],  # spot_id
                0,  # centroid_x (unused)
                0,  # centroid_y (unused)
                0,  # flux (unused)
                *semi_axes(
                    row["central_image_moment_11_pix"],
                    row["central_image_moment_20_pix"],
                    row["central_image_moment_02_pix"],
                ),  # semi-major and semi-minor axes
                row["flags"],  # flags
            )
            for idx, row in detected_objects.query(f'flags <= {SourceDetectionFlag.RIGHT.value}').iterrows()
        ]
    )

    num_objects = _detected_objects.shape[0]

    if num_objects == 0:
        logger.warning("No valid detected objects found, returning nan focus values.")
        return np.nan, np.full(6, np.nan)

    logger.info(f"Calculating focus errors from {num_objects} detected objects")

    try:
        dzs = calculate_focus_errors(
            _detected_objects, maxellip=max_ellipticity, maxsize=max_size, minsize=min_size
        )
        dz = np.nanmedian(dzs)
    except Exception as e:
        logger.error(f"Error calculating focus errors, returning nan values: {e}")
        dz = np.nan
        dzs = np.full(6, np.nan)

    logger.info(f"Focus values: {dz=} {dzs=}")
    return dz, dzs
