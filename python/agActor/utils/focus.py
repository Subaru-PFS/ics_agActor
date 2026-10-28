import logging

import numpy as np
import pandas as pd
from pfs.utils.datamodel.ag import SourceDetectionFlag

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_focus_errors
from agActor.utils.data import query_agc_data
from agActor.utils.math import semi_axes

logger = logging.getLogger(__name__)


def focus_objects(detected_objects: pd.DataFrame, enabledCameras: list[int] | None = None) -> pd.DataFrame:
    """The detections the focus calculation uses: those from enabled cameras with no flag but RIGHT.

    Parameters
    ----------
    detected_objects : pd.DataFrame
        Detected objects, with zero-based ``agc_camera_id`` and ``flags``.
    enabledCameras : list[int] | None
        One-indexed AG camera IDs to use. If None, all cameras are used.
    """
    if enabledCameras is not None:
        try:
            # agc_camera_id in the DataFrame is zero-based; enabledCameras is one-based.
            enabled_set = {int(camera_id) - 1 for camera_id in enabledCameras}
            detected_objects = detected_objects[detected_objects["agc_camera_id"].isin(enabled_set)]
        except (TypeError, ValueError) as e:
            # Only catch conversion errors from malformed enabledCameras values.
            # Other errors (e.g. missing column) should propagate so they are not masked.
            logger.warning(
                f"Failed to filter detected objects with enabled cameras {enabledCameras}: {e}; using all cameras",
                exc_info=True,
            )

    return detected_objects.query(f"flags <= {SourceDetectionFlag.RIGHT.value}")


def explain_missing_focus(
    detected_objects: pd.DataFrame, dzs, enabledCameras: list[int] | None = None
) -> str:
    """Say why each enabled camera has no focus value.

    A camera's focus comes from the difference between the stars on its two halves (the glass
    was removed from the RIGHT half), so it needs at least one star on each.

    Parameters
    ----------
    detected_objects : pd.DataFrame
        The detected objects passed to `focus`.
    dzs : array-like
        The per-camera focus errors `focus` returned, AG1 to AG6.
    enabledCameras : list[int] | None
        One-indexed AG camera IDs used. If None, all cameras are used.

    Returns
    -------
    str
        e.g. ``"AG2: 0 stars on the right half (4 on the left); AG5: no stars"``, or ``""``
        when every enabled camera has a focus value.
    """
    objects = focus_objects(detected_objects, enabledCameras)
    right = (objects["flags"].astype(int) & int(SourceDetectionFlag.RIGHT)) != 0
    try:
        cameras = range(1, 7) if enabledCameras is None else sorted(int(c) for c in enabledCameras)
    except (TypeError, ValueError):
        cameras = range(1, 7)  # as focus_objects falls back to all cameras
    reasons = []
    for camera_id in cameras:
        if np.isfinite(dzs[camera_id - 1]):
            continue
        on_camera = objects["agc_camera_id"] == camera_id - 1
        n_right = int((on_camera & right).sum())
        n_left = int((on_camera & ~right).sum())
        if n_left == 0 and n_right == 0:
            reasons.append(f"AG{camera_id}: no stars")
        elif n_right == 0:
            reasons.append(f"AG{camera_id}: 0 stars on the right half ({n_left} on the left)")
        elif n_left == 0:
            reasons.append(f"AG{camera_id}: 0 stars on the left half ({n_right} on the right)")
        else:
            reasons.append(f"AG{camera_id}: no focus from {n_left} left and {n_right} right stars")
    return "; ".join(reasons)


def focus(
    *,
    frame_id: int | None = None,
    detected_objects: pd.DataFrame | None = None,
    enabledCameras: list[int] | None = None,
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
    enabledCameras : list[int] | None
        One-indexed AG camera IDs to use. If None, all cameras are used.
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
            for idx, row in focus_objects(detected_objects, enabledCameras).iterrows()
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
    missing = explain_missing_focus(detected_objects, dzs, enabledCameras)
    if missing:
        logger.warning(f"No focus value from some cameras: {missing}")
    return dz, dzs
