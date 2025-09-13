import logging
import warnings
from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pfs.utils.coordinates import Subaru_POPT2_PFS

from agActor.coordinates import Subaru_POPT2_PFS_AG

logger = logging.getLogger(__name__)


def calculate_focus_errors(
    agarray: NDArray[np.float64],
    maxellip: float = 2.0e00,
    maxsize: float = 1.0e12,
    minsize: float = -1.0e00,
) -> NDArray[np.float64]:
    """
    Calculate focus errors from auto-guider array data.

    This method processes auto-guider array data to determine focus errors.

    Parameters
    ----------
    agarray : NDArray[np.float64]
        Auto-guider array data
    maxellip : float, optional
        Maximum ellipticity for source filtering, by default 2.0e+00
    maxsize : float, optional
        Maximum size for source filtering, by default 1.0e+12
    minsize : float, optional
        Minimum size for source filtering, by default -1.0e+00

    Returns
    -------
    NDArray[np.float64]
        Array of focus errors for each position
    """
    pfs = Subaru_POPT2_PFS_AG.PFS()

    # TODO there is assumption that six AG cameras are always present (and in for loop below).
    moment_differences = pfs.agarray2momentdifference(
        agarray, maxellip, maxsize, minsize
    )

    focus_errors = np.full(6, np.nan)
    for idx in range(6):
        focus_errors[idx] = pfs.momentdifference2focuserror(moment_differences[idx])

    return focus_errors


def calculate_offsets(
    guide_objects: pd.DataFrame,
    detected_array: NDArray[np.float64],
    tel_ra: float,
    tel_de: float,
    dt: Any,
    adc: Any,
    instpa: float,
    m2pos3: Any,
    wl: Any,
    inrflag: int = 1,
    scaleflag: int = 1,
    maxellip: float = 0.6,
    maxsize: float = 20.0,
    minsize: float = 0.92,
    maxresid: float = 0.5,
) -> Tuple[float, float, float, float, NDArray[np.float64], float, int]:
    """
    Perform field acquisition calculations with the instrument position angle.

    This method calculates the offsets in right ascension, declination,
    instrument rotation, and scale based on the comparison between
    catalog and detected star positions.

    Parameters
    ----------
    guide_objects : pd.DataFrame
        DataFrame containing catalog star information for the guide objects
    detected_array : NDArray[np.float64]
        Array containing detected star information
    tel_ra : float
        Telescope right ascension in degrees
    tel_de : float
        Telescope declination in degrees
    dt : Any
        Date/time information
    adc : Any
        Atmospheric dispersion corrector information
    instpa : float
        Instrument position angle in degrees
    m2pos3 : Any
        Secondary mirror position information
    wl : Any
        Wavelength information
    inrflag : int, optional
        Flag for instrument rotation calculation, by default 1
    scaleflag : int, optional
        Flag for scale calculation, by default 1
    maxellip : float, optional
        Maximum ellipticity for source filtering, by default 0.6
    maxsize : float, optional
        Maximum size for source filtering, by default 20.0
    minsize : float, optional
        Minimum size for source filtering, by default 0.92
    maxresid : float, optional
        Maximum residual for source filtering, by default 0.5

    Returns
    -------
    Tuple[float, float, float, float, NDArray[np.float64], float, int]
        Tuple containing:
        - Right ascension offset
        - Declination offset
        - Instrument rotation offset
        - Scale offset
        - Match results array
        - Median distance
        - Number of valid sources
    """
    subaru = Subaru_POPT2_PFS.Subaru()
    initial_inr = subaru.radec2inr(tel_ra, tel_de, dt)
    instrument_rotation = initial_inr + instpa

    pfs = Subaru_POPT2_PFS_AG.PFS()

    logger.info(f"Guide object before filtering: {len(guide_objects)}")
    try:
        good_guide_objects = guide_objects.query('filtered_by == 0')
    except KeyError:
        logger.info(f"guide_objects missing 'filtered_by' attribute, using all objects")
        good_guide_objects = guide_objects

    logger.info(f"Guide objects after filtering: {len(good_guide_objects)}")

    ra_values = good_guide_objects.ra.values
    dec_values = good_guide_objects.dec.values
    magnitude_values = good_guide_objects.mag.values

    basis_vector_0, basis_vector_1 = pfs.makeBasis(
        tel_ra, tel_de, ra_values, dec_values, dt, adc, instrument_rotation, m2pos3, wl
    )

    basis_vector_0 = np.insert(basis_vector_0, 2, magnitude_values, axis=1)
    basis_vector_1 = np.insert(basis_vector_1, 2, magnitude_values, axis=1)

    # Detected sources filtering
    logger.info(f"Filtering detected objects with {maxellip=} {maxsize=} {minsize=}")
    filtered_detected_array, valid_detections = pfs.sourceFilter(
        detected_array, maxellip, maxsize, minsize
    )
    logger.info(f"Found {len(filtered_detected_array)} detected sources after filtering")

    ra_offset, de_offset, inr_offset, scale_offset, match_results = pfs.RADECInRShiftA(
        filtered_detected_array[:, 2],
        filtered_detected_array[:, 3],
        filtered_detected_array[:, 4],
        filtered_detected_array[:, 7],
        basis_vector_0,
        basis_vector_1,
        inrflag,
        scaleflag,
        maxresid,
    )
    valid_resid_idx = match_results[:, 8] == 1.0
    logger.info(f"Matched sources with valid residuals: {len(match_results[valid_resid_idx])}")

    residual_squares = match_results[:, 6] ** 2 + match_results[:, 7] ** 2
    residual_squares[valid_resid_idx] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        try:
            median_distance = np.nanmedian(np.sqrt(residual_squares))
        except RuntimeWarning:
            logger.warning("All nan values for residuals, setting median_distance to np.nan")
            median_distance = np.nan

    return (
        ra_offset,
        de_offset,
        inr_offset,
        scale_offset,
        match_results,
        median_distance,
        valid_detections,
        good_guide_objects,
    )
