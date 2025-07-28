from datetime import datetime, timezone
from logging import Logger
from numbers import Number
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from astropy.table import Table
from numpy import dtype, ndarray
from numpy.lib import recfunctions as rfn
from pfs.utils.coordinates import coordinates
from pfs.utils.datamodel.ag import AutoGuiderStarMask

from agActor.catalog import gen2_gaia as gaia
from agActor.catalog.pfs_design import pfsDesign as pfs_design
from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_acquisition_offsets
from agActor.utils import to_altaz
from agActor.utils.logging import log_message
from agActor.utils.math import semi_axes
from agActor.utils.opdb import opDB as opdb

# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
_KEYMAP = {
    "fit_dinr": ("inrflag", int),
    "fit_dscale": ("scaleflag", int),
    "max_ellipticity": ("maxellip", float),
    "max_size": ("maxsize", float),
    "min_size": ("minsize", float),
    "max_residual": ("maxresid", float),
}


def filter_kwargs(kwargs):

    return {k: v for k, v in kwargs.items() if k in _KEYMAP}


def _map_kwargs(kwargs):

    return {_KEYMAP[k][0]: _KEYMAP[k][1](v) for k, v in kwargs.items() if k in _KEYMAP}


def parse_kwargs(kwargs):

    if (center := kwargs.pop("center", None)) is not None:
        ra, dec, *optional = center
        kwargs.setdefault("ra", ra)
        kwargs.setdefault("dec", dec)
        kwargs.setdefault("inst_pa", optional[0] if len(optional) > 0 else 0)
    if (offset := kwargs.pop("offset", None)) is not None:
        dra, ddec, *optional = offset
        kwargs.setdefault("dra", dra)
        kwargs.setdefault("ddec", ddec)
        kwargs.setdefault("dpa", optional[0] if len(optional) > 0 else kwargs.get("dinr", 0))
        kwargs.setdefault(
            "dinr", optional[1] if len(optional) > 1 else optional[0] if len(optional) > 0 else 0
        )
    if (design := kwargs.pop("design", None)) is not None:
        design_id, design_path = design
        kwargs.setdefault("design_id", design_id)
        kwargs.setdefault("design_path", design_path)
    if (status_id := kwargs.pop("status_id", None)) is not None:
        visit_id, sequence_id = status_id
        kwargs.setdefault("visit_id", visit_id)
        kwargs.setdefault("sequence_id", sequence_id)
    if (tel_status := kwargs.pop("tel_status", None)) is not None:
        _, _, inr, adc, m2_pos3, _, _, _, taken_at = tel_status
        kwargs.setdefault("taken_at", taken_at)
        kwargs.setdefault("inr", inr)
        kwargs.setdefault("adc", adc)
        kwargs.setdefault("m2_pos3", m2_pos3)


def get_tel_status(*, frame_id, logger=None, **kwargs):
    """Get the telescope status information for a specific frame ID.

    Args:
        frame_id: The frame ID to retrieve telescope status for.
        logger: Optional logger instance for logging messages.
        **kwargs: Additional keyword arguments that may include:
            taken_at: Timestamp when the frame was taken
            inr: Instrument rotator angle
            adc: Atmospheric dispersion corrector value
            m2_pos3: Secondary mirror position
            sequence_id: Sequence ID for querying telescope status

    Returns:
        tuple: A tuple containing:
            taken_at: Timestamp when the frame was taken
            inr: Instrument rotator angle in degrees
            adc: Atmospheric dispersion corrector value
            m2_pos3: Secondary mirror position in mm

    If any of the optional values are not provided in kwargs, they will be retrieved
    from the opdb database using the frame_id and sequence_id (if provided).
    """
    log_message(logger, f"Getting telescope status for {frame_id=}")

    # Extract values from kwargs if provided
    taken_at = kwargs.get("taken_at")
    inr = kwargs.get("inr")
    adc = kwargs.get("adc")
    m2_pos3 = kwargs.get("m2_pos3")

    # Check if we need to fetch any missing values from the database
    if any(value is None for value in (taken_at, inr, adc, m2_pos3)):
        # First, query the agc_exposure table to get basic information, including visit_id.
        log_message(logger, f"Getting agc_exposure from opdb for frame_id={frame_id}")
        visit_id, _, db_taken_at, _, _, db_inr, db_adc, _, _, _, db_m2_pos3 = opdb.query_agc_exposure(
            frame_id
        )

        # If sequence_id is provided, get more accurate information from tel_status table
        sequence_id = kwargs.get("sequence_id")
        if sequence_id is not None:
            log_message(logger, f"Getting telescope status from opdb for {visit_id=},{sequence_id=}")
            _, _, db_inr, db_adc, db_m2_pos3, _, _, _, _, db_taken_at = opdb.query_tel_status(
                visit_id, sequence_id
            )

        # Use database values for any missing parameters
        taken_at = taken_at or db_taken_at
        inr = inr or db_inr
        adc = adc or db_adc
        m2_pos3 = m2_pos3 or db_m2_pos3

    log_message(logger, f"{taken_at=},{inr=},{adc=},{m2_pos3=}")
    return taken_at, inr, adc, m2_pos3


def acquire_field(
    *,
    frame_id: int,
    obswl: float = 0.62,
    altazimuth: bool = False,
    logger: Logger | None = None,
    **kwargs: Any,
) -> tuple[
    Any | None,
    Any | None,
    Any | None,
    float,
    float,
    float,
    float,
    float,
    float,
    Any,
    Any,
    Any,
    float,
    float,
    float,
    float,
    float,
]:
    """Perform the necessary steps for field acquisition, including which guide stars are used.

    The guide stars are looked up in three different ways:

    1. If the `design_path` is provided, the PfsDesign file is used. This is the preferred method.
    2. If the `design_path` is not provided, the `design_id` is used to query the opdb.
    3. If neither the `design_id` nor `design_path` are provided, the gaia database is queried using the RA/Dec.

    TODO: Replace return values with a dataclass.

    Parameters
    ----------
    frame_id : int
        The frame ID to retrieve telescope status for.
    obswl : float, optional
        Observation wavelength in nm, defaults to 0.62.
    altazimuth : bool, optional
        Also return the AltAz coordinates in degrees, defaults to False.
    logger : object, optional
        Optional logger instance for logging messages.
    **kwargs : dict
        Additional keyword arguments that may include:

        - taken_at : datetime or float
            Timestamp when the frame was taken
        - inr : float
            Instrument rotator angle in degrees
        - adc : float
            Atmospheric dispersion corrector value
        - m2_pos3 : float
            Secondary mirror position in mm
        - sequence_id : int
            Sequence ID for querying telescope status

    Returns
    -------
    tuple
        A tuple containing (in order):
        ra : float
            Right ascension of field center in degrees
        dec : float
            Declination of field center in degrees
        inst_pa : float
            Instrument position angle in degrees
        ra_offset : float
            Right ascension offset in arcseconds
        dec_offset : float
            Declination offset in arcseconds
        inr_offset : float
            Instrument rotator offset in arcseconds
        scale_offset : float
            Scale change factor
        dalt : float or None
            Altitude offset in arcseconds if altazimuth=True, else None
        daz : float or None
            Azimuth offset in arcseconds if altazimuth=True, else None
        guide_objects : ndarray
            Structured array of guide objects with calculated fields
        detected_objects : ndarray
            Structured array of detected objects
        identified_objects : ndarray
            Structured array of matched guide and detected objects
        dx : float
            Offset in x direction in arcseconds
        dy : float
            Offset in y direction in arcseconds
        size : float
            Representative spot size in pixels
        peak : float
            Representative peak intensity
        flux : float
            Representative flux

    Notes
    -----
    If kwargs is missing any values, they will be provided by a lookup in the `get_tel_status` function.
    """
    log_message(logger, f"Calling acquire_field with {frame_id=}, {obswl=}, {altazimuth=}")
    parse_kwargs(kwargs)
    taken_at, inr, adc, m2_pos3 = get_tel_status(frame_id=frame_id, logger=logger, **kwargs)
    log_message(logger, f"Getting detected objects from opdb.agc_data for {frame_id=}")
    detected_objects = opdb.query_agc_data(frame_id)

    # This creates a list of only the valid objects, where d[-1] is the `flag` column.
    # The flag should either be zero (left side of detector) or one (right side with glass).
    valid_detected_objects = [d for d in detected_objects if d[-1] <= 1]

    if len(valid_detected_objects) == 0:
        raise RuntimeError("No valid spots detected, can't compute offsets")

    log_message(logger, f"Got {len(detected_objects)} detected objects")

    design_id = kwargs.get("design_id")
    design_path = kwargs.get("design_path")
    log_message(logger, f"{design_id=},{design_path=}")
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")

    if design_path is None and design_id is None:
        log_message(logger, "No design_id or design_path provided, getting guide objects from gaia db.")
        guide_objects, *_ = gaia.get_objects(
            ra=ra, dec=dec, obstime=taken_at, inst_pa=inst_pa, adc=adc, m2pos3=m2_pos3, obswl=obswl
        )
    else:
        if design_path is not None:
            log_message(
                logger, f"Getting guide objects from pfs design file via {design_path=} and {design_id=}"
            )
            guide_objects, _ra, _dec, _inst_pa = pfs_design(
                design_id, design_path, logger=logger
            ).guide_objects(obstime=taken_at)
        else:
            log_message(logger, f"No design_path provided, getting guide objects from opdb via {design_id=}")
            _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
            guide_objects = opdb.query_pfs_design_agc(design_id)

        ra = ra or _ra
        dec = dec or _dec
        inst_pa = inst_pa or _inst_pa

    log_message(logger, f"Using {ra=},{dec=},{inst_pa=}")

    log_message(logger, f"Got {len(guide_objects)} guide objects before filtering.")
    guide_objects = filter_guide_objects(guide_objects, logger)

    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
        log_message(logger, f"ra modified by dra: {ra=}")
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
        log_message(logger, f"ddec modified by ddec: {dec=}")
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
        log_message(logger, f"inst_pa modified by dpa: {inst_pa=}")
    if "dinr" in kwargs:
        inr += kwargs.get("dinr") / 3600
        log_message(logger, f"inr modified by dinr: {inr=}")

    log_message(logger, f"Final values for calculating offsets: {ra=},{dec=},{inst_pa=},{inr=}")

    _kwargs = filter_kwargs(kwargs)

    log_message(logger, f"Calling calculate_guide_offsets with {_kwargs=}")

    (
        ra_offset,
        dec_offset,
        inr_offset,
        scale_offset,
        dalt,
        daz,
        guide_objects,
        detected_objects,
        identified_objects,
        dx,
        dy,
        size,
        peak,
        flux,
    ) = calculate_guide_offsets(
        guide_objects,
        valid_detected_objects,
        ra,
        dec,
        taken_at,
        adc,
        inst_pa,
        m2_pos3=m2_pos3,
        obswl=obswl,
        altazimuth=altazimuth,
        logger=logger,
        **_kwargs,
    )

    return (
        ra,
        dec,
        inst_pa,
        ra_offset,
        dec_offset,
        inr_offset,
        scale_offset,
        dalt,
        daz,
        guide_objects,
        detected_objects,
        identified_objects,
        dx,
        dy,
        size,
        peak,
        flux,
    )


def calculate_guide_offsets(
    guide_objects: Sequence[Tuple[Any, ...]],
    detected_objects: Sequence[Tuple[Any, ...]],
    ra: float,
    dec: float,
    taken_at: datetime | float | Any,
    adc: Sequence[float],
    inst_pa: float = 0.0,
    m2_pos3: float = 6.0,
    obswl: float = 0.62,
    altazimuth: bool = False,
    logger: Logger | None = None,
    **kwargs: Dict[str, Any],
) -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any | None,
    Any | None,
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    Any,
    Any,
    int | float | complex | Any,
    int | ndarray[Any, dtype[Any]],
    int | ndarray[Any, dtype[Any]],
]:
    """Calculate guide offsets for the detected objects using the guide objects from the catalog.

    This function calculates the guide offsets (right ascension, declination, and rotation)
    by matching detected objects with guide objects from the catalog. It also calculates
    scale changes and converts coordinates between different reference frames.

    TODO: Replace return values with a dataclass.

    Parameters
    ----------
    guide_objects : array-like
        Array of guide objects from the catalog with their coordinates and properties.
    detected_objects : array-like
        Array of objects detected in the image with their coordinates and properties.
    ra : float
        Right ascension of the field center in degrees.
    dec : float
        Declination of the field center in degrees.
    taken_at : datetime or float
        Time when the image was taken, either as a datetime object or a timestamp.
    adc : array-like
        ADC (Atmospheric Dispersion Corrector) parameters.
    inst_pa : float, optional
        Instrument position angle in degrees. Default is 0.0.
    m2_pos3 : float, optional
        M2 mirror position along the optical axis in mm. Default is 6.0.
    obswl : float, optional
        Observation wavelength in microns. Default is 0.62.
    altazimuth : bool, optional
        If True, convert offsets to altitude-azimuth coordinates. Default is False.
    logger : object, optional
        Logger object for logging messages. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the field acquisition and focusing calculation.

    Returns
    -------
    ra_offset : float
        Right ascension offset in arcseconds.
    dec_offset : float
        Declination offset in arcseconds.
    inr_offset : float
        Rotation offset in arcseconds.
    scale_offset : float
        Scale change.
    dalt: float
        altitude offset in arcseconds if altazimuth is True, else None.
    daz: float
        azimuth offset in arcseconds if altazimuth is True, else None.
    guide_objects:
        structured array of guide objects with additional calculated fields
    detected_objects:
        structured array of detected objects
    identified_objects:
        structured array of matched guide and detected objects
    dx: float
        offset in x direction in arcseconds.
    dy: float
        offsets in y directions in arcseconds.
    size: float
        representative spot size in pixels.
    peak: float
        representative peak intensity.
    flux: float
        representative flux.
    """

    # RA, Dec, Magnitude
    _guide_objects = np.array([(x[1], x[2], x[3]) for x in guide_objects])

    _detected_objects = np.array(
        [
            (
                x[0],
                x[1],
                *coordinates.det2dp(int(x[0]), x[3], x[4]),
                x[10],
                *semi_axes(x[5], x[6], x[7]),
                x[-1],
            )
            for x in detected_objects
        ]
    )
    _kwargs = _map_kwargs(kwargs)

    # Convert taken_at to a UTC datetime object if it's not already
    if isinstance(taken_at, datetime):
        obstime = taken_at.astimezone(tz=timezone.utc)
    elif isinstance(taken_at, Number):
        obstime = datetime.fromtimestamp(taken_at, tz=timezone.utc)
    else:
        obstime = taken_at

    log_message(logger, f"Calling calculate_acquisition_offsets (old FAinstpa) with {_kwargs=}")
    ra_offset, dec_offset, inr_offset, scale_offset, match_results, median_distance, valid_sources = (
        calculate_acquisition_offsets(
            _guide_objects,
            _detected_objects,
            ra,
            dec,
            obstime,
            adc,
            inst_pa,
            m2_pos3,
            obswl,
            **_kwargs,
        )
    )
    ra_offset *= 3600
    dec_offset *= 3600
    inr_offset *= 3600
    log_message(
        logger,
        f"From calculate_acquisition_offsets (old FAinstpa) "
        f"{ra_offset=},{dec_offset=},{inr_offset=},{scale_offset=}",
    )

    dalt = None
    daz = None
    if altazimuth:
        alt, az, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=ra_offset, ddec=dec_offset)
        log_message(logger, f"{alt=},{az=},{dalt=},{daz=}")
    guide_objects = np.array(
        [
            (
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                x[6],
                *coordinates.det2dp(x[4], float(x[5]), float(x[6])),
                x[7],  # guide_star_flag
                x[8],  # filter_flag
            )
            for x in guide_objects
        ],
        dtype=[
            ("source_id", np.int64),  # u8 (80) not supported by FITSIO
            ("ra", np.float64),
            ("dec", np.float64),
            ("mag", np.float32),
            ("camera_id", np.int16),
            ("guide_object_xdet", np.float32),
            ("guide_object_ydet", np.float32),
            ("guide_object_x", np.float32),
            ("guide_object_y", np.float32),
            ("guide_star_flag", np.int32),  # guide star flag
            ("filter_flag", np.int32),  # filter flag
        ],
    )
    detected_objects = np.array(
        detected_objects,
        dtype=[
            ("camera_id", np.int16),
            ("spot_id", np.int16),
            ("moment_00", np.float32),
            ("centroid_x", np.float32),
            ("centroid_y", np.float32),
            ("central_moment_11", np.float32),
            ("central_moment_20", np.float32),
            ("central_moment_02", np.float32),
            ("peak_x", np.uint16),
            ("peak_y", np.uint16),
            ("peak", np.uint16),
            ("background", np.float32),
            ("flags", np.uint8),
        ],
    )

    (index_v,) = np.where(valid_sources)
    identified_objects = np.array(
        [
            (
                k,  # index of detected objects
                int(x[0]),  # index of identified guide object
                float(x[1]),
                float(x[2]),  # detector plane coordinates of detected object
                float(x[3]),
                float(x[4]),  # detector plane coordinates of identified guide object
                *coordinates.dp2det(
                    detected_objects[k][0], float(x[3]), float(x[4])
                ),  # detector coordinates of identified guide object
            )
            for k, x in (
                (int(index_v[i]), x)
                for i, x in enumerate(
                    zip(
                        match_results[:, 9],
                        match_results[:, 0],
                        match_results[:, 1],
                        match_results[:, 2],
                        match_results[:, 3],
                        match_results[:, 8],
                    )
                )
                if int(x[5])
            )
        ],
        dtype=[
            ("detected_object_id", np.int16),
            ("guide_object_id", np.int16),
            ("detected_object_x", np.float32),
            ("detected_object_y", np.float32),
            ("guide_object_x", np.float32),
            ("guide_object_y", np.float32),
            ("guide_object_xdet", np.float32),
            ("guide_object_ydet", np.float32),
        ],
    )

    # convert to arcsec
    log_message(logger, f"Converting ra_offset, dec_offset to arcsec: {ra_offset=},{dec_offset=}")
    dx = -ra_offset * np.cos(np.deg2rad(dec))  # arcsec
    dy = dec_offset  # arcsec (HSC definition)
    log_message(logger, f"{dx=},{dy=}")

    flux, peak, size = find_representative_spot(detected_objects, identified_objects)

    return (
        ra_offset,
        dec_offset,
        inr_offset,
        scale_offset,
        dalt,
        daz,
        guide_objects,
        detected_objects,
        identified_objects,
        dx,
        dy,
        size,
        peak,
        flux,
    )


def find_representative_spot(
    detected_objects: np.ndarray,
    identified_objects: np.ndarray,
) -> tuple[float, float, float]:
    """Find representative flux, peak intensity, and spot size by median of pointing errors.

    Parameters
    ----------
    detected_objects : np.ndarray
        Structured array containing detected star data including:
        - central_moment_11 : float
            Cross moment of detected object
        - central_moment_20 : float
            Second moment in x direction
        - central_moment_02 : float
            Second moment in y direction
        - peak : float
            Peak intensity value
        - moment_00 : float
            Total flux
    identified_objects : np.ndarray
        Structured array containing matched guide and detected object data with:
        - detected_object_x : float
            X coordinate of detected object
        - detected_object_y : float
            Y coordinate of detected object
        - guide_object_x : float
            X coordinate of guide object
        - guide_object_y : float
            Y coordinate of guide object
        - detected_object_id : int
            Index linking to detected_objects array

    Returns
    -------
    tuple[float, float, float]
        flux : float
            Representative flux.
        peak : float
            Representative peak intensity in pixels.
        size : float
            Representative spot size in pixels, calculated as sqrt(a*b) where
            a,b are the semi-major and semi-minor axes.

    Notes
    -----
    Representative values are taken from the detected object with median pointing error.
    If no valid objects are found, return (0,0,0).

    Also note that this doesn't appear to be working correctly as we were seeing
    strange values during observations. This function and its intent should be revisited.
    """
    flux = 0.0
    peak = 0.0  # pix
    size = 0.0  # pix

    # squares of pointing errors in detector plane coordinates.
    pointing_error_x = identified_objects["detected_object_x"] - identified_objects["guide_object_x"]
    pointing_error_y = identified_objects["detected_object_y"] - identified_objects["guide_object_y"]
    pointing_errors_squared = pointing_error_x**2 + pointing_error_y**2

    # Identify valid (non-NaN) pointing errors
    # Create a boolean mask for non-NaN values
    valid_mask = ~np.isnan(pointing_errors_squared)

    # Filter for only valid pointing errors
    valid_pointing_errors_squared = pointing_errors_squared[valid_mask]

    num_valid = len(valid_pointing_errors_squared)

    if num_valid == 0:
        # No valid objects found, return default zeros
        return flux, peak, size

    # Find the index of the median pointing error within the *valid* errors
    median_idx_in_valid = np.argpartition(valid_pointing_errors_squared, num_valid // 2)[num_valid // 2]

    # Get the actual index in the *original* identified_objects array
    # This involves mapping back from the valid_mask
    original_indices_of_valid = np.where(valid_mask)[0]
    median_original_idx = original_indices_of_valid[median_idx_in_valid]

    # Get the detected_object_id corresponding to the median pointing error
    k = identified_objects["detected_object_id"][median_original_idx]

    median_detected_object = detected_objects[k]

    cm_11 = median_detected_object["central_moment_11"]
    cm_20 = median_detected_object["central_moment_20"]
    cm_02 = median_detected_object["central_moment_02"]

    # Calculate spot size using semi_axes helper function.
    a, b = semi_axes(cm_11, cm_20, cm_02)
    size = (a * b) ** 0.5

    # Extract peak intensity and flux
    peak = median_detected_object["peak"]
    flux = median_detected_object["moment_00"]

    return float(flux), float(peak), float(size)


def filter_guide_objects(guide_objects, logger=None, initial=False):
    """Apply filtering to the guide objects based on their flags.

    This function filters guide objects based on various quality flags. For initial
    coarse guiding, it only filters out galaxies. For fine guiding (initial=False),
    it applies additional filters to ensure high quality guide stars from GAIA.

    Parameters
    ----------
    guide_objects : numpy.ndarray
        Structured array containing guide object data including flags.
        Must have fields for basic star data (objId, ra, dec, mag, etc.)
        and optionally a 'flag' field for filtering.
    logger : logging.Logger, optional
        Logger instance for debug messages. If None, no logging occurs.
    initial : bool, optional
        If True, only filter galaxies. If False (default), apply all quality filters
        including GAIA star requirements.

    Returns
    -------
    numpy.ndarray
        Filtered guide objects array with additional columns:
        - guideStarFlag: int32 flag for guide star status
        - filterFlag: int32 indicating which filters were applied

    Notes
    -----
    The filtering adds two columns to track which objects were filtered and why:
    - guideStarFlag: Legacy column, currently always 0
    - filterFlag: Bitwise combination of AutoGuiderStarMask values indicating
      which filters removed the object. 0 means the object passed all filters.
    """
    # Apply filtering if we have a flag column.
    have_flags = "flag" in guide_objects.dtype.names
    guide_objects_df = None
    if have_flags:
        log_message(logger, "Applying filters to guide objects")
        try:
            # Use Table to convert, which handles big-endian and little-endian issues.
            guide_objects_df = Table(guide_objects).to_pandas()
            log_message(logger, f"Got {len(guide_objects_df)} guide objects.")

            column_names = ["objId", "ra", "dec", "mag", "agId", "agX", "agY", "flag"]
            guide_objects_df.columns = column_names

            # Add a column to indicate which flat was used for filtering.
            guide_objects_df["filtered_by"] = 0

            # Filter the guide objects to only include the ones that are not flagged as galaxies.
            log_message(logger, "Filtering guide objects to remove galaxies.")
            galaxy_idx = (guide_objects_df.flag.values & AutoGuiderStarMask.GALAXY) != 0
            guide_objects_df.loc[galaxy_idx, "filtered_by"] = AutoGuiderStarMask.GALAXY.value
            log_message(
                logger,
                f"Filtering by {AutoGuiderStarMask.GALAXY.name}, removes {galaxy_idx.sum()} guide objects.",
            )

            # The initial coarse guide uses all the stars and the fine guide uses only the GAIA stars.
            if initial is False:
                filters_for_inclusion = [
                    AutoGuiderStarMask.GAIA,
                    AutoGuiderStarMask.NON_BINARY,
                    AutoGuiderStarMask.ASTROMETRIC,
                    AutoGuiderStarMask.PMRA_SIG,
                    AutoGuiderStarMask.PMDEC_SIG,
                    AutoGuiderStarMask.PARA_SIG,
                    AutoGuiderStarMask.PHOTO_SIG,
                ]

                # Go through the filters and mark which stars would be flagged as NOT meeting the mask requirement.
                for f in filters_for_inclusion:
                    not_filtered = guide_objects_df.filtered_by != AutoGuiderStarMask.GALAXY
                    include_filter = (guide_objects_df.flag.values & f) == 0
                    to_be_filtered = (include_filter & not_filtered) != 0
                    guide_objects_df.loc[to_be_filtered, "filtered_by"] |= f.value
                    log_message(
                        logger, f"Filtering by {f.name}, removes {to_be_filtered.sum()} guide objects."
                    )

                log_message(
                    logger,
                    f'After filtering, {len(guide_objects_df.query("filtered_by == 0"))} guide objects remain.',
                )
        except Exception as e:
            log_message(logger, f"Error filtering guide objects: {e}", level="WARNING")
            log_message(logger, "No filtering applied, using all guide objects.")
    else:
        log_message(logger, "No filtering applied, using all guide objects.")
    # Add the column that indicates what was filtered.
    if guide_objects_df is not None:
        log_message(logger, "Adding filter flag column to guide objects.")
        filterFlag_column = guide_objects_df.filtered_by.to_numpy("<i4")
        filterFlag_column = np.array(filterFlag_column, dtype=[("filterFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, filterFlag_column), asrecarray=True, flatten=True)
    else:
        log_message(logger, "No filtering applied, using all guide objects.")
        # If not present, we need to add zero entries for the guide_star_flag and filter_flag.
        guideStarFlag_column = np.zeros(len(guide_objects), dtype=[("guideStarFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, guideStarFlag_column), asrecarray=True, flatten=True)

        filterFlag_column = np.zeros(len(guide_objects), dtype=[("filterFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, filterFlag_column), asrecarray=True, flatten=True)

    return guide_objects
