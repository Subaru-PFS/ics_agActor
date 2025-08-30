from datetime import datetime, timezone
from logging import Logger
from numbers import Number
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from astropy.table import Table
from numpy.lib import recfunctions as rfn
from pfs.utils.coordinates import coordinates
from pfs.utils.datamodel.ag import AutoGuiderStarMask

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_acquisition_offsets
from agActor.utils import to_altaz
from agActor.utils.data import GuideOffsets, get_detected_objects, get_guide_objects
from agActor.utils.logging import log_message
from agActor.utils.math import semi_axes

# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
_KEYMAP = {
    "fit_dinr": ("inrflag", int),
    "fit_dscale": ("scaleflag", int),
    "max_ellipticity": ("maxellip", float),
    "max_size": ("maxsize", float),
    "min_size": ("minsize", float),
    "max_residual": ("maxresid", float),
    "max_correction": ("max_correction", float),
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
        kwargs.setdefault(
            "dpa", optional[0] if len(optional) > 0 else kwargs.get("dinr", 0)
        )
        kwargs.setdefault(
            "dinr",
            optional[1]
            if len(optional) > 1
            else optional[0]
            if len(optional) > 0
            else 0,
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


def acquire_field(
    *,
    frame_id: int,
    obswl: float = 0.62,
    altazimuth: bool = False,
    logger: Logger | None = None,
    **kwargs: Any,
) -> GuideOffsets:
    """Perform the necessary steps for field acquisition, including which guide stars are used.

    The guide stars are looked up in three different ways:

    1. If the `design_path` is provided, the PfsDesign file is used. This is the preferred method.
    2. If the `design_path` is not provided, the `design_id` is used to query the opdb.
    3. If neither the `design_id` nor `design_path` are provided, the gaia database is queried using the RA/Dec.

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
    GuideOffsets
        A dataclass containing:
        - ra (float): Right ascension of the field in degrees
        - dec (float): Declination of the field in degrees
        - inst_pa (float): Instrument position angle in degrees
        - ra_offset (float): Right ascension offset in arcseconds
        - dec_offset (float): Declination offset in arcseconds
        - inr_offset (float): Instrument rotator offset in arcseconds
        - scale_offset (float): Scale change
        - dalt (float): Altitude offset in arcseconds if altazimuth is True, else None
        - daz (float): Azimuth offset in arcseconds if altazimuth is True, else None
        - guide_objects (np.ndarray): Guide objects with additional calculated fields
        - detected_objects (np.ndarray): Detected objects from the frame
        - identified_objects (np.ndarray): Matched guide and detected objects
        - dx (float): X offset in arcseconds
        - dy (float): Y offset in arcseconds
        - size (float): Representative spot size in pixels
        - peak (float): Representative peak intensity
        - flux (float): Representative flux

    Notes
    -----
    If kwargs is missing any values, they will be provided by a lookup in the `get_telescope_status` function.
    """
    log_message(
        logger,
        f"Calling acquire_field with {frame_id=}, {obswl=}, {altazimuth=} {kwargs=}",
    )

    # Get the detected_objects, which will raise an exception if no valid spots are detected.
    detected_objects = get_detected_objects(frame_id)
    log_message(logger, f"Detected objects: {len(detected_objects)=}")

    parse_kwargs(kwargs)

    guide_object_results = get_guide_objects(
        frame_id, obswl=obswl, logger=logger, **kwargs
    )
    guide_objects = guide_object_results.guide_objects
    ra = guide_object_results.ra
    dec = guide_object_results.dec
    inr = guide_object_results.inr
    inst_pa = guide_object_results.inst_pa
    m2_pos3 = guide_object_results.m2_pos3
    adc = guide_object_results.adc
    taken_at = guide_object_results.taken_at
    log_message(logger, f"Using {ra=},{dec=},{inst_pa=}")
    log_message(
        logger, f"Got {len(guide_objects)} guide objects before guide filtering."
    )
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

    log_message(
        logger, f"Final values for calculating offsets: {ra=},{dec=},{inst_pa=},{inr=}"
    )

    _kwargs = filter_kwargs(kwargs)

    log_message(logger, f"Calling calculate_guide_offsets with {_kwargs=}")

    guide_offsets = calculate_guide_offsets(
        guide_objects,
        detected_objects,
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

    return guide_offsets


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
) -> GuideOffsets:
    """Calculate guide offsets for the detected objects using the guide objects from the catalog.

    This function calculates the guide offsets (right ascension, declination, and rotation)
    by matching detected objects with guide objects from the catalog. It also calculates
    scale changes and converts coordinates between different reference frames.

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
    GuideOffsets
        A dataclass containing:
        - ra (float): Right ascension of the field in degrees
        - dec (float): Declination of the field in degrees
        - inst_pa (float): Instrument position angle in degrees
        - ra_offset (float): Right ascension offset in arcseconds
        - dec_offset (float): Declination offset in arcseconds
        - inr_offset (float): Instrument rotator offset in arcseconds
        - scale_offset (float): Scale change
        - dalt (float): Altitude offset in arcseconds if altazimuth is True, else None
        - daz (float): Azimuth offset in arcseconds if altazimuth is True, else None
        - guide_objects (NDArray): Guide objects with additional calculated fields
        - detected_objects (NDArray): Detected objects from the frame
        - identified_objects (NDArray): Matched guide and detected objects
        - dx (float): X offset in arcseconds
        - dy (float): Y offset in arcseconds
        - size (float): Representative spot size in pixels
        - peak (float): Representative peak intensity
        - flux (float): Representative flux
    """

    # RA, Dec, Magnitude for guide objects.
    _guide_objects = np.array([(x[1], x[2], x[3]) for x in guide_objects])

    # Get the RA/Dec in the detector plane from the centroid XY values for guide objects.
    dp_ra, dp_dec = coordinates.det2dp(
        detected_objects["camera_id"],
        detected_objects["centroid_x"],
        detected_objects["centroid_y"],
    )

    # Get the semi-major and semi-minor axes for the guide objects.
    semi_major, semi_minor = semi_axes(
        detected_objects["central_moment_11"],
        detected_objects["central_moment_20"],
        detected_objects["central_moment_02"],
    )

    # Put the detected objects in a format for the calculate_acquisition_offsets.
    _detected_objects = np.array(
        [
            detected_objects["camera_id"],
            detected_objects["spot_id"],
            dp_ra,
            dp_dec,
            detected_objects["peak"],
            semi_major,
            semi_minor,
            detected_objects["flags"],
        ]
    ).T

    _kwargs = _map_kwargs(kwargs)

    # Convert taken_at to a UTC datetime object if it's not already
    if isinstance(taken_at, datetime):
        obstime = taken_at.astimezone(tz=timezone.utc)
    elif isinstance(taken_at, Number):
        obstime = datetime.fromtimestamp(taken_at, tz=timezone.utc)
    else:
        obstime = taken_at

    log_message(
        logger, f"Calling calculate_acquisition_offsets (old FAinstpa) with {_kwargs=}"
    )
    (
        ra_offset,
        dec_offset,
        inr_offset,
        scale_offset,
        match_results,
        median_distance,
        valid_sources,
    ) = calculate_acquisition_offsets(
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
        alt, az, dalt, daz = to_altaz.to_altaz(
            ra, dec, taken_at, dra=ra_offset, ddec=dec_offset
        )
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
                    detected_objects[k]["camera_id"], float(x[3]), float(x[4])
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
    log_message(
        logger,
        f"Converting ra_offset, dec_offset to arcsec: {ra_offset=},{dec_offset=}",
    )
    dx = -ra_offset * np.cos(np.deg2rad(dec))  # arcsec
    dy = dec_offset  # arcsec (HSC definition)
    log_message(logger, f"{dx=},{dy=}")

    flux, peak, size = find_representative_spot(detected_objects, identified_objects)

    return GuideOffsets(
        ra=ra,
        dec=dec,
        inst_pa=inst_pa,
        ra_offset=ra_offset,
        dec_offset=dec_offset,
        inr_offset=inr_offset,
        scale_offset=scale_offset,
        dalt=dalt,
        daz=daz,
        guide_objects=guide_objects,
        detected_objects=detected_objects,
        identified_objects=identified_objects,
        dx=dx,
        dy=dy,
        size=size,
        peak=peak,
        flux=flux,
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
    size = 0  # pix
    peak = 0  # pix
    flux = 0  # pix
    esq = (
        identified_objects["detected_object_x"] - identified_objects["guide_object_x"]
    ) ** 2 + (
        identified_objects["detected_object_y"] - identified_objects["guide_object_y"]
    ) ** 2  # squares of pointing errors in detector plane coordinates
    n = len(esq) - np.isnan(esq).sum()
    if n > 0:
        i = np.argpartition(esq, n // 2)[
            n // 2
        ]  # index of "median" of identified objects
        k = identified_objects["detected_object_id"][
            i
        ]  # index of "median" of detected objects
        a, b = semi_axes(
            detected_objects["central_moment_11"][k],
            detected_objects["central_moment_20"][k],
            detected_objects["central_moment_02"][k],
        )
        size = (a * b) ** 0.5
        peak = detected_objects["peak"][k]
        flux = detected_objects["moment_00"][k]

    return flux, peak, size


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
            guide_objects_df.loc[galaxy_idx, "filtered_by"] = (
                AutoGuiderStarMask.GALAXY.value
            )
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
                    not_filtered = (
                        guide_objects_df.filtered_by != AutoGuiderStarMask.GALAXY
                    )
                    include_filter = (guide_objects_df.flag.values & f) == 0
                    to_be_filtered = (include_filter & not_filtered) != 0
                    guide_objects_df.loc[to_be_filtered, "filtered_by"] |= f.value
                    log_message(
                        logger,
                        f"Filtering by {f.name}, removes {to_be_filtered.sum()} guide objects.",
                    )

                log_message(
                    logger,
                    f"After filtering, {len(guide_objects_df.query('filtered_by == 0'))} guide objects remain.",
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
        guide_objects = rfn.merge_arrays(
            (guide_objects, filterFlag_column), asrecarray=True, flatten=True
        )
    else:
        log_message(logger, "No filtering applied, using all guide objects.")
        # If not present, we need to add zero entries for the guide_star_flag and filter_flag.
        guideStarFlag_column = np.zeros(
            len(guide_objects), dtype=[("guideStarFlag", "<i4")]
        )
        guide_objects = rfn.merge_arrays(
            (guide_objects, guideStarFlag_column), asrecarray=True, flatten=True
        )

        filterFlag_column = np.zeros(len(guide_objects), dtype=[("filterFlag", "<i4")])
        guide_objects = rfn.merge_arrays(
            (guide_objects, filterFlag_column), asrecarray=True, flatten=True
        )

    return guide_objects
