import logging
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from pfs.utils.coordinates import coordinates
from pfs.utils.datamodel.ag import SourceDetectionFlag

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_offsets
from agActor.utils import to_altaz
from agActor.utils.data import BAD_DETECTION_FLAGS, GuideOffsets, get_detected_objects, get_guide_objects
from agActor.utils.math import semi_axes

logger = logging.getLogger(__name__)

# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
_KEYMAP = {
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
    design_id: int,
    frame_id: int,
    visit0: int | None = None,
    obswl: float = 0.62,
    altazimuth: bool = True,
    is_guide: bool = False,
    **kwargs: Any,
) -> GuideOffsets:
    """Perform the necessary steps for field acquisition, including which guide stars are used.

    Parameters
    ----------
    design_id: int
        The design ID to retrieve guide stars for.
    frame_id : int
        The frame ID to use for detected objects and telescope status.
    visit0 : int
        The visit ID to retrieve guide stars from the pfs_config_agc table. If not
        provided, use the pfsDesign file with transformations.
    obswl : float, optional
        Observation wavelength in nm, defaults to 0.62.
    altazimuth : bool, optional
        Also return the AltAz coordinates in degrees, defaults to True.
    is_guide : bool, optional
        If we should filter the guide objects for acquisition, defaults to False.
        Should almost always be False for this function but provided here as a convenience.
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
    logger.info(
        f"Calling acquire_field with {frame_id=}, {obswl=}, {altazimuth=} {kwargs=}"
    )

    # Get the detected_objects, which will raise an exception if no valid spots are detected.
    filter_flags = BAD_DETECTION_FLAGS
    filter_bad_shape = kwargs.get("filter_bad_shape", False)
    if filter_bad_shape:
        filter_flags = filter_flags | SourceDetectionFlag.BAD_SHAPE
    else:
        filter_flags = filter_flags & ~SourceDetectionFlag.BAD_SHAPE
    detected_objects = get_detected_objects(frame_id, filter_flags=filter_flags)
    logger.info(f"Detected objects: {len(detected_objects)}")

    parse_kwargs(kwargs)

    guide_catalog = get_guide_objects(design_id=design_id, visit0=visit0, is_guide=is_guide, **kwargs)
    guide_objects = guide_catalog.guide_objects
    ra = guide_catalog.ra
    dec = guide_catalog.dec
    inr = guide_catalog.inr
    inst_pa = guide_catalog.inst_pa
    m2_pos3 = guide_catalog.m2_pos3
    adc = guide_catalog.adc
    taken_at = guide_catalog.taken_at
    logger.info(f"Using {ra=},{dec=},{inst_pa=}")

    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
        logger.info(f"ra modified by dra: {ra=}")
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
        logger.info(f"ddec modified by ddec: {dec=}")
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
        logger.info(f"inst_pa modified by dpa: {inst_pa=}")
    if "dinr" in kwargs:
        inr += kwargs.get("dinr") / 3600
        logger.info(f"inr modified by dinr: {inr=}")

    logger.info(f"Final values for calculating offsets: {ra=},{dec=},{inst_pa=},{inr=}")


    logger.info(f"Calling calculate_guide_offsets with {adc=}")

    _kwargs = filter_kwargs(kwargs)

    guide_offsets = get_guide_offsets(
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
        **_kwargs,
    )

    return guide_offsets


def get_guide_offsets(
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
    max_ellipticity: float = 2.0e0,
    max_size: float = 1.0e12,
    min_size: float = -1.0e0,
    max_residual: float = 0.5,
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
    max_ellipticity : float, optional
        Maximum ellipticity for source filtering, by default 2.0e0.
    max_size : float, optional
        Maximum size for source filtering, by default 1.0e12.
    min_size : float, optional
        Minimum size for source filtering, by default -1.0e0.
    max_residual : float, optional
        Maximum residual for source filtering, by default 0.5.
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
    # Get the RA/Dec in the detector plane from the centroid XY values for guide objects.
    logger.info("Calculating detector plane coordinates for detected objects")
    dp_ra, dp_dec = coordinates.det2dp(
        detected_objects["agc_camera_id"],
        detected_objects["centroid_x_pix"],
        detected_objects["centroid_y_pix"],
    )

    # Get the semi-major and semi-minor axes for the guide objects.
    logger.info("Calculating major and minor semi axes for detected objects")
    semi_major, semi_minor = semi_axes(
        detected_objects["central_image_moment_11_pix"],
        detected_objects["central_image_moment_20_pix"],
        detected_objects["central_image_moment_02_pix"],
    )

    # Put the detected objects in a format for the calculate_acquisition_offsets.
    logger.info("Building detected objects array for calculations")
    _detected_objects = np.array(
        [
            detected_objects["agc_camera_id"],
            detected_objects["spot_id"],
            dp_ra,
            dp_dec,
            detected_objects["peak_intensity"],
            semi_major,
            semi_minor,
            detected_objects["flags"],
        ]
    ).T

    # Convert taken_at to a UTC datetime object if it's not already
    if isinstance(taken_at, datetime):
        obstime = taken_at.astimezone(tz=timezone.utc)
    elif isinstance(taken_at, Number):
        obstime = datetime.fromtimestamp(taken_at, tz=timezone.utc)
    else:
        obstime = taken_at

    logger.info(f"Using {obstime=} for calculating offsets")

    logger.info(f"Calling calculate_acquisition_offsets (old FAinstpa)")

    (
        ra_offset,
        dec_offset,
        inr_offset,
        scale_offset,
        match_results,
        median_distance,
        valid_detections,
        good_guide_objects,
    ) = calculate_offsets(
        guide_objects=guide_objects,
        detected_array=_detected_objects,
        tel_ra=ra,
        tel_de=dec,
        dt=obstime,
        adc=adc,
        instpa=inst_pa,
        m2pos3=m2_pos3,
        wl=obswl,
        max_ellipticity=max_ellipticity,
        max_size=max_size,
        min_size=min_size,
        max_residual=max_residual,
    )
    ra_offset *= 3600
    dec_offset *= 3600
    inr_offset *= 3600
    logger.info(
        f"calculate_acquisition_offsets (old FAinstpa): "
        f"{ra_offset=},{dec_offset=},{inr_offset=},{scale_offset=}"
    )

    dalt = None
    daz = None
    if altazimuth:
        alt, az, dalt, daz = to_altaz.to_altaz(
            ra, dec, taken_at, dra=ra_offset, ddec=dec_offset
        )
        logger.info(f"{alt=},{az=},{dalt=},{daz=}")

    # Add the detector plane coordinates to the guide objects.
    guide_x_dp, guide_y_dp = coordinates.det2dp(
        guide_objects.agc_camera_id, guide_objects.x, guide_objects.y
    )
    guide_objects["x_dp"] = guide_x_dp
    guide_objects["y_dp"] = guide_y_dp

    # 0 obj_x          1
    # 1 obj_y          2
    # 2 catalog_x      3
    # 3 catalog_y      4
    # 4 err_x
    # 5 err_y
    # 6 resid_x
    # 7 resid_y
    # 8 valid_resid    5
    # 9 min_dist_index 0

    match_results_df = pd.DataFrame(
        match_results,
        columns=(
            "detected_object_x_mm",
            "detected_object_y_mm",
            "guide_object_x_mm",
            "guide_object_y_mm",
            "err_x",
            "err_y",
            "resid_x",
            "resid_y",
            "matched",
            "guide_object_id",
        ),
    )
    match_results_df.index = detected_objects[valid_detections].index
    match_results_df.index.name = "detected_object_id"
    match_results_df.reset_index(inplace=True)
    matched_guide_idx = match_results_df.guide_object_id.values
    match_results_df.guide_object_id = good_guide_objects.iloc[matched_guide_idx].index

    match_results_df["agc_camera_id"] = detected_objects.loc[
        match_results_df.detected_object_id
    ]["agc_camera_id"].values

    guide_x_pix, guide_y_pix = coordinates.dp2det(
        match_results_df["agc_camera_id"],
        match_results_df["guide_object_x_mm"],
        match_results_df["guide_object_y_mm"],
    )

    detected_x_pix, detected_y_pix = coordinates.dp2det(
        match_results_df["agc_camera_id"],
        match_results_df["detected_object_x_mm"],
        match_results_df["detected_object_y_mm"],
    )

    match_results_df["guide_object_x_pix"] = guide_x_pix
    match_results_df["guide_object_y_pix"] = guide_y_pix

    match_results_df["detected_object_x_pix"] = detected_x_pix
    match_results_df["detected_object_y_pix"] = detected_y_pix

    identified_objects = match_results_df[
        [
            "detected_object_id",
            "guide_object_id",
            "detected_object_x_mm",
            "detected_object_y_mm",
            "guide_object_x_mm",
            "guide_object_y_mm",
            "detected_object_x_pix",
            "detected_object_y_pix",
            "guide_object_x_pix",
            "guide_object_y_pix",
            "agc_camera_id",
            "matched",
        ]
    ]

    logger.info(
        f"Identified objects: {len(identified_objects)} Number valid: {len(identified_objects.query('matched == 1'))}"
    )
    if len(identified_objects) == 0:
        logger.warning(f"No detected objects detected, offsets will be zero.")

    # convert to arcsec
    logger.info(
        f"Converting ra_offset, dec_offset to arcsec: {ra_offset=},{dec_offset=}"
    )
    dx = -ra_offset * np.cos(np.deg2rad(dec))  # arcsec
    dy = dec_offset  # arcsec (HSC definition)
    logger.info(f"Converting offsets to pixels: {dx=},{dy=}")

    flux, peak, size = find_representative_spot(detected_objects, identified_objects)
    logger.info(f"Calculated star stats: {flux=},{peak=},{size=}")

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
    ag_plate_scale: float = (206.265 * 13) / 15000,
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
    ag_plate_scale : float
        The plate scale for the AG cameras. `13` is micron per pixel and 15000 mm is
        the focal length at prime focus.

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
    # squares of pointing errors in detector plane coordinates
    esq = (
        identified_objects["detected_object_x_mm"]
        - identified_objects["guide_object_x_mm"]
    ) ** 2 + (
        identified_objects["detected_object_y_mm"]
        - identified_objects["guide_object_y_mm"]
    ) ** 2
    n = len(esq) - np.isnan(esq).sum()
    if n > 0:
        # index of "median" of identified objects
        i = np.argpartition(esq, n // 2)[n // 2]

        # index of "median" of detected objects
        k = identified_objects["detected_object_id"][i]

        a, b = semi_axes(
            detected_objects["central_image_moment_11_pix"][k],
            detected_objects["central_image_moment_20_pix"][k],
            detected_objects["central_image_moment_02_pix"][k],
        )

        size = ag_plate_scale * 2 * (a * b) ** 0.5
        peak = detected_objects["peak_intensity"][k]
        flux = detected_objects["image_moment_00_pix"][k]

    return flux, peak, size
