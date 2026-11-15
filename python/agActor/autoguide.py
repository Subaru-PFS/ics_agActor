import logging

import numpy as np

from agActor import field_acquisition
from agActor.utils.data import (
    GuideCatalog,
    GuideOffsets,
    get_detected_objects,
    get_telescope_status,
)

logger = logging.getLogger(__name__)


def sky_ra_offset(dra: float, dec: float) -> float:
    """The RA offset on the sky, ``dra cos(dec)``, for a coordinate offset ``dra`` (arcsec) at ``dec`` (deg)."""
    return float(dra * np.cos(np.deg2rad(dec)))


def offset_in_range(dra: float, ddec: float, dec: float, max_correction: float) -> bool:
    """Is a measured guide offset small enough to send to the telescope?

    ``dra`` is a coordinate offset (arcsec of right ascension, as ``GuideOffsets.ra_offset``,
    ``guideErrors.dRA`` and ``agc_guide_offset.guide_delta_ra`` all are), so it is compared
    on the sky, as ``dra cos(dec)``, and ``max_correction`` means the same distance at every
    declination (INSTRM-2931: compared raw, a 10 arcsec limit was 4 arcsec on the sky at Dec 66).

    Parameters
    ----------
    dra, ddec : `float`
        The measured offsets, arcsec; ``dra`` in RA coordinate.
    dec : `float`
        Declination of the field, degrees.
    max_correction : `float`
        The largest offset, on the sky, that will be sent, arcsec.
    """
    return bool(abs(sky_ra_offset(dra, dec)) < max_correction and abs(ddec) < max_correction)


def get_exposure_offsets(
    *,
    frame_id: int,
    guide_catalog: GuideCatalog,
    cameras: list[int] | None = None,
    obswl: float = 0.62,
    max_ellipticity: float = 2.0e0,
    max_size: float = 1.0e12,
    min_size: float = -1.0e0,
    max_residual: float = 0.5,
    **kwargs,
):
    """Calculate guiding corrections based on detected objects in a frame.

    The function performs the following steps:
    1. Gets telescope status information for the frame.
    2. Retrieves detected objects from the frame.
    3. Applies any coordinate adjustments specified in kwargs.
    4. Calculates guiding offsets by matching detected objects with guide objects.

    Parameters:
        frame_id (int): Frame ID to use for retrieving detected objects and telescope status.
        guide_catalog (GuideCatalog): Guide objects and field center information.
        cameras (list[int] or None): Optional list of one-indexed camera ids whose
            detections are allowed to contribute to the astrometric fit.
            Detections from all cameras are still fetched; only the fit is
            restricted.  If None, all cameras contribute to the fit.
        obswl (float): Observation wavelength in microns, used for atmospheric refraction
            calculations, by default 0.62 microns.
        max_ellipticity (float): Maximum ellipticity for source filtering, by default 2.0e0.
        max_size (float): Maximum size for source filtering, by default 1.0e12.
        min_size (float): Minimum size for source filtering, by default -1.0e0.
        max_residual (float): Maximum residual for source filtering, by default 0.5.
        **kwargs: Additional keyword arguments that may include:
            - dra, ddec: Small adjustments to coordinates in arcseconds
            - dpa: Small adjustment to position angle in arcseconds
            - dinr: Small adjustment to instrument rotator angle in arcseconds

    Returns:
        GuideOffsets: A dataclass containing:
            - ra (float): Right ascension of the field in degrees
            - dec (float): Declination of the field in degrees
            - inst_pa (float): Instrument position angle in degrees
            - ra_offset (float): Right ascension offset in arcseconds
            - dec_offset (float): Declination offset in arcseconds
            - inr_offset (float): Instrument rotator offset in arcseconds
            - scale_offset (float): Scale offset
            - dalt (float): Altitude offset in arcseconds
            - daz (float): Azimuth offset in arcseconds
            - guide_objects (np.ndarray): Guide objects used for calculations
            - detected_objects (np.ndarray): Detected objects from the frame
            - identified_objects (np.ndarray): Matched guide and detected objects
            - dx (float): X offset in arcseconds
            - dy (float): Y offset in arcseconds
            - size (float): Representative spot size
            - peak (float): Representative peak intensity
            - flux (float): Representative flux
    """

    logger.info(f"Calling autoguide.autoguide with frame_id={frame_id}")
    field_acquisition.parse_kwargs(kwargs)

    logger.info("Getting telescope status")
    taken_at, inr, adc, m2_pos3 = get_telescope_status(frame_id=frame_id, **kwargs)

    # Get the detected_objects, which will raise an exception if no valid spots are detected.

    detected_objects = get_detected_objects(frame_id)

    ra = guide_catalog.ra
    dec = guide_catalog.dec
    inst_pa = guide_catalog.inst_pa

    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
        logger.info(f"ra modified by dra: {ra=}")
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
        logger.info(f"dec modified by ddec: {dec=}")
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
        logger.info(f"inst_pa modified by dpa: {inst_pa=}")
    if "dinr" in kwargs:
        # Note: inr is updated but calculate_guide_offsets uses inst_pa for matching.
        inr += kwargs.get("dinr") / 3600
        logger.info(f"inr modified by dinr: {inr=}")

    logger.info(
        "Calling field_acquisition.calculate_guide_offsets from autoguide.get_exposure_offsets"
    )
    guide_offsets = field_acquisition.get_guide_offsets(
        guide_catalog.guide_objects,
        detected_objects,
        ra,
        dec,
        taken_at,
        adc,
        inst_pa,
        m2_pos3=m2_pos3,
        obswl=obswl,
        altazimuth=True,
        max_ellipticity=max_ellipticity,
        max_size=max_size,
        min_size=min_size,
        max_residual=max_residual,
        enabled_camera_ids=cameras,
    )

    guide_offsets.ra = ra
    guide_offsets.dec = dec
    guide_offsets.inst_pa = inst_pa

    return guide_offsets
