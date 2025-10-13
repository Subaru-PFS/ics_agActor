import logging

from pfs.utils.datamodel.ag import SourceDetectionFlag

from agActor import field_acquisition
from agActor.utils.data import (
    BAD_DETECTION_FLAGS,
    GuideCatalog,
    GuideOffsets,
    get_detected_objects,
    get_telescope_status,
)

logger = logging.getLogger(__name__)


def get_exposure_offsets(
    *,
    frame_id: int,
    guide_catalog: GuideCatalog,
    obswl: float = 0.62,
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
        obswl (float): Observation wavelength in microns, used for atmospheric refraction
            calculations, by default 0.62 microns.
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

    filter_flags = BAD_DETECTION_FLAGS
    filter_bad_shape = kwargs.get("filter_bad_shape", False)
    if filter_bad_shape:
        filter_flags = filter_flags | SourceDetectionFlag.BAD_SHAPE
    else:
        filter_flags = filter_flags & ~SourceDetectionFlag.BAD_SHAPE
    detected_objects = get_detected_objects(frame_id, filter_flags=filter_flags)

    ra = guide_catalog.ra
    dec = guide_catalog.dec
    inst_pa = guide_catalog.inst_pa

    _kwargs = field_acquisition.filter_kwargs(kwargs)
    logger.info(f"_kwargs={_kwargs}")

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
        **_kwargs,
    )

    guide_offsets.ra = ra
    guide_offsets.dec = dec
    guide_offsets.inst_pa = inst_pa

    return guide_offsets
