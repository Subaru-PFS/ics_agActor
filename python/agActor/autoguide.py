from agActor import field_acquisition
from agActor.catalog.pfs_design import pfsDesign as pfs_design
from agActor.utils.data import GuideOffsets, get_detected_objects, get_guide_objects, get_telescope_status
from agActor.utils.logging import log_message
from agActor.utils.opdb import opDB as opdb


class Field:

    design = None
    center = None
    guide_objects = None


def set_design(*, logger=None, **kwargs):

    field_acquisition.parse_kwargs(kwargs)
    design_id = kwargs.get("design_id")
    design_path = kwargs.get("design_path")
    log_message(logger, f"{design_id=},{design_path=}")
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")
    if any(x is None for x in (ra, dec, inst_pa)):
        if any(x is not None for x in (design_id, design_path)):
            if design_path is not None:
                log_message(logger, f"Setting psf_design via {design_path=}")
                _ra, _dec, _inst_pa = pfs_design(design_id, design_path, logger=logger).center
                log_message(logger, f"ra={_ra},dec={_dec},inst_pa={_inst_pa}")
            else:
                log_message(logger, f"Setting psf_design via {design_id=}")
                _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
                log_message(logger, f"ra={_ra},dec={_dec},inst_pa={_inst_pa}")
            if ra is None:
                ra = _ra
            if dec is None:
                dec = _dec
            if inst_pa is None:
                inst_pa = _inst_pa
    log_message(logger, f"ra={ra},dec={dec},inst_pa={inst_pa}")
    log_message(logger, f"Setting Field.design to {design_id=},{design_path=}")
    Field.design = design_id, design_path
    log_message(logger, f"Setting Field.center to {ra=},{dec=},{inst_pa=}")
    Field.center = ra, dec, inst_pa
    log_message(logger, f"Setting Field.guide_objects to empty list []")
    Field.guide_objects = []  # delay loading of guide objects


def set_design_agc(*, frame_id=None, obswl=0.62, logger=None, **kwargs):
    """Set up guide objects for the current field for autoguiding.

    This function retrieves guide objects from various sources and stores them in Field.guide_objects.
    It uses the field center coordinates (ra, dec, inst_pa) from Field.center and design information
    from Field.design if available. The guide objects are filtered to select the most suitable stars
    for autoguiding.

    The guide objects can come from different sources depending on the parameters:
    1. If frame_id is provided: Detected objects are retrieved from the operational database
       and processed using astrometry.
    2. If design_id/design_path are available: Guide objects are loaded from PFS design file
       or from the operational database.
    3. Otherwise: Guide objects are retrieved from the Gaia catalog based on the field center.

    Parameters:
        frame_id (int, optional): Frame ID to use for retrieving guide objects from the
            operational database. If provided, detected objects will be queried.
        obswl (float): Observation wavelength in microns. Used for atmospheric refraction
            calculations when retrieving guide objects, by default 0.62 microns.
        logger: Logger object for logging messages during execution.
        **kwargs: Additional keyword arguments that may include:
            - design_id: PFS design identifier
            - design_path: Path to PFS design file
            - dra, ddec, dinr: Small adjustments to coordinates in arcseconds
            - detected_objects: Can be provided directly instead of querying with frame_id

    Side Effects:
        Updates Field.guide_objects with the filtered guide objects.

    Notes:
        This function is typically called before autoguide() to prepare the guide objects
        that will be used for calculating guiding corrections.
    """
    log_message(logger, f"frame_id={frame_id}")
    field_acquisition.parse_kwargs(kwargs)

    # Get field center from Field.center
    ra, dec, inst_pa = Field.center
    log_message(logger, f"Field center: ra={ra}, dec={dec}, inst_pa={inst_pa}")

    # Add field center to kwargs
    kwargs["ra"] = ra
    kwargs["dec"] = dec
    kwargs["inst_pa"] = inst_pa

    # Add design information to kwargs
    if frame_id is None:
        design_id, design_path = Field.design
        log_message(logger, f"design_id={design_id}, design_path={design_path}")
        kwargs["design_id"] = design_id
        kwargs["design_path"] = design_path

    # If frame_id is provided, get detected objects
    if frame_id is not None:
        log_message(logger, f"Setting pfs_design_agc via frame_id={frame_id}")
        log_message(logger, f"Getting agc_data from opdb for frame_id={frame_id}")
        detected_objects = opdb.query_agc_data(frame_id)
        log_message(logger, f"Got {len(detected_objects)=} detected objects")
        kwargs["detected_objects"] = detected_objects

    # Get guide objects using the enhanced get_guide_objects function
    guide_object_results = get_guide_objects(frame_id=frame_id, obswl=obswl, logger=logger, **kwargs)

    guide_objects = guide_object_results.guide_objects
    log_message(logger, f"Got {len(guide_objects)} guide objects before filtering.")
    guide_objects = field_acquisition.filter_guide_objects(guide_objects, logger)

    log_message(logger, "Setting Field.guide_objects")
    Field.guide_objects = guide_objects


def autoguide(*, frame_id, obswl=0.62, logger=None, **kwargs):
    """Calculate guiding corrections based on detected objects in a frame.

    This function compares detected objects from a frame with the guide objects
    stored in Field.guide_objects to calculate guiding corrections. It uses the
    field center coordinates from Field.center and applies any small adjustments
    provided in kwargs.

    The function performs the following steps:
    1. Retrieves guide objects from Field.guide_objects (previously set by set_design_agc)
    2. Gets telescope status information for the frame
    3. Retrieves detected objects from the frame
    4. Applies any coordinate adjustments specified in kwargs
    5. Calculates guiding offsets by matching detected objects with guide objects

    Parameters:
        frame_id (int): Frame ID to use for retrieving detected objects and telescope status
        obswl (float): Observation wavelength in microns, used for atmospheric refraction
            calculations, by default 0.62 microns
        logger: Logger object for logging messages during execution
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

    Notes:
        This function requires that Field.guide_objects has been previously populated
        by calling set_design_agc. It uses field_acquisition.calculate_guide_offsets
        to perform the actual offset calculations.
    """

    log_message(logger, f"Calling autoguide.autoguide with frame_id={frame_id}")
    field_acquisition.parse_kwargs(kwargs)
    guide_objects = Field.guide_objects

    ra, dec, inst_pa = Field.center
    log_message(logger, f"ra={ra},dec={dec}")
    log_message(logger, "Getting telescope status")
    taken_at, inr, adc, m2_pos3 = get_telescope_status(frame_id=frame_id, logger=logger, **kwargs)

    # Get the detected_objects, which will raise an exception if no valid spots are detected.
    detected_objects = get_detected_objects(frame_id)

    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
    if "dinr" in kwargs:
        inr += kwargs.get("dinr") / 3600
    log_message(logger, f"ra={ra},dec={dec},inst_pa={inst_pa},inr={inr}")
    _kwargs = field_acquisition.filter_kwargs(kwargs)
    log_message(logger, f"_kwargs={_kwargs}")

    log_message(logger, "Calling field_acquisition.calculate_guide_offsets from autoguide.autoguide")
    guide_offsets = field_acquisition.calculate_guide_offsets(
        guide_objects,
        detected_objects,
        ra,
        dec,
        taken_at,
        adc,
        inst_pa,
        m2_pos3=m2_pos3,
        obswl=obswl,
        altazimuth=True,
        logger=logger,
        **_kwargs,
    )

    guide_offsets.ra = ra
    guide_offsets.dec = dec
    guide_offsets.inst_pa = inst_pa

    return guide_offsets
